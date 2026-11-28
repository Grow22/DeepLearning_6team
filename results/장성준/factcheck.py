#!/usr/bin/env python3
"""가짜 글 판정 도구: Claude가 웹 검색으로 뉴스 사진 설명글이 사실인지 확인한다 (기본: Claude Haiku 5.5).

글마다 Claude API에 한 번 물어본다. Claude는 웹 검색(글당 최대 2번)으로 원래 기사를 찾아
누가·무엇을·어디서·언제·숫자 같은 세부가 맞는지 비교하고, 판정(사실/거짓/근거 부족)과
거짓일 확률, 근거 링크를 남긴다. 글이 AI가 쓴 것처럼 보이는지(문체)는 판단에 쓰지 않게 한다.

실행 (저장소 맨 위 폴더에서, prepare_all.py로 데이터를 받은 뒤):
    pip install anthropic huggingface_hub pyarrow
    export ANTHROPIC_API_KEY=...                                   # 키는 저장소·채팅에 올리지 않는다
    python results/장성준/factcheck.py --set validation --limit 30  # 먼저 30개로 시험 (1달러 미만)
    python results/장성준/factcheck.py                              # 시험 글 1,660개 전체 (약 $40)
    python results/장성준/factcheck.py --report                     # 결과 표만 다시 보기

결과 (data/ 폴더는 깃에 올라가지 않는다):
    data/factcheck/<모델>-<세트>.jsonl   글마다 판정, 확률, 이유, 검색어, 근거 링크, 비용
중간에 멈춰도 다시 실행하면 끝난 글은 건너뛰고, 오류가 났던 글만 다시 한다.
쓴 금액이 --budget(기본 $45)에 닿으면 새 글을 더 보내지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MODEL = "claude-haiku-5-5"
# 100만 토큰당 달러 (요청 하나의 입력 토큰 상한, 입력, 출력). 하이쿠는 입력이 10만 토큰을 넘는 요청이 더 비싸다.
PRICES = {
    "claude-haiku-5-5": [(100_000, 0.10, 0.50), (None, 0.50, 2.50)],
    "claude-sonnet-5-5": [(None, 2.00, 10.00)],
    "claude-opus-5-5": [(None, 4.00, 20.00)],
}
CACHE_WRITE, CACHE_READ = 1.25, 0.10  # 캐시에 쓰는 입력·캐시에서 읽는 입력은 입력 값의 이 배수
SEARCH_PRICE = 10 / 1000              # 웹 검색 1번
MAX_SEARCHES = 2
SOURCE = {"test1_nyt_mj": "NYT", "test2_bbc_dalle": "BBC", "test3_cnn_dalle": "CNN",
          "test4_bbc_sdxl": "BBC", "test5_cnn_sdxl": "CNN", "validation": "NYT"}
VERDICTS = ("TRUE", "FALSE", "INSUFFICIENT")
VERDICT_KO = {"TRUE": "사실", "FALSE": "거짓", "INSUFFICIENT": "근거 부족"}
SYSTEM = (
    "You are a careful fact-checker for news photo captions.\n"
    "You will receive one caption. Decide whether what it states is factually accurate.\n\n"
    "How to check:\n"
    f"1. Use web search (at most {MAX_SEARCHES} searches) to find original news coverage of the event.\n"
    "2. Compare the caption's specific details with the sources: who, what action, where, when, numbers, "
    "and any added claims.\n"
    "3. TRUE: reliable sources confirm the event and the caption's key details.\n"
    "   FALSE: sources show the details are different (changed action, number, place, date or person, or "
    "invented claims), or the described event appears not to have happened.\n"
    "   INSUFFICIENT: you could not find enough evidence either way. Still give your best p_false.\n"
    "4. Judge only factual accuracy. Do not use writing style, or whether the text sounds machine-written, "
    "as evidence.\n\n"
    "End your answer with exactly one line of JSON and nothing after it:\n"
    '{"verdict": "TRUE" | "FALSE" | "INSUFFICIENT", "p_false": <probability from 0 to 1 that the caption is '
    'false>, "reason": "<one short sentence about what you found>"}'
)


def norm(text: str) -> str:
    return " ".join(text.split())


def text_id(text: str) -> str:
    return hashlib.sha1(norm(text).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 글 불러오기
# ---------------------------------------------------------------------------
def load_test_captions(data: Path) -> list[dict]:
    """교차 시험 세트에 나오는 글을 중복 없이 모은다 (시험 글 전부: 사실 830 + 거짓 830)."""
    path = data / "ai_photos" / "cross_test.csv"
    if not path.exists():
        raise SystemExit("cross_test.csv가 없어요. 먼저 python data_preparation/prepare_all.py 를 실행해 주세요.")
    by_text = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text = norm(r["text"])
            c = by_text.setdefault(text, {"id": text_id(text), "text": text, "label_fake": int(r["text_fake"]),
                                          "sources": set()})
            if c["label_fake"] != int(r["text_fake"]):
                raise SystemExit(f"같은 글에 정답이 둘이에요: {text[:60]}")
            c["sources"].add(SOURCE[r["split"]])
    return [{**c, "source": "/".join(sorted(c.pop("sources")))} for c in by_text.values()]


def load_validation_captions(limit: int, seed: int, rows=None) -> list[dict]:
    """MiRAGeNews validation 글에서 사실·거짓 글을 반반 뽑는다 (질문 문구를 시험해 볼 때)."""
    if rows is None:
        sys.path.insert(0, str(REPO_ROOT / "data_preparation" / "ai_photos"))
        import prepare_ai_photos as pp
        import pyarrow.parquet as pq
        meta = REPO_ROOT / "data_preparation" / "ai_photos" / "combined_test_pairs.meta.json"
        revision = json.loads(meta.read_text(encoding="utf-8"))["revision"]
        rows = []
        for name in pp.split_files(pp.hub_list_files(revision), "validation"):
            table = pq.read_table(pp.hub_download(revision, name), columns=["text", "label"])
            rows += list(zip(table.column("text").to_pylist(), table.column("label").to_pylist()))
    by_text = {}
    for text, label in rows:
        by_text.setdefault(norm(text), int(label))  # label 1 = AI 쪽 = 지어낸 글
    rng = random.Random(seed)
    picked = []
    for label in (0, 1):
        group = sorted(t for t, lab in by_text.items() if lab == label)
        picked += rng.sample(group, min(len(group), (limit + 1 - label) // 2 if limit else len(group)))
    return [{"id": text_id(t), "text": t, "label_fake": by_text[t], "source": "NYT"} for t in picked]


# ---------------------------------------------------------------------------
# Claude에게 묻기
# ---------------------------------------------------------------------------
def parse_verdict(text: str) -> dict | None:
    """답의 마지막 JSON에서 판정, 거짓일 확률, 이유를 꺼낸다. 없거나 이상하면 None."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        try:
            obj = json.loads(m.group(0))
            verdict = str(obj.get("verdict", "")).strip().upper()
            p_false = float(obj.get("p_false"))
        except (ValueError, TypeError, AttributeError):
            continue
        if verdict in VERDICTS and math.isfinite(p_false):
            return {"verdict": verdict, "p_false": min(1.0, max(0.0, p_false)),
                    "reason": str(obj.get("reason", ""))[:500]}
    return None


def request_cost(model: str, usage) -> float:
    """요청 하나의 비용(달러): 입력·출력 토큰 + 웹 검색 횟수."""
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    tool = getattr(usage, "server_tool_use", None)
    searches = (getattr(tool, "web_search_requests", 0) or 0) if tool else 0
    if model not in PRICES:
        raise SystemExit(f"{model}의 가격을 몰라요. PRICES에 넣어 주세요.")
    total_in = inp + write + read
    price_in, price_out = next((pi, po) for limit, pi, po in PRICES[model] if limit is None or total_in <= limit)
    return (inp * price_in + write * price_in * CACHE_WRITE + read * price_in * CACHE_READ
            + out * price_out) / 1e6 + searches * SEARCH_PRICE


def check_caption(client, model: str, text: str, effort: str | None = None, max_turns: int = 3) -> dict:
    """글 하나를 확인한다. 검색이 길어져 API가 잠시 멈추면(pause_turn) 받은 답을 그대로 돌려보내 이어 간다."""
    messages = [{"role": "user", "content": f"Caption: {text}"}]
    rec = {"queries": [], "urls": [], "citations": [], "search_errors": [], "searches": 0,
           "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}
    answer = []
    for _ in range(max_turns):
        kwargs = {"model": model, "max_tokens": 8000, "system": SYSTEM, "messages": messages,
                  "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": MAX_SEARCHES}]}
        if effort:
            kwargs["output_config"] = {"effort": effort}
        resp = client.messages.create(**kwargs)
        usage = resp.usage
        rec["calls"] += 1
        rec["cost"] += request_cost(model, usage)
        rec["input_tokens"] += (usage.input_tokens or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0) \
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        rec["output_tokens"] += usage.output_tokens or 0
        tool = getattr(usage, "server_tool_use", None)
        rec["searches"] += (getattr(tool, "web_search_requests", 0) or 0) if tool else 0
        for block in resp.content:
            if block.type == "server_tool_use" and isinstance(block.input, dict):
                rec["queries"].append(str(block.input.get("query", "")))
            elif block.type == "web_search_tool_result":
                if isinstance(block.content, list):
                    rec["urls"] += [r.url for r in block.content[:5]]
                else:
                    rec["search_errors"].append(getattr(block.content, "error_code", "unknown"))
            elif block.type == "text":
                answer.append(block.text)
                for c in block.citations or []:
                    if getattr(c, "url", None):
                        rec["citations"].append({"url": c.url, "title": getattr(c, "title", ""),
                                                 "cited_text": getattr(c, "cited_text", "")})
        rec["stop_reason"] = resp.stop_reason
        if resp.stop_reason != "pause_turn":
            break
        messages = messages + [{"role": "assistant", "content": resp.content}]
    rec["urls"] = list(dict.fromkeys(rec["urls"]))
    rec["answer"] = "\n".join(answer)[-1500:]
    parsed = parse_verdict("\n".join(answer))
    rec.update(parsed if parsed else {"error": f"판정을 읽지 못했어요 (stop_reason={rec['stop_reason']})"})
    return rec


# ---------------------------------------------------------------------------
# 여러 글 처리 (이어 하기, 예산 한도)
# ---------------------------------------------------------------------------
def load_records(path: Path) -> dict:
    """저장된 결과. 같은 글이 여러 번 있으면 마지막 것이 남는다."""
    records = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records[rec["id"]] = rec
    return records


def spent_so_far(path: Path) -> float:
    total = 0.0
    if path.exists():
        with open(path, encoding="utf-8") as f:
            total = sum(json.loads(line).get("cost", 0.0) for line in f if line.strip())
    return total


def run(captions: list[dict], client, model: str, out_path: Path, budget: float, workers: int = 4,
        effort: str | None = None, max_errors_in_a_row: int = 5) -> dict:
    """아직 안 한 글(또는 오류 난 글)만 확인한다. 쓴 금액이 budget에 닿으면 새 글을 보내지 않는다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_records(out_path)
    todo = [c for c in captions if c["id"] not in done or done[c["id"]].get("error")]
    state = {"spent": spent_so_far(out_path), "errors_in_a_row": 0, "stopped": "", "new": 0}
    lock = threading.Lock()
    started = time.time()

    def work(c):
        try:
            rec = check_caption(client, model, c["text"], effort)
        except Exception as e:  # noqa: BLE001  (요청 실패는 기록해 두고 다음에 다시 한다)
            rec = {"error": f"{type(e).__name__}: {e}"[:300], "cost": 0.0}
        rec.update({"id": c["id"], "text": c["text"], "label_fake": c["label_fake"], "source": c["source"],
                    "model": model, "effort": effort or "default",
                    "time": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["spent"] += rec.get("cost", 0.0)
            state["new"] += 1
            state["errors_in_a_row"] = state["errors_in_a_row"] + 1 if rec.get("error") else 0
            if state["new"] % 20 == 0 or state["new"] == len(todo):
                rate = (time.time() - started) / state["new"]
                print(f"  {state['new']}/{len(todo)}개 (지금까지 ${state['spent']:.2f}, "
                      f"남은 시간 약 {rate * (len(todo) - state['new']) / 60:.0f}분)", flush=True)
        return rec

    print(f"확인할 글 {len(todo)}개 (이미 끝난 글 {len(captions) - len(todo)}개), "
          f"지금까지 쓴 금액 ${state['spent']:.2f}, 한도 ${budget:.2f}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running = set()
        for c in todo:
            while len(running) >= workers:  # 자리가 날 때까지 기다린 뒤에 한도를 확인한다
                finished, running = wait(running, return_when=FIRST_COMPLETED)
            with lock:
                if state["spent"] >= budget:
                    state["stopped"] = f"예산 한도(${budget:.2f})에 닿아서 멈췄어요."
                elif state["errors_in_a_row"] >= max_errors_in_a_row:
                    state["stopped"] = f"오류가 {max_errors_in_a_row}번 이어져서 멈췄어요. 저장된 error를 확인해 주세요."
            if state["stopped"]:
                break
            running.add(pool.submit(work, c))
        wait(running)
    if state["stopped"]:
        print(state["stopped"], flush=True)
    return state


# ---------------------------------------------------------------------------
# 결과 표
# ---------------------------------------------------------------------------
def auc(labels, scores) -> float:
    sys.path.insert(0, str(HERE))
    import score_cards
    return score_cards.auc(labels, scores)


def report(records: list[dict], title: str) -> str:
    ok = [r for r in records if not r.get("error")]
    errors = len(records) - len(ok)
    cost = sum(r.get("cost", 0.0) for r in records)
    searches = sum(r.get("searches", 0) for r in records)

    def stats(rows):
        if not rows:
            return {"n": 0}
        y = [r["label_fake"] for r in rows]
        pred = [int(r["p_false"] >= 0.5) for r in rows]
        real = [p == 0 for p, t in zip(pred, y) if t == 0]
        fake = [p == 1 for p, t in zip(pred, y) if t == 1]
        return {"n": len(rows), "acc": sum(p == t for p, t in zip(pred, y)) / len(rows),
                "auc": auc(y, [r["p_false"] for r in rows]),
                "real": sum(real) / len(real) if real else float("nan"),
                "fake": sum(fake) / len(fake) if fake else float("nan")}

    def pct(x):
        return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"

    def num(x):
        return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"

    lines = [f"## {title}", "", "판정 기준: 거짓일 확률 0.5 이상이면 거짓", "",
             "| 출처 | 글 수 | 정확도 | AUC | 사실 글을 사실로 | 거짓 글을 거짓으로 |", "|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in ok:
        groups[r["source"]].append(r)
    for name in sorted(groups) + ["전체"]:
        s = stats(ok if name == "전체" else groups[name])
        bold = "**" if name == "전체" else ""
        lines.append(f"| {bold}{name}{bold} | {s['n']:,} | {pct(s.get('acc'))} | {num(s.get('auc'))} | "
                     f"{pct(s.get('real'))} | {pct(s.get('fake'))} |")
    counts = Counter((r["label_fake"], r["verdict"]) for r in ok)
    lines += ["", "**판정 분포**", "", "| 정답 | 사실 | 거짓 | 근거 부족 |", "|---|---|---|---|"]
    for label, name in ((0, "사실 글"), (1, "거짓 글")):
        lines.append(f"| {name} | " + " | ".join(f"{counts[(label, v)]:,}" for v in VERDICTS) + " |")
    lines += ["", f"비용: ${cost:.2f} (웹 검색 {searches:,}번, 글당 평균 ${cost / max(len(records), 1):.4f})"
              + (f" · 오류 {errors}개 (다시 실행하면 그 글만 다시 해요)" if errors else ""), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None, client=None, validation_rows=None) -> int:
    parser = argparse.ArgumentParser(description="가짜 글 판정 (Claude + 웹 검색)")
    parser.add_argument("--set", default="test", choices=["test", "validation"], dest="which")
    parser.add_argument("--limit", type=int, default=0, help="이만큼만 확인 (validation은 사실·거짓 반반)")
    parser.add_argument("--model", default=MODEL, choices=sorted(PRICES))
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                        help="생각하는 양 (기본: 모델 기본값, 하이쿠는 medium)")
    parser.add_argument("--budget", type=float, default=45.0, help="쓴 금액이 이 달러에 닿으면 멈춤")
    parser.add_argument("--workers", type=int, default=4, help="동시에 보내는 요청 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out", default=None, help="결과 파일 (기본: data/factcheck/<모델>-<세트>.jsonl)")
    parser.add_argument("--report", action="store_true", help="확인은 안 하고 결과 표만 보기")
    parser.add_argument("--yes", action="store_true", help="비용 확인 질문 없이 바로 시작")
    args = parser.parse_args(argv)

    data = Path(args.data)
    out = Path(args.out) if args.out else data / "factcheck" / f"{args.model}-{args.which}.jsonl"
    if args.which == "test":
        captions = load_test_captions(data)
        if args.limit:
            captions = random.Random(args.seed).sample(captions, min(args.limit, len(captions)))
    else:
        captions = load_validation_captions(args.limit or 30, args.seed, validation_rows)
    title = f"가짜 글 판정 결과 ({args.model}, {'시험' if args.which == 'test' else 'validation'} 글)"

    if not args.report:
        done = load_records(out)
        todo = [c for c in captions if c["id"] not in done or done[c["id"]].get("error")]
        estimate = len(todo) * (MAX_SEARCHES * SEARCH_PRICE + 0.004)
        if len(todo) > 50 and not args.yes:
            try:
                answer = input(f"글 {len(todo)}개를 확인해요. 최대 약 ${estimate:.0f}가 들 수 있어요 "
                               f"(한도 ${args.budget:.0f}). 시작할까요? [y/N] ")
            except EOFError:
                answer = ""
            if answer.strip().lower() not in ("y", "yes"):
                print("시작하지 않았어요.")
                return 1
        if client is None:
            import anthropic
            client = anthropic.Anthropic(max_retries=6)  # 키는 ANTHROPIC_API_KEY 환경 변수에서 읽는다
        run(captions, client, args.model, out, args.budget, args.workers, args.effort)

    wanted = {c["id"] for c in captions}
    records = [r for r in load_records(out).values() if r["id"] in wanted]
    if not records:
        print("아직 결과가 없어요.")
        return 0
    text = report(records, title)
    print(text)
    out.with_suffix(".md").write_text(text, encoding="utf-8")
    print(f"결과: {out} / 표: {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
