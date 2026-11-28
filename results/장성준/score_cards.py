#!/usr/bin/env python3
"""카드 편집 실험 채점: 판정 도구가 낸 점수로 ① 원본 ② 카드 ③ 잘라낸 사진의 결과를 split별로 낸다.

어떤 판정 도구든 아래 형식의 CSV만 만들면 채점할 수 있다. score는 클수록 AI라고 본 것이다.
    input,id,score
    original,test1_nyt_mj-00000,0.93     id = data/ai_photos/ai_photos.csv의 image_id
    card,card-test1_nyt_mj-0-250,0.41    id = data/cards/cards.csv의 card_id
    crop,card-test1_nyt_mj-0-250,0.88    crop = 카드에서 잘라낸 사진

실행: python results/장성준/score_cards.py data/univfd/scores.csv [--threshold 0.5] [--out 결과.md]
정답은 data/ 폴더의 표에서 가져온다 (점수 파일에 다른 칸이 있어도 쓰지 않는다).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TEST_SPLITS = ["test1_nyt_mj", "test2_bbc_dalle", "test3_cnn_dalle", "test4_bbc_sdxl", "test5_cnn_sdxl"]
SOURCE = {"test1_nyt_mj": "NYT · Midjourney", "test2_bbc_dalle": "BBC · DALL-E 3", "test3_cnn_dalle": "CNN · DALL-E 3",
          "test4_bbc_sdxl": "BBC · SDXL", "test5_cnn_sdxl": "CNN · SDXL"}
INPUTS = {"original": "① 원본", "card": "② 카드", "crop": "③ 잘라낸 사진"}


def auc(labels, scores) -> float:
    """AI 사진에 진짜 사진보다 높은 점수를 줄 확률 (점수가 같으면 반으로 친다). 0.5 = 찍기, 1 = 완벽."""
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks, start = np.empty(len(scores)), 0
    for end in range(1, len(scores) + 1):
        if end == len(scores) or scores[order[end]] != scores[order[start]]:
            ranks[order[start:end]] = (start + 1 + end) / 2  # 같은 점수끼리는 순위 평균
            start = end
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_truth(data: Path) -> tuple[dict, dict]:
    with open(data / "ai_photos" / "ai_photos.csv", newline="", encoding="utf-8") as f:
        photos = {r["image_id"]: {"split": r["split"], "is_ai": int(r["label"])} for r in csv.DictReader(f)}
    with open(data / "cards" / "cards.csv", newline="", encoding="utf-8") as f:
        cards = {r["card_id"]: {"split": r["split"], "is_ai": int(r["is_ai"]), "image_id": r["image_id"],
                                "text_fake": int(r["text_fake"]), "cross": bool(r["cross_item_id"])}
                 for r in csv.DictReader(f)}
    return photos, cards


def read_scores(path: Path, photos: dict, cards: dict) -> dict:
    """점수 파일을 읽어 입력 종류별 [(id, 점수, 정답 정보)]로 돌려준다. 잘못된 줄이 있으면 멈춘다."""
    items, seen = defaultdict(list), set()
    with open(path, newline="", encoding="utf-8") as f:
        for n, r in enumerate(csv.DictReader(f), start=2):
            kind, item_id = r.get("input", ""), r.get("id", "")
            if kind not in INPUTS:
                raise SystemExit(f"{path} {n}번째 줄: input은 original, card, crop 중 하나여야 해요 ({kind!r}).")
            table = photos if kind == "original" else cards
            if item_id not in table:
                raise SystemExit(f"{path} {n}번째 줄: 모르는 id예요 ({item_id!r}).")
            if (kind, item_id) in seen:
                raise SystemExit(f"{path} {n}번째 줄: 같은 문항이 두 번 있어요 ({kind}, {item_id}).")
            try:
                score = float(r["score"])
            except (TypeError, ValueError):
                score = float("nan")
            if not math.isfinite(score):
                raise SystemExit(f"{path} {n}번째 줄: 점수가 숫자가 아니에요 ({r.get('score')!r}).")
            seen.add((kind, item_id))
            items[kind].append((item_id, score, table[item_id]))
    return items


def summarize(items: list, threshold: float) -> dict:
    y = np.array([t["is_ai"] for _, _, t in items])
    s = np.array([score for _, score, _ in items])
    pred = s >= threshold
    return {"n": len(items), "acc": float((pred == y).mean()) if len(items) else float("nan"), "auc": auc(y, s),
            "real_acc": float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan"),
            "ai_acc": float(pred[y == 1].mean()) if (y == 1).any() else float("nan")}


def text_flips(items: list, threshold: float) -> dict:
    """같은 사진에 글만 다른 교차 시험 카드 쌍(사실 글 / 거짓 글)에서 판정이 바뀐 비율과 점수 차이."""
    by_photo = defaultdict(dict)
    for _, score, t in items:
        if t.get("cross"):
            by_photo[t["image_id"]][t["text_fake"]] = score
    pairs = [(v[0], v[1]) for v in by_photo.values() if len(v) == 2]
    if not pairs:
        return {"pairs": 0, "flip": float("nan"), "diff": float("nan")}
    a, b = np.array(pairs).T
    return {"pairs": len(pairs), "flip": float(((a >= threshold) != (b >= threshold)).mean()),
            "diff": float(np.abs(a - b).mean())}


def evaluate(items: dict, threshold: float) -> dict:
    result = {"splits": {}, "all": {}, "flips": {}}
    for kind in INPUTS:
        rows = items.get(kind, [])
        result["all"][kind] = summarize(rows, threshold)
        for split in TEST_SPLITS:
            result["splits"].setdefault(split, {})[kind] = summarize([r for r in rows if r[2]["split"] == split],
                                                                     threshold)
        if kind != "original":
            result["flips"][kind] = text_flips(rows, threshold)
    return result


def pct(x: float) -> str:
    return "-" if math.isnan(x) else f"{x * 100:.1f}%"


def num(x: float) -> str:
    return "-" if math.isnan(x) else f"{x:.3f}"


def report(result: dict, threshold: float, title: str = "카드 편집 실험 결과") -> str:
    def cell(m):
        return "-" if not m["n"] else f"{pct(m['acc'])} ({num(m['auc'])})"

    lines = [f"## {title}", "", f"판정 기준: 점수 {threshold} 이상이면 AI", "",
             "### split별 정확도 (괄호: AUC)", "",
             "| split | 출처 · 생성기 | " + " | ".join(INPUTS.values()) + " |", "|---|---|---|---|---|"]
    for split in TEST_SPLITS:
        lines.append(f"| {split} | {SOURCE[split]} | " +
                     " | ".join(cell(result["splits"][split][k]) for k in INPUTS) + " |")
    lines.append("| **전체** | | " + " | ".join(f"**{cell(result['all'][k])}**" for k in INPUTS) + " |")
    lines += ["", "### 진짜·AI별로 맞힌 비율 (전체)", "", "| 입력 | 진짜를 진짜로 | AI를 AI로 | 문항 수 |",
              "|---|---|---|---|"]
    for kind, name in INPUTS.items():
        m = result["all"][kind]
        lines.append(f"| {name} | {pct(m['real_acc'])} | {pct(m['ai_acc'])} | {m['n']:,} |")
    lines += ["", "### 글만 다른 카드끼리 판정이 바뀐 비율", "",
              "같은 사진에 사실 글 카드와 거짓 글 카드가 하나씩 있어요. 판정이 글에 흔들리지 않으면 0%예요.", "",
              "| 입력 | 판정이 바뀐 비율 | 점수 차이 평균 | 쌍 |", "|---|---|---|---|"]
    for kind, f in result["flips"].items():
        lines.append(f"| {INPUTS[kind]} | {pct(f['flip'])} | {num(f['diff'])} | {f['pairs']:,} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="카드 편집 실험 채점 (① 원본 ② 카드 ③ 잘라낸 사진)")
    parser.add_argument("scores", help="점수 CSV (input,id,score)")
    parser.add_argument("--data", default=str(REPO_ROOT / "data"), help="prepare_all.py·make_cards.py로 받은 data 폴더")
    parser.add_argument("--threshold", type=float, default=0.5, help="이 점수 이상이면 AI로 판정")
    parser.add_argument("--title", default="카드 편집 실험 결과")
    parser.add_argument("--out", help="결과 표를 저장할 파일 (.md)")
    args = parser.parse_args(argv)

    photos, cards = load_truth(Path(args.data))
    items = read_scores(Path(args.scores), photos, cards)
    text = report(evaluate(items, args.threshold), args.threshold, args.title)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
