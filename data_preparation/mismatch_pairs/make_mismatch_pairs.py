#!/usr/bin/env python3
"""MiRAGeNews 진짜 쌍(label 0)으로 '글-사진 짝' 데이터를 만든다.

- 맞음(label 0): 사진 + 원래 글
- 안 맞음(label 1): 사진 + 같은 묶음에 있는 다른 사진의 글
  - easy: 무작위로 고른 글
  - hard: 내용이 비슷한 글(TF-IDF 유사도). 같은 사건으로 보이는 글은 뺀다.

결과 CSV에는 행 번호만 들어간다(사진·글 원본은 없음).
같은 revision의 MiRAGeNews를 불러와 이 표대로 조립하면 팀 전체가 같은 데이터를 쓴다.

사용법:
    pip install datasets scikit-learn numpy
    python make_mismatch_pairs.py --out mismatch_pairs.csv --preview 5
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

DATASET = "anson-huang/mirage-news"

# 글은 같은 묶음 안에서만 바꾼다. 시험 세트는 출처(NYT/BBC/CNN)별로 묶는다.
# (BBC·CNN 진짜 사진은 두 시험 세트에 같이 들어 있을 수 있어서, 같은 글은 한 번만 쓴다.)
GROUPS: dict[str, list[str]] = {
    "train": ["train"],
    "validation": ["validation"],
    "test_nyt": ["test1_nyt_mj"],
    "test_bbc": ["test2_bbc_dalle", "test4_bbc_sdxl"],
    "test_cnn": ["test3_cnn_dalle", "test5_cnn_sdxl"],
}

# 거의 모든 기사 설명에 들어가는 요일은 유사도 계산에서 뺀다.
WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}

FIELDS = ["pair_id", "set", "group", "image_split", "image_row",
          "text_split", "text_row", "label", "text_sim"]


# ---------------------------------------------------------------------------
# 1. 진짜 쌍 모으기
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def collect_real(rows_by_split: dict[str, list[tuple[str, int]]],
                 groups: dict[str, list[str]] = GROUPS) -> dict:
    """묶음별로 진짜 쌍(label 0)만 모은다. 같은 묶음 안에서 글이 같은 행은 한 번만 쓴다.

    rows_by_split: split 이름 -> [(글, label), ...] (데이터셋 행 순서 그대로)
    """
    real = {}
    for group, splits in groups.items():
        items, seen, duplicates = [], set(), 0
        for split in splits:
            for row, (text, label) in enumerate(rows_by_split[split]):
                if label != 0:
                    continue
                key = normalize(text)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                items.append((split, row, text))
        real[group] = {"items": items, "duplicates": duplicates}
    return real


# ---------------------------------------------------------------------------
# 2. 글 유사도와 글 배정
# ---------------------------------------------------------------------------
def similarity(texts: list[str]) -> np.ndarray:
    """글끼리의 TF-IDF 코사인 유사도 (0~1)."""
    vectorizer = TfidfVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS | WEEKDAYS),
                                 ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    return (matrix @ matrix.T).toarray().astype(np.float32)


def _violations(perm: np.ndarray, sim: np.ndarray, dup_thr: float) -> np.ndarray:
    idx = np.arange(len(perm))
    return np.flatnonzero((perm == idx) | (sim[idx, perm] >= dup_thr))


def repair(perm: np.ndarray, sim: np.ndarray, dup_thr: float,
           rng: np.random.Generator, max_rounds: int = 200) -> np.ndarray:
    """자기 글이 붙었거나 같은 사건으로 보이는 글이 붙은 자리를 다른 자리와 맞바꿔 없앤다."""
    n = len(perm)
    for _ in range(max_rounds):
        bad = _violations(perm, sim, dup_thr)
        if bad.size == 0:
            return perm
        for i in bad:
            if perm[i] != i and sim[i, perm[i]] < dup_thr:
                continue  # 앞선 맞바꿈으로 이미 해결됨
            for k in rng.permutation(n):
                if k == i:
                    continue
                a, b = perm[k], perm[i]  # i는 a를, k는 b를 받는다
                if a != i and sim[i, a] < dup_thr and b != k and sim[k, b] < dup_thr:
                    perm[i], perm[k] = a, b
                    break
    raise RuntimeError("조건에 맞는 짝을 찾지 못했어요. 묶음이 너무 작거나 --dup-thr가 너무 낮아요.")


def easy_assignment(sim: np.ndarray, dup_thr: float, rng: np.random.Generator) -> np.ndarray:
    """무작위 배정. 글마다 정확히 한 번씩 쓰인다."""
    return repair(rng.permutation(len(sim)), sim, dup_thr, rng)


def hard_assignment(sim: np.ndarray, dup_thr: float, rng: np.random.Generator,
                    topk: int) -> tuple[np.ndarray, int]:
    """비슷한 글 우선 배정. 글마다 정확히 한 번씩 쓰인다.

    사진마다 비슷한 글 후보 topk개를 뽑고, 유사도가 높은 짝부터 차례로 확정한다.
    후보를 못 받은 사진은 남은 글 중에서 무작위로 받는다(반환값의 두 번째 수).
    """
    n = len(sim)
    allowed = sim.copy()
    np.fill_diagonal(allowed, -1.0)
    allowed[allowed >= dup_thr] = -1.0
    k = min(topk, n - 1)
    cand = np.argsort(-allowed, axis=1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = cand.ravel()
    sims = allowed[rows, cols]
    keep = sims > 0
    rows, cols, sims = rows[keep], cols[keep], sims[keep]
    order = np.lexsort((cols, rows, -sims))  # 유사도 높은 순, 같으면 번호 순

    perm = np.full(n, -1)
    used = np.zeros(n, dtype=bool)
    for t in order:
        i, j = rows[t], cols[t]
        if perm[i] == -1 and not used[j]:
            perm[i] = j
            used[j] = True
    left = np.flatnonzero(perm == -1)
    perm[left] = rng.permutation(np.flatnonzero(~used))
    return repair(perm, sim, dup_thr, rng), int(left.size)


# ---------------------------------------------------------------------------
# 3. 짝 표 만들기와 검증
# ---------------------------------------------------------------------------
def build_pairs(real: dict, seed: int = 42, dup_thr: float = 0.3,
                topk: int = 20) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    table, stats = [], {}
    for group, info in real.items():
        items = info["items"]
        n = len(items)
        if n < 2:
            raise ValueError(f"{group}: 진짜 쌍이 {n}개뿐이라 글을 바꿀 수 없어요.")
        sim = similarity([text for _, _, text in items])
        easy = easy_assignment(sim, dup_thr, rng)
        hard, fallback = hard_assignment(sim, dup_thr, rng, topk)
        idx = np.arange(n)
        stats[group] = {
            "real_photos": n,
            "duplicates_skipped": info["duplicates"],
            "easy_mean_sim": round(float(sim[idx, easy].mean()), 4),
            "hard_mean_sim": round(float(sim[idx, hard].mean()), 4),
            "hard_random_fallback": fallback,
        }
        for set_name, perm in (("easy", easy), ("hard", hard)):
            block = []
            for i, (split, row, _) in enumerate(items):
                t_split, t_row, _ = items[perm[i]]
                block.append((split, row, split, row, 0, 1.0))
                block.append((split, row, t_split, t_row, 1, round(float(sim[i, perm[i]]), 4)))
            # 앞에서부터 잘라 써도 한쪽 정답만 나오지 않게 섞는다
            for num, k in enumerate(rng.permutation(len(block))):
                image_split, image_row, text_split, text_row, label, text_sim = block[k]
                table.append({
                    "pair_id": f"{set_name}-{group}-{num:05d}", "set": set_name, "group": group,
                    "image_split": image_split, "image_row": image_row,
                    "text_split": text_split, "text_row": text_row,
                    "label": label, "text_sim": text_sim,
                })
    return table, stats


def verify(table: list[dict], real: dict, dup_thr: float) -> None:
    """짝 표가 약속대로 만들어졌는지 확인한다. 어긋나면 AssertionError."""
    by = defaultdict(list)
    for r in table:
        by[(r["set"], r["group"])].append(r)
    assert set(by) == {(s, g) for s in ("easy", "hard") for g in real}, "세트·묶음 구성이 달라요"
    for (set_name, group), rs in by.items():
        n = len(real[group]["items"])
        where = f"{set_name}/{group}"
        allowed = {(split, row) for split, row, _ in real[group]["items"]}
        labels = Counter(r["label"] for r in rs)
        assert labels[0] == n and labels[1] == n, f"{where}: 맞음·안 맞음이 1:1이 아니에요 {labels}"
        for r in rs:
            assert (r["image_split"], r["image_row"]) in allowed, f"{where}: 묶음 밖 사진 {r}"
            assert (r["text_split"], r["text_row"]) in allowed, f"{where}: 묶음 밖 글 {r}"
        match = [r for r in rs if r["label"] == 0]
        mismatch = [r for r in rs if r["label"] == 1]
        assert all((r["image_split"], r["image_row"]) == (r["text_split"], r["text_row"]) for r in match), \
            f"{where}: 맞음 쌍에 다른 글이 붙었어요"
        assert all((r["image_split"], r["image_row"]) != (r["text_split"], r["text_row"]) for r in mismatch), \
            f"{where}: 안 맞음 쌍에 자기 글이 붙었어요"
        assert len({(r["image_split"], r["image_row"]) for r in mismatch}) == n, f"{where}: 사진이 겹쳐요"
        assert len({(r["text_split"], r["text_row"]) for r in mismatch}) == n, f"{where}: 글이 겹쳐요"
        assert all(r["text_sim"] <= round(dup_thr, 4) for r in mismatch), \
            f"{where}: 같은 사건으로 보이는 글이 안 맞음 쌍에 들어갔어요"
    for group in real:
        easy_match = {(r["image_split"], r["image_row"]) for r in by[("easy", group)] if r["label"] == 0}
        hard_match = {(r["image_split"], r["image_row"]) for r in by[("hard", group)] if r["label"] == 0}
        assert easy_match == hard_match, f"{group}: easy와 hard의 맞음 쌍이 달라요"


def write_csv(table: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)


# ---------------------------------------------------------------------------
# 4. Hugging Face에서 불러오기 / 짝 표로 조립하기
# ---------------------------------------------------------------------------
def load_from_hub(revision: str = "main"):
    """MiRAGeNews의 글·label을 불러온다. 사진은 짝 표를 만들 때 필요 없다."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(DATASET, revision=revision).sha
    ds = load_dataset(DATASET, revision=sha)
    needed = sorted({split for splits in GROUPS.values() for split in splits})
    missing = [split for split in needed if split not in ds]
    if missing:
        raise SystemExit(f"데이터에 없는 split이 있어요: {missing} (있는 split: {list(ds)})")

    rows_by_split, label_counts = {}, {}
    for split in ds:
        names = getattr(ds[split].features["label"], "names", None)
        if names is not None and [x.lower() for x in names[:2]] != ["real", "fake"]:
            raise SystemExit(f"{split}: label 이름이 예상(0=real, 1=fake)과 달라요: {names}")
        texts = list(ds[split]["text"])
        labels = [int(x) for x in ds[split]["label"]]
        rows_by_split[split] = list(zip(texts, labels))
        label_counts[split] = {"real": labels.count(0), "fake": labels.count(1)}
    return rows_by_split, label_counts, sha


def iter_pairs(csv_path: str, ds, set_name: str = "easy", group: str | None = None):
    """짝 표대로 (사진, 글, 정답, 표의 행)을 하나씩 꺼낸다.

    ds는 load_dataset(DATASET, revision=<meta의 revision>) 결과.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["set"] != set_name or (group is not None and r["group"] != group):
                continue
            image = ds[r["image_split"]][int(r["image_row"])]["image"]
            text = ds[r["text_split"]][int(r["text_row"])]["text"]
            yield image, text, int(r["label"]), r


# ---------------------------------------------------------------------------
# 5. 실행
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None, loader=load_from_hub) -> int:
    parser = argparse.ArgumentParser(description="MiRAGeNews 글-사진 짝 데이터 만들기")
    parser.add_argument("--out", default="mismatch_pairs.csv", help="짝 표(CSV) 저장 경로")
    parser.add_argument("--revision", default="main", help="MiRAGeNews 버전(기본: 최신을 commit 번호로 고정)")
    parser.add_argument("--seed", type=int, default=42)
    # 실제 데이터에서 0.45~0.5 구간에 같은 사건 글이 섞여 있어서 0.3으로 정함
    # (예: 같은 날 법원에 도착한 같은 인물의 사진 두 장)
    parser.add_argument("--dup-thr", type=float, default=0.3,
                        help="글 유사도가 이 값 이상이면 같은 사건으로 보고 안 맞음 쌍에서 뺀다")
    parser.add_argument("--topk", type=int, default=20, help="hard 세트에서 사진마다 볼 비슷한 글 후보 수")
    parser.add_argument("--preview", type=int, default=0, help="train에서 예시 몇 개를 화면에 보여 줄지")
    args = parser.parse_args(argv)

    rows_by_split, label_counts, revision = loader(args.revision)
    print(f"MiRAGeNews revision: {revision}")
    print(f"{'split':<18}{'진짜':>7}{'가짜':>7}")
    for split, c in label_counts.items():
        print(f"{split:<18}{c['real']:>7}{c['fake']:>7}")

    real = collect_real(rows_by_split)
    table, stats = build_pairs(real, seed=args.seed, dup_thr=args.dup_thr, topk=args.topk)
    verify(table, real, args.dup_thr)
    write_csv(table, args.out)

    meta = {
        "dataset": DATASET,
        "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "dup_thr": args.dup_thr,
        "topk": args.topk,
        "label": {"0": "맞음 (원래 글)", "1": "안 맞음 (다른 사진의 글)"},
        "groups": GROUPS,
        "source_label_counts": label_counts,
        "stats": stats,
        "total_rows": len(table),
    }
    meta_path = re.sub(r"\.csv$", "", args.out) + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n{'묶음':<12}{'진짜 사진':>9}{'세트당 쌍':>9}{'easy 유사도':>12}{'hard 유사도':>12}")
    for group, s in stats.items():
        print(f"{group:<12}{s['real_photos']:>9}{2 * s['real_photos']:>9}"
              f"{s['easy_mean_sim']:>12.3f}{s['hard_mean_sim']:>12.3f}")
    total_real = sum(s["real_photos"] for s in stats.values())
    print(f"\n진짜 사진 {total_real}장 → 세트당 {2 * total_real}쌍 (easy·hard 합계 {len(table)}행)")
    print(f"검증 통과. 저장: {args.out}, {meta_path}")

    if args.preview:
        texts = {(split, row): text for split, rows in rows_by_split.items()
                 for row, (text, _) in enumerate(rows)}
        for set_name in ("easy", "hard"):
            print(f"\n[{set_name} 예시: 사진의 원래 글 → 붙인 글]")
            shown = 0
            for r in table:
                if r["set"] == set_name and r["group"] == "train" and r["label"] == 1:
                    print(f"- {texts[(r['image_split'], r['image_row'])][:90]}")
                    print(f"  → {texts[(r['text_split'], r['text_row'])][:90]} (유사도 {r['text_sim']})")
                    shown += 1
                    if shown >= args.preview:
                        break
    return 0


if __name__ == "__main__":
    sys.exit(main())
