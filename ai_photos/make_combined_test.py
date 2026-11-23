#!/usr/bin/env python3
"""통합 시험 세트 표(진짜/AI 사진 x 맞는/안 맞는 글)를 만든다.

시험용 split마다 진짜 사진 250장, AI 사진 250장을 한 번씩만 쓴다.
- 진짜 사진: 절반은 원래 글(맞음), 절반은 다른 진짜 사진의 글(안 맞음)
- AI 사진: 절반은 원래 글(맞음, 사진을 만들 때 쓴 글), 절반은 다른 AI 사진의 글(안 맞음)
안 맞는 글은 같은 split, 같은 종류(진짜끼리, AI끼리) 안에서만 바꾸고 글마다 한 번씩만 쓴다.
같은 사건으로 보이는 글(유사도 0.3 이상)은 안 맞음 쌍에 쓰지 않는다.

결과 CSV에는 행 번호와 정답만 들어간다. 사진·글은 prepare_ai_photos.py가 받아서 조립한다.

사용법 (표를 새로 만들 때만):
    pip install datasets scikit-learn numpy
    python ai_photos/make_combined_test.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mismatch_pairs"))
import make_mismatch_pairs as mm  # noqa: E402  (유사도 계산·글 배정 함수를 같이 쓴다)

TEST_SPLITS = ["test1_nyt_mj", "test2_bbc_dalle", "test3_cnn_dalle", "test4_bbc_sdxl", "test5_cnn_sdxl"]
FIELDS = ["item_id", "split", "image_row", "text_row", "is_ai", "is_mismatch"]


def build_combined(rows_by_split: dict, seed: int = 42, dup_thr: float = 0.3,
                   splits: list[str] = TEST_SPLITS) -> tuple[list[dict], dict]:
    """rows_by_split: split -> [(글, label), ...]. label 0 = 진짜, 1 = AI."""
    rng = np.random.default_rng(seed)
    table, stats = [], {}
    for split in splits:
        rows = rows_by_split[split]
        block = []
        for is_ai in (0, 1):
            photos = [row for row, (_, label) in enumerate(rows) if label == is_ai]
            order = rng.permutation(len(photos))
            n_mismatch = len(photos) // 2
            mismatch = [photos[k] for k in order[:n_mismatch]]
            match = [photos[k] for k in order[n_mismatch:]]
            block += [(row, row, is_ai, 0) for row in match]
            if n_mismatch:
                if n_mismatch < 2:
                    raise ValueError(f"{split}: 안 맞음으로 만들 사진이 너무 적어요.")
                sim = mm.similarity([rows[row][0] for row in mismatch])
                perm = mm.easy_assignment(sim, dup_thr, rng)
                block += [(row, mismatch[perm[i]], is_ai, 1) for i, row in enumerate(mismatch)]
        # 앞에서부터 잘라 써도 한 경우만 나오지 않게 섞는다
        for k, j in enumerate(rng.permutation(len(block))):
            image_row, text_row, is_ai, is_mismatch = block[j]
            table.append({"item_id": f"{split}-{k:04d}", "split": split, "image_row": image_row,
                          "text_row": text_row, "is_ai": is_ai, "is_mismatch": is_mismatch})
        counts = Counter((r[2], r[3]) for r in block)
        stats[split] = {"real_match": counts[(0, 0)], "real_mismatch": counts[(0, 1)],
                        "ai_match": counts[(1, 0)], "ai_mismatch": counts[(1, 1)]}
    return table, stats


def verify(table: list[dict], rows_by_split: dict, dup_thr: float) -> None:
    """표가 약속대로 만들어졌는지 확인한다. 어긋나면 AssertionError."""
    for split in {r["split"] for r in table}:
        rows = rows_by_split[split]
        items = [r for r in table if r["split"] == split]
        images = [r["image_row"] for r in items]
        texts = [r["text_row"] for r in items]
        assert len(set(images)) == len(images) == len(rows), f"{split}: 사진이 빠지거나 겹쳐요"
        assert len(set(texts)) == len(texts), f"{split}: 같은 글이 두 번 나와요"
        for r in items:
            assert rows[r["image_row"]][1] == r["is_ai"], f"{split}: AI 정답이 원본과 달라요 {r}"
            if r["is_mismatch"]:
                assert r["text_row"] != r["image_row"], f"{split}: 안 맞음에 자기 글이 붙었어요 {r}"
                assert rows[r["text_row"]][1] == r["is_ai"], f"{split}: 진짜·AI 글이 섞였어요 {r}"
            else:
                assert r["text_row"] == r["image_row"], f"{split}: 맞음에 다른 글이 붙었어요 {r}"
        counts = Counter((r["is_ai"], r["is_mismatch"]) for r in items)
        for is_ai in (0, 1):
            n = sum(1 for _, label in rows if label == is_ai)
            assert counts[(is_ai, 1)] == n // 2 and counts[(is_ai, 0)] == n - n // 2, \
                f"{split}: 맞음·안 맞음이 반반이 아니에요 {counts}"
        for is_ai in (0, 1):
            mismatch = [r for r in items if r["is_mismatch"] and r["is_ai"] == is_ai]
            if not mismatch:
                continue
            # 만들 때와 같은 글 묶음(안 맞음으로 뽑힌 사진들의 글)으로 유사도를 다시 계산한다
            index = {r["image_row"]: i for i, r in enumerate(mismatch)}
            assert all(r["text_row"] in index for r in mismatch), f"{split}: 맞음 사진의 글이 다른 사진에 붙었어요"
            sim = mm.similarity([rows[r["image_row"]][0] for r in mismatch])
            assert all(sim[index[r["image_row"]], index[r["text_row"]]] < dup_thr for r in mismatch), \
                f"{split}: 같은 사건으로 보이는 글이 안 맞음 쌍에 들어갔어요"


def main(argv: list[str] | None = None, loader=mm.load_from_hub) -> int:
    parser = argparse.ArgumentParser(description="통합 시험 세트 표 만들기")
    parser.add_argument("--out", default=str(HERE / "combined_test_pairs.csv"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dup-thr", type=float, default=0.3)
    args = parser.parse_args(argv)

    rows_by_split, label_counts, revision = loader(args.revision)
    table, stats = build_combined(rows_by_split, seed=args.seed, dup_thr=args.dup_thr)
    verify(table, rows_by_split, args.dup_thr)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)
    meta = {
        "dataset": mm.DATASET, "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed, "dup_thr": args.dup_thr,
        "label": {"is_ai": "0 = 진짜 사진, 1 = AI 사진", "is_mismatch": "0 = 글이 맞음, 1 = 글이 안 맞음"},
        "stats": stats, "total_rows": len(table),
    }
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MiRAGeNews revision: {revision}")
    for split, s in stats.items():
        print(f"{split:16s} 진짜+맞음 {s['real_match']}  진짜+안 맞음 {s['real_mismatch']}  "
              f"AI+맞음 {s['ai_match']}  AI+안 맞음 {s['ai_mismatch']}")
    print(f"검증 통과. 저장: {args.out} ({len(table)}문항), {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
