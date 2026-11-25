#!/usr/bin/env python3
"""교차 시험 세트 표를 만든다: 같은 사건의 진짜/AI 사진 × 사실/거짓 글.

MiRAGeNews 테스트 split은 앞 절반이 AI 쌍(AI 사진 + GPT-4가 지어낸 글), 뒤 절반이 진짜 쌍(진짜 사진 + 기사 글)이고,
같은 순번끼리 같은 사건이다.
- 지어낸 글 i번: 기사 글 i번을 GPT-4가 거짓으로 바꿔 쓴 글
- AI 사진 i번: 지어낸 글 i번으로 만든 사진

사건마다 네 문항을 만든다. 글은 모두 같은 사건의 것이라 사진과 맞는 글이다.
    진짜 사진 + 기사 글     is_ai 0, text_fake 0
    진짜 사진 + 지어낸 글   is_ai 0, text_fake 1   (기존 통합 시험 세트에 없던 조합)
    AI 사진 + 기사 글       is_ai 1, text_fake 0   (기존 통합 시험 세트에 없던 조합)
    AI 사진 + 지어낸 글     is_ai 1, text_fake 1

결과 CSV에는 행 번호와 정답만 들어간다. 사진·글은 prepare_ai_photos.py가 받아서 조립한다.

사용법 (표를 새로 만들 때만):
    pip install datasets scikit-learn numpy
    python data_preparation/ai_photos/make_cross_test.py
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
import make_mismatch_pairs as mm  # noqa: E402  (글 유사도 계산을 같이 쓴다)

TEST_SPLITS = ["test1_nyt_mj", "test2_bbc_dalle", "test3_cnn_dalle", "test4_bbc_sdxl", "test5_cnn_sdxl"]
FIELDS = ["item_id", "split", "image_row", "text_row", "is_ai", "text_fake"]
CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (is_ai, text_fake)


def events(rows: list[tuple[str, int]]) -> list[tuple[int, int]]:
    """(AI 행, 진짜 행) 짝 목록. 앞 절반이 AI(label 1), 뒤 절반이 진짜(label 0)여야 한다."""
    n = len(rows) // 2
    labels = [label for _, label in rows]
    if n == 0 or len(rows) != 2 * n or labels[:n] != [1] * n or labels[n:] != [0] * n:
        raise ValueError("앞 절반이 AI, 뒤 절반이 진짜인 구조가 아니에요.")
    return [(i, n + i) for i in range(n)]


def check_alignment(rows: list[tuple[str, int]], min_top1: float = 0.8, min_ratio: float = 10.0) -> dict:
    """같은 순번의 지어낸 글과 기사 글이 같은 사건인지 글 유사도로 확인한다.

    같은 순번끼리의 유사도가 한 칸 어긋난 순번끼리보다 훨씬 높고, 대부분 같은 순번이 가장 비슷해야 한다.
    """
    n = len(rows) // 2
    sim = mm.similarity([t for t, _ in rows])[:n, n:]
    k = np.arange(n)
    same = float(sim[k, k].mean())
    shifted = float(sim[k, (k + 1) % n].mean())
    top1 = float((sim.argmax(1) == k).mean())
    if top1 < min_top1 or same < min_ratio * max(shifted, 1e-6):
        raise ValueError(f"같은 순번끼리 같은 사건이 아닌 것 같아요 "
                         f"(같은 순번이 가장 비슷한 비율 {top1:.1%}, 유사도 {same:.3f} vs 어긋난 순번 {shifted:.3f})")
    return {"same_row_sim": round(same, 4), "shifted_row_sim": round(shifted, 4), "same_row_top1": round(top1, 4)}


def build_cross(rows_by_split: dict, seed: int = 42, splits: list[str] = TEST_SPLITS) -> tuple[list[dict], dict]:
    """rows_by_split: split -> [(글, label), ...]. label 0 = 진짜, 1 = AI."""
    rng = np.random.default_rng(seed)
    table, stats = [], {}
    for split in splits:
        rows = rows_by_split[split]
        pairs = events(rows)
        stats[split] = {"events": len(pairs), **check_alignment(rows)}
        block = []
        for ai_row, real_row in pairs:
            block += [
                (real_row, real_row, 0, 0),  # 진짜 사진 + 기사 글
                (real_row, ai_row, 0, 1),    # 진짜 사진 + 지어낸 글
                (ai_row, real_row, 1, 0),    # AI 사진 + 기사 글
                (ai_row, ai_row, 1, 1),      # AI 사진 + 지어낸 글
            ]
        # 같은 사진의 두 문항이 붙어 나오지 않게 섞는다
        for k, j in enumerate(rng.permutation(len(block))):
            image_row, text_row, is_ai, text_fake = block[j]
            table.append({"item_id": f"cross-{split}-{k:04d}", "split": split, "image_row": image_row,
                          "text_row": text_row, "is_ai": is_ai, "text_fake": text_fake})
    return table, stats


def verify(table: list[dict], rows_by_split: dict) -> None:
    """표가 약속대로 만들어졌는지 확인한다. 어긋나면 AssertionError."""
    for split in sorted({r["split"] for r in table}):
        rows = rows_by_split[split]
        pairs = events(rows)
        partner = {}
        for ai_row, real_row in pairs:
            partner[ai_row], partner[real_row] = real_row, ai_row
        items = [r for r in table if r["split"] == split]
        n = len(pairs)
        assert len({r["item_id"] for r in items}) == len(items), f"{split}: 문항 번호가 겹쳐요"
        cells = Counter((r["is_ai"], r["text_fake"]) for r in items)
        assert all(cells[c] == n for c in CELLS) and len(items) == 4 * n, f"{split}: 네 칸이 같은 수가 아니에요 {cells}"
        for r in items:
            assert rows[r["image_row"]][1] == r["is_ai"], f"{split}: AI 정답이 원본과 달라요 {r}"
            assert rows[r["text_row"]][1] == r["text_fake"], f"{split}: 글 정답이 원본과 달라요 {r}"
            assert r["text_row"] in (r["image_row"], partner[r["image_row"]]), f"{split}: 다른 사건의 글이 붙었어요 {r}"
        # 사진마다 사실 글 1번 + 거짓 글 1번, 글마다 진짜 사진 1번 + AI 사진 1번
        by_image, by_text = {}, {}
        for r in items:
            by_image.setdefault(r["image_row"], []).append(r["text_fake"])
            by_text.setdefault(r["text_row"], []).append(r["is_ai"])
        assert len(by_image) == 2 * n and all(sorted(v) == [0, 1] for v in by_image.values()), \
            f"{split}: 사진마다 사실 글·거짓 글이 한 번씩이 아니에요"
        assert len(by_text) == 2 * n and all(sorted(v) == [0, 1] for v in by_text.values()), \
            f"{split}: 글마다 진짜 사진·AI 사진이 한 번씩이 아니에요"
        # 사진 정답으로 글 정답을 맞히면 정확히 50%여야 한다
        agree = sum(r["is_ai"] == r["text_fake"] for r in items)
        assert 2 * agree == len(items), f"{split}: 사진 정답과 글 정답이 엮여 있어요 ({agree}/{len(items)})"


def main(argv: list[str] | None = None, loader=mm.load_from_hub) -> int:
    parser = argparse.ArgumentParser(description="교차 시험 세트 표 만들기")
    parser.add_argument("--out", default=str(HERE / "cross_test_pairs.csv"))
    parser.add_argument("--revision", default="main", help="MiRAGeNews 버전 (통합 시험 세트와 같은 버전을 쓴다)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rows_by_split, _, revision = loader(args.revision)
    table, stats = build_cross(rows_by_split, seed=args.seed)
    verify(table, rows_by_split)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)
    meta = {
        "dataset": mm.DATASET, "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "label": {"is_ai": "0 = 진짜 사진, 1 = AI 사진",
                  "text_fake": "0 = 기사 글(사실), 1 = GPT-4가 지어낸 글(거짓)"},
        "note": "글은 모두 같은 사건의 것이라 사진과 맞는 글이다.",
        "stats": stats, "total_rows": len(table),
    }
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MiRAGeNews revision: {revision}")
    for split, s in stats.items():
        print(f"{split:16s} 사건 {s['events']}개 → {4 * s['events']}문항 "
              f"(같은 순번이 가장 비슷한 비율 {s['same_row_top1']:.1%})")
    print(f"검증 통과. 저장: {args.out} ({len(table)}문항), {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
