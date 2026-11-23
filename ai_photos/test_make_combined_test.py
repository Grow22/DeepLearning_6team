"""합성 데이터로 make_combined_test.py를 시험한다. 실행: python -m pytest -q ai_photos"""
import csv
import json
import random
from collections import Counter

import pytest

import make_combined_test as mc

WORDS = {
    "flood": "flood river rain water rescue villagers boats",
    "vote": "election vote ballot candidate polling campaign",
    "fire": "fire firefighters blaze smoke building burned",
    "court": "court judge trial verdict lawyer defendant",
    "storm": "storm hurricane wind damage coast evacuation",
}


def caption(rng, uid):
    topic = rng.choice(list(WORDS))
    return f"{' '.join(rng.sample(WORDS[topic].split(), 4))} in city {uid}."


@pytest.fixture(scope="module")
def rows_by_split():
    rng = random.Random(0)
    rows = {}
    for split in mc.TEST_SPLITS:
        real = [(caption(rng, f"{split}r{i}"), 0) for i in range(20)]
        fake = [(f"Imagined {caption(rng, f'{split}f{i}')}", 1) for i in range(20)]
        rows[split] = fake + real  # 가짜를 앞에 몰아 둔다 (실제 데이터 순서 흉내)
    # 같은 사건(거의 같은 글) 짝: 안 맞음 쌍으로 묶이면 안 된다
    text = rows["test1_nyt_mj"][25][0]
    rows["test1_nyt_mj"][26] = (text.replace("in city", "near city"), 0)
    return rows


@pytest.fixture(scope="module")
def built(rows_by_split):
    return mc.build_combined(rows_by_split, seed=42, dup_thr=0.3)


def test_four_cases_are_balanced(built, rows_by_split):
    table, stats = built
    mc.verify(table, rows_by_split, 0.3)
    for split in mc.TEST_SPLITS:
        assert stats[split] == {"real_match": 10, "real_mismatch": 10, "ai_match": 10, "ai_mismatch": 10}
    assert len(table) == 40 * len(mc.TEST_SPLITS)


def test_each_photo_and_caption_used_once(built):
    table, _ = built
    for split in mc.TEST_SPLITS:
        items = [r for r in table if r["split"] == split]
        assert len({r["image_row"] for r in items}) == len(items)
        assert len({r["text_row"] for r in items}) == len(items)


def test_same_event_never_paired(built):
    table, _ = built
    pairs = {(r["image_row"], r["text_row"]) for r in table if r["split"] == "test1_nyt_mj" and r["is_mismatch"]}
    assert (25, 26) not in pairs and (26, 25) not in pairs


def test_same_seed_same_table(rows_by_split, built):
    table, _ = built
    assert mc.build_combined(rows_by_split, seed=42, dup_thr=0.3)[0] == table
    assert mc.build_combined(rows_by_split, seed=7, dup_thr=0.3)[0] != table


def test_verify_catches_mixed_kinds(built, rows_by_split):
    table, _ = built
    broken = [dict(r) for r in table]
    real_mis = next(r for r in broken if r["is_mismatch"] and r["is_ai"] == 0)
    ai_row = next(r["image_row"] for r in broken if r["split"] == real_mis["split"] and r["is_ai"] == 1)
    real_mis["text_row"] = ai_row  # 진짜 사진에 AI 글을 붙이면 잡혀야 한다
    with pytest.raises(AssertionError):
        mc.verify(broken, rows_by_split, 0.3)


def test_cli_writes_table_and_meta(tmp_path, rows_by_split):
    out = tmp_path / "combined.csv"
    code = mc.main(["--out", str(out)], loader=lambda revision: (rows_by_split, {}, "rev1"))
    assert code == 0
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == mc.FIELDS and len(rows) == 200
    meta = json.loads((tmp_path / "combined.meta.json").read_text(encoding="utf-8"))
    assert meta["revision"] == "rev1" and meta["total_rows"] == 200
