"""합성 데이터로 make_cross_test.py를 시험한다. 실행: python -m pytest -q data_preparation"""
import csv
import json
import random

import pytest

import make_cross_test as mc

TOPICS = [
    ("flood", "river", "boats"), ("election", "ballot", "voters"), ("blaze", "smoke", "firefighters"),
    ("court", "judge", "trial"), ("hurricane", "coast", "evacuation"), ("stocks", "traders", "exchange"),
    ("rally", "square", "crowd"), ("school", "students", "teachers"), ("hospital", "doctors", "patients"),
    ("airport", "flights", "passengers"), ("factory", "workers", "strike"), ("stadium", "fans", "match"),
]


def split_rows(rng, prefix):
    """앞 절반은 지어낸 글(AI), 뒤 절반은 기사 글(진짜). 같은 순번끼리 같은 사건(같은 주제·고유 단어)."""
    topics = TOPICS[:]
    rng.shuffle(topics)
    fakes, reals = [], []
    for i, (a, b, c) in enumerate(topics):
        key = f"{prefix}event{i}"
        reals.append((f"The {a} {b} near {key} drew attention on Tuesday.", 0))
        fakes.append((f"Shocking {a} {c} scandal at {key} today.", 1))
    return fakes + reals


@pytest.fixture(scope="module")
def rows_by_split():
    rng = random.Random(0)
    return {split: split_rows(rng, f"s{k}x") for k, split in enumerate(mc.TEST_SPLITS)}


@pytest.fixture(scope="module")
def built(rows_by_split):
    return mc.build_cross(rows_by_split, seed=42)


def test_four_cells_and_each_piece_twice(built, rows_by_split):
    table, stats = built
    mc.verify(table, rows_by_split)
    assert len(table) == 4 * 12 * len(mc.TEST_SPLITS)
    for split in mc.TEST_SPLITS:
        assert stats[split]["events"] == 12 and stats[split]["same_row_top1"] == 1.0


def test_photo_answer_does_not_predict_text_answer(built):
    table, _ = built
    agree = sum(r["is_ai"] == r["text_fake"] for r in table)
    assert agree * 2 == len(table)  # 사진 정답으로 글 정답을 맞히면 정확히 50%


def test_text_is_from_same_event(built, rows_by_split):
    table, _ = built
    for r in table:
        rows = rows_by_split[r["split"]]
        key_image = rows[r["image_row"]][0].split("event")[1].split()[0]
        key_text = rows[r["text_row"]][0].split("event")[1].split()[0]
        assert key_image == key_text


def test_same_seed_same_table(rows_by_split, built):
    table, _ = built
    assert mc.build_cross(rows_by_split, seed=42)[0] == table
    assert mc.build_cross(rows_by_split, seed=7)[0] != table


def test_verify_catches_other_event_text(built, rows_by_split):
    table, _ = built
    broken = [dict(r) for r in table]
    victim = next(r for r in broken if r["is_ai"] == 0 and r["text_fake"] == 1)
    other = next(r for r in broken if r["split"] == victim["split"] and r["is_ai"] == 1
                 and r["image_row"] != victim["text_row"])
    victim["text_row"] = other["image_row"]  # 다른 사건의 지어낸 글
    with pytest.raises(AssertionError):
        mc.verify(broken, rows_by_split)


def test_misaligned_rows_are_refused(rows_by_split):
    rows = list(rows_by_split["test1_nyt_mj"])
    n = len(rows) // 2
    shifted = rows[:n] + rows[n + 1:] + rows[n:n + 1]  # 기사 글을 한 칸씩 밀어 짝을 어긋나게 함
    with pytest.raises(ValueError):
        mc.check_alignment(shifted)


def test_wrong_structure_is_refused():
    with pytest.raises(ValueError):
        mc.events([("a", 0), ("b", 1), ("c", 0), ("d", 1)])


def test_cli_writes_table_and_meta(tmp_path, rows_by_split):
    out = tmp_path / "cross.csv"
    code = mc.main(["--out", str(out)], loader=lambda revision: (rows_by_split, {}, "rev1"))
    assert code == 0
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == mc.FIELDS and len(rows) == 4 * 12 * len(mc.TEST_SPLITS)
    meta = json.loads((tmp_path / "cross.meta.json").read_text(encoding="utf-8"))
    assert meta["revision"] == "rev1" and meta["total_rows"] == len(rows)
