"""가짜(합성) 데이터로 make_mismatch_pairs.py를 시험한다. 실행: python -m pytest -q"""
import csv
import json
import random
from collections import Counter

import pytest

import make_mismatch_pairs as mm

TOPICS = {
    "flood": "flood river rain water rescue villagers boats",
    "election": "election vote ballot candidate polling campaign",
    "protest": "protesters police clash rally demonstration square",
    "fire": "fire firefighters blaze smoke building burned",
    "court": "court judge trial verdict lawyer defendant",
    "sports": "match stadium fans team goal championship",
    "storm": "storm hurricane wind damage coast evacuation",
    "market": "market stocks traders exchange shares prices",
}
CITIES = ["Baghdad", "Paris", "Chicago", "Lagos", "Manila", "Lima", "Oslo", "Delhi",
          "Cairo", "Seoul", "Quito", "Hanoi", "Accra", "Dublin", "Tunis", "Riga"]


def caption(rng: random.Random, uid: str) -> str:
    topic = rng.choice(list(TOPICS))
    words = rng.sample(TOPICS[topic].split(), 4)
    return f"In {rng.choice(CITIES)} on Tuesday, {' '.join(words)} {uid}."


def make_split(rng, n_real, n_fake, prefix, shared_real=None):
    """가짜를 앞에, 진짜를 뒤에 몰아 둔다(실제 데이터의 행 순서를 흉내 냄)."""
    fakes = [(f"Fabricated scene {prefix} {i} generated", 1) for i in range(n_fake)]
    reals = shared_real if shared_real is not None else [
        (caption(rng, f"{prefix}r{i}"), 0) for i in range(n_real)]
    return fakes + reals


@pytest.fixture(scope="module")
def rows_by_split():
    rng = random.Random(0)
    bbc_real = [(caption(rng, f"bbc{i}"), 0) for i in range(10)]
    cnn_real = [(caption(rng, f"cnn{i}"), 0) for i in range(8)]
    rows = {
        "train": make_split(rng, 60, 60, "tr"),
        "validation": make_split(rng, 20, 20, "va"),
        "test1_nyt_mj": make_split(rng, 10, 10, "t1"),
        "test2_bbc_dalle": make_split(rng, 0, 10, "t2", shared_real=bbc_real),
        "test4_bbc_sdxl": make_split(rng, 0, 10, "t4", shared_real=bbc_real),
        "test3_cnn_dalle": make_split(rng, 0, 10, "t3", shared_real=cnn_real),
        "test5_cnn_sdxl": make_split(rng, 0, 10, "t5",
                                     shared_real=cnn_real + [(caption(rng, "cnnx0"), 0),
                                                             (caption(rng, "cnnx1"), 0)]),
    }
    # 같은 사건(거의 같은 글) 짝 3개를 train에 넣는다: 안 맞음 쌍으로 묶이면 안 된다
    base = [r for r in rows["train"] if r[1] == 0][:3]
    for i, (text, _) in enumerate(base):
        rows["train"].append((text.replace("Tuesday", "Wednesday").rstrip(".") + " today.", 0))
    return rows


@pytest.fixture(scope="module")
def built(rows_by_split):
    real = mm.collect_real(rows_by_split)
    table, stats = mm.build_pairs(real, seed=42, dup_thr=0.5, topk=20)
    return real, table, stats


def test_verify_passes_and_counts(built):
    real, table, stats = built
    mm.verify(table, real, 0.5)
    assert stats["train"]["real_photos"] == 63
    assert stats["validation"]["real_photos"] == 20
    assert stats["test_nyt"]["real_photos"] == 10
    assert len(table) == 2 * 2 * sum(s["real_photos"] for s in stats.values())


def test_duplicates_across_test_splits_used_once(built):
    _, _, stats = built
    assert stats["test_bbc"]["real_photos"] == 10 and stats["test_bbc"]["duplicates_skipped"] == 10
    assert stats["test_cnn"]["real_photos"] == 10 and stats["test_cnn"]["duplicates_skipped"] == 8


def test_only_real_rows_used(built, rows_by_split):
    _, table, _ = built
    for r in table:
        assert rows_by_split[r["image_split"]][r["image_row"]][1] == 0
        assert rows_by_split[r["text_split"]][r["text_row"]][1] == 0


def test_same_event_never_paired_as_mismatch(built, rows_by_split):
    _, table, _ = built
    texts = {(s, i): t for s, rows in rows_by_split.items() for i, (t, _) in enumerate(rows)}
    sim_pairs = 0
    for r in table:
        if r["label"] != 1:
            continue
        a = texts[(r["image_split"], r["image_row"])]
        b = texts[(r["text_split"], r["text_row"])]
        a_core = a.split(",")[1].replace("today.", "").strip(" .")
        b_core = b.split(",")[1].replace("today.", "").strip(" .")
        sim_pairs += a_core == b_core
    assert sim_pairs == 0


def test_same_seed_same_result_and_other_seed_differs(rows_by_split, built):
    real, table, _ = built
    again, _ = mm.build_pairs(mm.collect_real(rows_by_split), seed=42, dup_thr=0.5, topk=20)
    other, _ = mm.build_pairs(mm.collect_real(rows_by_split), seed=7, dup_thr=0.5, topk=20)
    assert again == table
    assert other != table


def test_hard_is_more_similar_than_easy(built):
    _, _, stats = built
    for group in ("train", "validation"):
        assert stats[group]["hard_mean_sim"] > stats[group]["easy_mean_sim"]


def test_rows_are_shuffled(built):
    _, table, _ = built
    labels = [r["label"] for r in table if r["set"] == "easy" and r["group"] == "train"]
    half = len(labels) // 2
    assert 0 < sum(labels[:half]) < half  # 앞쪽 절반이 한쪽 정답으로 몰려 있지 않다


def test_two_items_swap_and_one_item_fails():
    two = {"g": {"items": [("s", 0, "flood river rescue boats"), ("s", 1, "court judge trial verdict")],
                 "duplicates": 0}}
    table, _ = mm.build_pairs(two, seed=1)
    mm.verify(table, two, 0.5)
    pairs = {(r["image_row"], r["text_row"]) for r in table if r["label"] == 1}
    assert pairs == {(0, 1), (1, 0)}
    one = {"g": {"items": [("s", 0, "flood river rescue boats")], "duplicates": 0}}
    with pytest.raises(ValueError):
        mm.build_pairs(one)


def test_verify_catches_broken_table(built):
    real, table, _ = built
    broken = [dict(r) for r in table]
    victim = next(r for r in broken if r["label"] == 1)
    victim["text_split"], victim["text_row"] = victim["image_split"], victim["image_row"]
    with pytest.raises(AssertionError):
        mm.verify(broken, real, 0.5)


def test_cli_writes_csv_and_meta(tmp_path, rows_by_split):
    counts = {s: dict(Counter("real" if lab == 0 else "fake" for _, lab in rows))
              for s, rows in rows_by_split.items()}
    out = tmp_path / "pairs.csv"
    code = mm.main(["--out", str(out), "--preview", "2"],
                   loader=lambda revision: (rows_by_split, counts, "abc123"))
    assert code == 0
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and list(rows[0]) == mm.FIELDS
    meta = json.loads((tmp_path / "pairs.meta.json").read_text(encoding="utf-8"))
    assert meta["revision"] == "abc123" and meta["total_rows"] == len(rows)


def test_iter_pairs_assembles_image_and_text(tmp_path, rows_by_split, built):
    _, table, _ = built
    out = tmp_path / "pairs.csv"
    mm.write_csv(table, str(out))
    ds = {s: [{"image": f"IMG-{s}-{i}", "text": t} for i, (t, _) in enumerate(rows)]
          for s, rows in rows_by_split.items()}
    seen = Counter()
    for image, text, label, r in mm.iter_pairs(str(out), ds, set_name="hard", group="train"):
        own = rows_by_split[r["image_split"]][int(r["image_row"])][0]
        assert image == f"IMG-{r['image_split']}-{r['image_row']}"
        assert (text == own) == (label == 0)
        seen[label] += 1
    assert seen[0] == seen[1] == 63
