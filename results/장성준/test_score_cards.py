"""score_cards.py 시험. 실행: python -m pytest -q results"""
import csv

import numpy as np
import pytest

import score_cards as sc

SPLIT = "test1_nyt_mj"


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def data(tmp_path):
    """사진 4장(0·1번 AI, 2·3번 진짜)과, 사진마다 사실 글·거짓 글 교차 시험 카드 2장."""
    d = tmp_path / "data"
    (d / "ai_photos").mkdir(parents=True)
    (d / "cards").mkdir()
    write_csv(d / "ai_photos" / "ai_photos.csv", ["image_id", "split", "label"],
              [{"image_id": f"{SPLIT}-{i:05d}", "split": SPLIT, "label": int(i < 2)} for i in range(4)])
    cards = [{"card_id": f"card-{i}-{tf}", "split": SPLIT, "image_id": f"{SPLIT}-{i:05d}", "is_ai": int(i < 2),
              "text_fake": tf, "cross_item_id": f"cross-{i}-{tf}"} for i in range(4) for tf in (0, 1)]
    write_csv(d / "cards" / "cards.csv", list(cards[0]), cards)
    return d


def write_scores(path, rows):
    write_csv(path, ["input", "id", "score"], [dict(zip(["input", "id", "score"], r)) for r in rows])


def test_auc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300) + y * 0.3, 1)  # 같은 점수가 많음
    assert sc.auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(sc.auc([1, 1], [0.2, 0.3]))  # 한쪽 정답만 있으면 계산 안 함


def test_scores_are_graded_against_the_tables(data, tmp_path):
    rows = [("original", f"{SPLIT}-{i:05d}", s) for i, s in enumerate([0.9, 0.4, 0.2, 0.1])]
    card = {(0, 0): 0.6, (0, 1): 0.4, (1, 0): 0.7, (1, 1): 0.8, (2, 0): 0.3, (2, 1): 0.2, (3, 0): 0.1, (3, 1): 0.1}
    rows += [("card", f"card-{i}-{tf}", s) for (i, tf), s in card.items()]
    write_scores(tmp_path / "s.csv", rows)
    result = sc.evaluate(sc.read_scores(tmp_path / "s.csv", *sc.load_truth(data)), 0.5)

    orig = result["all"]["original"]
    assert (orig["n"], orig["acc"], orig["real_acc"], orig["ai_acc"], orig["auc"]) == (4, 0.75, 1.0, 0.5, 1.0)
    assert result["splits"][SPLIT]["original"] == orig
    assert result["flips"]["card"] == {"pairs": 4, "flip": 0.25, "diff": pytest.approx(0.1)}  # 0번 사진만 바뀜
    assert result["all"]["crop"]["n"] == 0
    text = sc.report(result, 0.5)
    assert "75.0% (1.000)" in text and "| ③ 잘라낸 사진 | - | - | 0 |" in text


@pytest.mark.parametrize("row", [
    ("photo", f"{SPLIT}-00000", "0.5"),      # 모르는 입력 종류
    ("original", "nope", "0.5"),             # 모르는 id
    ("card", f"{SPLIT}-00000", "0.5"),       # 카드인데 사진 id
    ("original", f"{SPLIT}-00000", "abc"),   # 점수가 숫자가 아님
])
def test_bad_rows_stop(data, tmp_path, row):
    write_scores(tmp_path / "s.csv", [row])
    with pytest.raises(SystemExit):
        sc.read_scores(tmp_path / "s.csv", *sc.load_truth(data))


def test_duplicate_rows_stop(data, tmp_path):
    write_scores(tmp_path / "s.csv", [("crop", "card-0-0", "0.1"), ("crop", "card-0-0", "0.2")])
    with pytest.raises(SystemExit):
        sc.read_scores(tmp_path / "s.csv", *sc.load_truth(data))
