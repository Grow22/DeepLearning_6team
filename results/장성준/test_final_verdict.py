"""final_verdict.py 시험. 실행: python -m pytest -q results"""
import numpy as np
import pytest

import final_verdict as fv


@pytest.mark.parametrize("fake,mismatch,ai,want", [
    (True, True, True, 0),     # 글이 거짓이면 다른 것과 상관없이 허위 (글이 거짓)
    (False, True, True, 1),    # 글은 사실이지만 사진과 안 맞으면 허위 (글-사진 불일치)
    (False, False, True, 2),   # 내용은 괜찮고 사진만 AI면 주의 — 허위로 치지 않음
    (False, False, False, 3),  # 정상
])
def test_decide_order(fake, mismatch, ai, want):
    assert fv.decide(fake, mismatch, ai) == want


def test_evaluate():
    truth = np.array([0, 0, 1, 2, 3, 3])
    pred = np.array([0, 3, 0, 2, 2, 3])
    m = fv.evaluate(truth, pred)
    assert m["acc4"] == pytest.approx(3 / 6)
    assert m["recall"] == pytest.approx(2 / 3)      # 진짜 허위 3장 중 2장을 허위로 판정 (두 번째 카드는 정상으로 놓침)
    assert m["precision"] == pytest.approx(2 / 2)   # 허위라고 한 2장이 모두 진짜 허위
    assert m["acc2"] == pytest.approx(5 / 6)
    assert m["per_cat"] == pytest.approx([0.5, 0.0, 1.0, 0.5])


def test_load_cached_refuses_missing_or_partial(tmp_path):
    with pytest.raises(SystemExit):
        fv.load_cached(tmp_path, "photo-x")
    folder = tmp_path / "photo-x"
    folder.mkdir()
    (folder / "ids.txt").write_text("a\nb", encoding="utf-8")
    np.save(folder / "0000.npy", np.zeros((1, 768), np.float32))
    with pytest.raises(SystemExit):
        fv.load_cached(tmp_path, "photo-x")  # 2장 중 1장만 계산된 경우
    np.save(folder / "0001.npy", np.ones((1, 768), np.float32))
    feats = fv.load_cached(tmp_path, "photo-x")
    assert set(feats) == {"a", "b"} and feats["b"][0] == 1
