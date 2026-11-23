"""prepare_all.py가 두 스크립트에 옵션을 제대로 넘기는지 확인한다. 실행: python -m pytest -q data_preparation"""
import prepare_all


def test_passes_options_to_both_scripts(monkeypatch):
    calls = {}
    monkeypatch.setattr(prepare_all.prepare_mismatch_data, "main", lambda argv: calls.setdefault("mismatch", argv))
    monkeypatch.setattr(prepare_all.prepare_ai_photos, "main", lambda argv: calls.setdefault("ai", argv))

    assert prepare_all.main([]) == 0
    assert calls == {"mismatch": [], "ai": ["--normalize"]}

    calls.clear()
    prepare_all.main(["--test-only"])
    assert calls["mismatch"] == ["--groups", "test_nyt", "test_bbc", "test_cnn"]
    assert calls["ai"][0] == "--normalize" and calls["ai"][1] == "--splits"
    assert calls["ai"][2:] == ["test1_nyt_mj", "test2_bbc_dalle", "test3_cnn_dalle",
                               "test4_bbc_sdxl", "test5_cnn_sdxl"]
