"""edit_strength.py 시험 (진짜 CLIP 대신 사진 평균색을 특징으로 쓰는 가짜 모델로). 실행: python -m pytest -q results"""
import csv
import pickle

import numpy as np
import pytest
from PIL import Image

import edit_strength as es
import univfd as uf


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ColorModel:
    """사진 묶음 → 채널별 평균을 768칸으로 늘린 특징. 몇 장 계산했는지 센다."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += len(x)
        return np.tile(x.mean(axis=(2, 3)), (1, 256))


def noise(w, h):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))


def bright_rows(img) -> int:
    """흰 글씨(200 이상)가 있는 줄 수."""
    return int((np.asarray(img).min(axis=2) >= 200).any(axis=1).sum())


def test_jpeg_lower_quality_loses_more_detail():
    img = noise(64, 48)
    base = np.asarray(img, dtype=float)
    out = {q: es.jpeg(img, q) for q in (95, 30)}
    err = {q: np.abs(np.asarray(o, dtype=float) - base).mean() for q, o in out.items()}
    assert out[30].size == (64, 48)
    assert 0 < err[95] < err[30]


@pytest.mark.parametrize("size, ratio, want", [((80, 60), 0.5, (40, 30)), ((80, 60), 0.35, (28, 21)),
                                               ((3, 2), 0.25, (1, 1))])
def test_shrink_scales_both_sides(size, ratio, want):
    assert es.shrink(Image.new("RGB", size), ratio).size == want


def test_cover_darkens_only_the_bottom_band_and_writes_text():
    img = Image.new("RGB", (400, 200), (128, 128, 128))
    out = np.asarray(es.cover(img, 0.3, "Breaking news from the city hall today " * 10, es.default_font()), dtype=int)
    assert (out[:140] == 128).all()          # 위 70%는 그대로
    assert np.median(out[140:]) <= 40         # 아래 30%는 어둡게 (128 → 약 20)
    assert out[140:].max() >= 200             # 흰 글씨가 있음
    empty = np.asarray(es.cover(img, 0.3, "", es.default_font()), dtype=int)
    assert (empty[:140] == 128).all() and empty[140:].max() < 60


def test_cover_writes_more_lines_when_it_covers_more():
    img, text = Image.new("RGB", (400, 200), (128, 128, 128)), "Breaking news from the city hall today " * 10
    assert bright_rows(es.cover(img, 0.5, text, es.default_font())) > bright_rows(es.cover(img, 0.2, text, es.default_font()))


def test_apply_edit_runs_the_named_edit():
    img = Image.new("RGB", (80, 60), (128, 128, 128))
    assert es.apply_edit(img, "shrink", "0.5", "", es.default_font()).size == (40, 30)
    covered = np.asarray(es.apply_edit(img, "cover", "0.5", "", es.default_font()))
    assert covered.shape == (60, 80, 3) and (covered[:30] == 128).all() and covered[30:].max() < 60
    with pytest.raises(ValueError):
        es.apply_edit(img, "blur", "1", "", es.default_font())


def test_summarize_groups_by_split_group():
    rows = [("test1_nyt_mj", 1, 0.9), ("test1_nyt_mj", 0, 0.2),
            ("test4_bbc_sdxl", 1, 0.4), ("test4_bbc_sdxl", 0, 0.1),
            ("test5_cnn_sdxl", 1, 0.8), ("test5_cnn_sdxl", 0, 0.7)]
    got = es.summarize_groups(rows)
    assert got["test1"]["n"] == 2 and got["test1"]["acc"] == 1.0
    sdxl = got["sdxl"]  # AI 0.4 ✗, 0.8 ✓ / 진짜 0.1 ✓, 0.7 ✗
    assert sdxl["n"] == 4 and sdxl["acc"] == 0.5 and sdxl["ai_acc"] == 0.5 and sdxl["real_acc"] == 0.5
    assert sdxl["auc"] == 0.75  # (0.4>0.1 ✓) (0.4>0.7 ✗) (0.8>0.1 ✓) (0.8>0.7 ✓)


def make_data(tmp_path):
    """AI 사진은 붉게, 진짜 사진은 푸르게 만든 가짜 데이터 + univfd.py가 남겨 두는 원본 특징·분류기."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d, rng, photos, cross = tmp_path / "data", np.random.default_rng(0), [], []
    for split in es.SPLITS:
        (d / "ai_photos" / "images" / split).mkdir(parents=True)
        for i in range(4):
            label = int(i < 2)
            color = (200, 60, 60) if label else (60, 60, 200)
            arr = np.clip(rng.normal(color, 25, size=(60, 80, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(d / "ai_photos" / "images" / split / f"{i}.jpg")
            path = f"images/{split}/{i}.jpg"
            photos.append({"image_id": f"{split}-{i:05d}", "split": split, "label": label, "source": "x",
                           "generator": "x", "image_path": path, "format": "jpg", "width": 80, "height": 60,
                           "normalized_path": ""})
            for fake in (0, 1):
                cross.append({"item_id": f"{split}-{i}-{fake}", "split": split, "image_path": path,
                              "text": f"caption {i} fake={fake}", "is_ai": label, "text_fake": fake,
                              "normalized_path": ""})
    write_csv(d / "ai_photos" / "ai_photos.csv", list(photos[0]), photos)
    write_csv(d / "ai_photos" / "cross_test.csv", list(cross[0]), cross)
    model, feat_dir, xs, ys = ColorModel(), d / "univfd" / "features", [], []
    for split in es.SPLITS:
        rows = [p for p in photos if p["split"] == split]
        xs.append(uf.embed(f"photo-{split}", [r["image_id"] for r in rows],
                           lambda k: uf.open_rgb(d / "ai_photos" / rows[k]["image_path"]), model, feat_dir))
        ys += [r["label"] for r in rows]
    clf = make_pipeline(StandardScaler(), LogisticRegression()).fit(uf.l2(np.concatenate(xs)), ys)
    (d / "univfd" / "head.pkl").write_bytes(pickle.dumps(clf))
    return d


def test_main_end_to_end_with_fake_model(tmp_path):
    d = make_data(tmp_path)
    cache = d / "univfd" / "features" / "photo-test1_nyt_mj"
    before = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    model, out = ColorModel(), tmp_path / "out"
    assert es.main(["--data", str(d), "--out", str(out)], model=model) == 0
    n_levels = sum(len(v) for v in es.LEVELS.values())
    assert model.calls == n_levels * 12  # 원본은 저장된 특징을 쓰고, 편집한 사진만 계산
    with open(out / "summary.csv", newline="", encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == (n_levels + len(es.LEVELS)) * len(es.GROUPS)  # 편집마다 맨 앞에 원본 단계
    base = [r for r in summary if r["edit"] == "jpeg" and r["level"] == "원본" and r["group"] == "test1"]
    assert len(base) == 1 and float(base[0]["acc"]) == 1.0 and base[0]["n"] == "4"
    assert (out / "edit_strength.png").stat().st_size > 0
    assert {p.name: p.stat().st_mtime_ns for p in cache.iterdir()} == before  # univfd.py 특징은 건드리지 않음
    es.main(["--data", str(d), "--out", str(out)], model=model)
    assert model.calls == n_levels * 12  # 다시 실행하면 저장해 둔 특징을 씀


def test_main_needs_univfd_results_first(tmp_path):
    d = make_data(tmp_path)
    (d / "univfd" / "head.pkl").unlink()
    with pytest.raises(SystemExit, match="univfd.py"):
        es.main(["--data", str(d), "--out", str(tmp_path / "out")], model=ColorModel())
