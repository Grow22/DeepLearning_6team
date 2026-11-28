"""univfd.py 시험 (진짜 CLIP 대신 사진 평균색을 특징으로 쓰는 가짜 모델로). 실행: python -m pytest -q results"""
import csv

import numpy as np
from PIL import Image

import score_cards as sc
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


def test_preprocess_shape_and_values():
    for size in [(10, 10), (3000, 100), (100, 3000), (224, 224)]:
        x = uf.preprocess(Image.new("RGB", size, (128, 64, 32)))
        assert x.shape == (3, 224, 224) and x.dtype == np.float32
        expect = (np.array([128, 64, 32]) / 255 - uf.MEAN) / uf.STD
        assert np.allclose(x.mean(axis=(1, 2)), expect, atol=1e-2)


def test_embed_saves_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(uf, "CHUNK", 4)
    images = [Image.new("RGB", (50, 40), (i * 20, 0, 0)) for i in range(10)]
    ids = [f"p{i}" for i in range(10)]
    model = ColorModel()
    first = uf.embed("set", ids, lambda k: images[k], model, tmp_path, batch=3)
    assert first.shape == (10, 768) and model.calls == 10
    (tmp_path / "set" / "0002.npy").unlink()  # 중간에 멈춘 것처럼 마지막 조각을 지움
    again = uf.embed("set", ids, lambda k: images[k], model, tmp_path, batch=3)
    assert np.array_equal(first, again) and model.calls == 12  # 지운 조각(2장)만 다시 계산
    uf.embed("set", ids[:5], lambda k: images[k], model, tmp_path, batch=3)
    assert model.calls == 17  # 사진 목록이 바뀌면 처음부터 다시


def test_main_end_to_end_with_fake_model(tmp_path):
    """AI 사진은 붉게, 진짜 사진은 푸르게 만든 가짜 데이터로 처음부터 끝까지 돌린다."""
    d, rng, photos, cards = tmp_path / "data", np.random.default_rng(0), [], []
    for split, n in [("train", 20), ("validation", 10)] + [(s, 4) for s in uf.TEST_SPLITS]:
        (d / "ai_photos" / "images" / split).mkdir(parents=True)
        for i in range(n):
            label = int(i < n // 2)
            color = (200, 60, 60) if label else (60, 60, 200)
            arr = np.clip(rng.normal(color, 25, size=(60, 80, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(d / "ai_photos" / "images" / split / f"{i}.png")
            photos.append({"image_id": f"{split}-{i:05d}", "split": split, "label": label,
                           "image_path": f"images/{split}/{i}.png"})
    write_csv(d / "ai_photos" / "ai_photos.csv", list(photos[0]), photos)
    for split in uf.TEST_SPLITS:
        (d / "cards" / "cards" / split).mkdir(parents=True)
        for i in range(4):
            for tf in (0, 1):
                card = Image.new("RGB", (108, 135), (128, 128, 128))
                with Image.open(d / "ai_photos" / "images" / split / f"{i}.png") as p:
                    card.paste(p.convert("RGB").resize((108, 78)), (0, 0))
                rel = f"cards/{split}/{i}-{tf}.jpg"
                card.save(d / "cards" / rel, quality=95)
                cards.append({"card_id": f"card-{split}-{i}-{tf}", "split": split, "image_id": f"{split}-{i:05d}",
                              "cross_item_id": f"x-{split}-{i}-{tf}", "is_ai": int(i < 2), "text_fake": tf,
                              "card_path": rel, "box_x1": 0, "box_y1": 0, "box_x2": 108, "box_y2": 78})
    write_csv(d / "cards" / "cards.csv", list(cards[0]), cards)

    out = tmp_path / "out"
    assert uf.main(["--data", str(d), "--out", str(out)], model=ColorModel()) == 0
    items = sc.read_scores(out / "scores.csv", *sc.load_truth(d))
    assert {k: len(v) for k, v in items.items()} == {"original": 20, "card": 40, "crop": 40}
    result = sc.evaluate(items, 0.5)
    assert result["all"]["original"]["acc"] == 1.0 and result["all"]["crop"]["acc"] == 1.0
    assert (out / "head.json").exists()
