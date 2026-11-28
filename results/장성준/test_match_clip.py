"""match_clip.py 시험 (진짜 CLIP 대신, 같은 사건이면 같은 특징을 내는 가짜 특징으로). 실행: python -m pytest -q results"""
import csv

import numpy as np

import match_clip as mc

DIM = 768


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def caption(e):
    return f"Caption about event {e}."


def test_helpers():
    assert mc.image_id_from_path("images/train/8448.jpg") == "train-08448"
    img = np.array([[3.0, 4.0] + [0.0] * (DIM - 2)])
    feats = mc.pair_features(img, img)
    assert feats.shape == (1, DIM + 1) and abs(feats[0, -1] - 1.0) < 1e-6  # 마지막 칸 = 코사인 유사도


def test_text_features_are_cached(tmp_path):
    calls = []

    def encode(texts):
        calls.extend(texts)
        return np.ones((len(texts), DIM), np.float32) * len(texts[0])
    a = mc.embed_texts(["ab", "abc", "ab"], encode, tmp_path / "t.npz")
    b = mc.embed_texts(["abc", " ab "], encode, tmp_path / "t.npz")
    assert a.shape == (3, DIM) and np.array_equal(a[1], b[0]) and np.array_equal(a[0], b[1])
    assert sorted(calls) == ["ab", "abc"]  # 같은 글은 한 번만 계산


def test_end_to_end_with_fake_features(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    vec = rng.normal(size=(20, DIM)).astype(np.float32)  # 사건마다 특징 하나
    d, event_of, photos, mismatch, combined, cards, ocr = tmp_path / "data", {}, [], [], [], [], []
    for split, group, events in (("train", "train", range(0, 8)), ("validation", "validation", range(8, 12)),
                                 ("test1_nyt_mj", "test_nyt", range(12, 16))):
        rows = []
        for row, e in enumerate(events):
            image_id = f"{split}-{row:05d}"
            event_of[image_id] = e
            photos.append({"image_id": image_id, "split": split, "label": 0, "image_path": f"images/{split}/{row}.jpg"})
            rows.append((f"images/{split}/{row}.jpg", e))
        for k, (path, e) in enumerate(rows):
            other = rows[(k + 1) % len(rows)][1]
            mismatch += [{"pair_id": f"{group}-{k}-0", "group": group, "image_path": path, "text": caption(e), "label": 0},
                         {"pair_id": f"{group}-{k}-1", "group": group, "image_path": path, "text": caption(other), "label": 1}]
            if split == "test1_nyt_mj":
                bad = k % 2
                item = f"{split}-{k:04d}"
                text = caption(other if bad else e)
                combined.append({"item_id": item, "split": split, "image_path": path, "text": text, "is_ai": 0,
                                 "is_mismatch": bad})
                card_id = f"card-{split}-{k}"
                event_of[card_id] = e  # 카드에서 잘라낸 사진 = 같은 사진
                cards.append({"card_id": card_id, "split": split, "combined_item_id": item, "is_mismatch": bad,
                              "card_path": f"cards/{k}.jpg", "box_x1": 0, "box_y1": 0, "box_x2": 10, "box_y2": 10})
                ocr.append({"card_id": card_id, "ocr_text": text})
    write_csv(d / "ai_photos" / "ai_photos.csv", photos)
    write_csv(d / "mismatch" / "easy.csv", mismatch)
    write_csv(d / "mismatch" / "hard.csv", [{**r, "pair_id": "h" + r["pair_id"]} for r in mismatch])
    write_csv(d / "ai_photos" / "combined_test.csv", combined)
    write_csv(d / "cards" / "cards.csv", cards)
    write_csv(d / "cards" / "ocr.csv", ocr)

    monkeypatch.setattr(mc.univfd, "embed", lambda name, ids, load, model, feat_dir, batch=16:
                        np.stack([vec[event_of[i]] for i in ids]))

    def encode(texts):
        return np.stack([vec[int(t.split()[-1].rstrip("."))] for t in texts])

    assert mc.main(["--data", str(d)], image_model=object(), text_encoder=encode) == 0
    table = (d / "match" / "report.md").read_text(encoding="utf-8")
    assert "| 통합 시험 원본 (사진 + 원래 글) | 4 | 100.0% | 1.000 |" in table
    assert "| 통합 시험 카드 (잘라낸 사진 + 읽어낸 글) | 4 | 100.0% | 1.000 |" in table
    with open(d / "match" / "scores.csv", newline="", encoding="utf-8") as f:
        sets = {r["set"] for r in csv.DictReader(f)}
    assert sets == {"easy-test_nyt", "hard-test_nyt", "combined", "card"}  # 학습·검증 문항은 점수 파일에 안 넣음
