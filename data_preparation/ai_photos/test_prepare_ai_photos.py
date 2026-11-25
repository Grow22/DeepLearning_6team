"""가짜 parquet 파일로 prepare_ai_photos.py를 시험한다. 실행: python -m pytest -q data_preparation"""
import csv
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

import prepare_ai_photos as prep


def image_bytes(size, color, fmt):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "image": pa.array([{"bytes": b, "path": None} for b, _, _ in rows],
                          type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "label": pa.array([lab for _, lab, _ in rows], type=pa.int64()),
        "text": pa.array([t for _, _, t in rows], type=pa.string()),
    })
    pq.write_table(table, path, row_group_size=2)


@pytest.fixture()
def hub(tmp_path):
    """train: 파일 2개(행 0~2, 3~5). test1: 진짜 2장 + AI 2장. 2번 행은 256px보다 작다."""
    train = [
        (image_bytes((400, 300), (10, 0, 0), "JPEG"), 1, "t0"),
        (image_bytes((400, 300), (20, 0, 0), "JPEG"), 1, "t1"),
        (image_bytes((300, 200), (30, 0, 0), "JPEG"), 1, "t2"),
        (image_bytes((400, 300), (40, 0, 0), "JPEG"), 0, "t3"),
        (image_bytes((400, 300), (50, 0, 0), "JPEG"), 0, "t4"),
        (image_bytes((300, 200), (60, 0, 0), "JPEG"), 0, "t5"),
    ]
    test1 = [
        (image_bytes((512, 512), (0, 10, 0), "PNG"), 1, "ai caption 0"),
        (image_bytes((512, 512), (0, 20, 0), "PNG"), 1, "ai caption 1"),
        (image_bytes((600, 338), (0, 30, 0), "JPEG"), 0, "real caption 2"),
        (image_bytes((600, 338), (0, 40, 0), "JPEG"), 0, "real caption 3"),
    ]
    root = tmp_path / "hub"
    write_parquet(root / "data/train-00000-of-00002.parquet", train[:3])
    write_parquet(root / "data/train-00001-of-00002.parquet", train[3:])
    write_parquet(root / "data/test1_nyt_mj-00000-of-00001.parquet", test1)
    files = [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.parquet")]

    combined = tmp_path / "combined_test_pairs.csv"
    with open(combined, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "split", "image_row", "text_row", "is_ai", "is_mismatch"])
        w.writerows([
            ("test1_nyt_mj-0000", "test1_nyt_mj", 0, 1, 1, 1),
            ("test1_nyt_mj-0001", "test1_nyt_mj", 1, 0, 1, 1),
            ("test1_nyt_mj-0002", "test1_nyt_mj", 2, 2, 0, 0),
            ("test1_nyt_mj-0003", "test1_nyt_mj", 3, 3, 0, 0),
        ])
    (tmp_path / "combined_test_pairs.meta.json").write_text(json.dumps({"revision": "rev1"}), encoding="utf-8")

    # 교차 시험 세트: test1의 AI 행 0·1 ↔ 진짜 행 2·3이 같은 사건
    cross = tmp_path / "cross_test_pairs.csv"
    with open(cross, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "split", "image_row", "text_row", "is_ai", "text_fake"])
        w.writerows([
            ("cross-test1_nyt_mj-0000", "test1_nyt_mj", 2, 2, 0, 0),
            ("cross-test1_nyt_mj-0001", "test1_nyt_mj", 2, 0, 0, 1),
            ("cross-test1_nyt_mj-0002", "test1_nyt_mj", 0, 2, 1, 0),
            ("cross-test1_nyt_mj-0003", "test1_nyt_mj", 0, 0, 1, 1),
        ])
    (tmp_path / "cross_test_pairs.meta.json").write_text(json.dumps({"revision": "rev1"}), encoding="utf-8")
    return {"combined": combined, "cross": cross, "source": {"train": train, "test1_nyt_mj": test1},
            "list_files": lambda revision: files, "download": lambda revision, name: str(root / name)}


def run(hub, out, *extra):
    return prep.main(["--splits", "train", "test1_nyt_mj", "--combined", str(hub["combined"]),
                      "--cross", str(hub["cross"]), "--out", str(out), *extra],
                     list_files=hub["list_files"], download=hub["download"])


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_photo_table_matches_source(hub, tmp_path):
    out = tmp_path / "out"
    assert run(hub, out, "--normalize") == 0
    photos = {p["image_id"]: p for p in read(out / "ai_photos.csv")}
    assert len(photos) == 10
    for split, rows in hub["source"].items():
        for row, (data, label, _) in enumerate(rows):
            p = photos[f"{split}-{row:05d}"]
            assert (out / p["image_path"]).read_bytes() == data  # 두 번째 파일의 행 번호도 맞는지
            assert int(p["label"]) == label
            assert p["generator"] == ("Midjourney" if label == 1 else "")
            assert p["source"] == "NYT"
    assert photos["test1_nyt_mj-00000"]["format"] == "png"
    assert (photos["train-00002"]["width"], photos["train-00002"]["height"]) == ("300", "200")


def test_normalized_images_are_same_size_jpeg(hub, tmp_path):
    out = tmp_path / "out"
    run(hub, out, "--normalize")
    photos = {p["image_id"]: p for p in read(out / "ai_photos.csv")}
    assert photos["train-00002"]["normalized_path"] == ""  # 256px보다 작아서 뺀다
    assert photos["train-00005"]["normalized_path"] == ""
    for p in photos.values():
        if p["normalized_path"]:
            with Image.open(out / p["normalized_path"]) as im:
                assert (im.format, im.size, im.mode) == ("JPEG", (256, 256), "RGB")


def test_without_normalize_no_normalized_files(hub, tmp_path):
    out = tmp_path / "out"
    run(hub, out)
    assert all(p["normalized_path"] == "" for p in read(out / "ai_photos.csv"))
    assert not (out / "normalized").exists()


def test_combined_table_assembles_text_and_photo(hub, tmp_path):
    out = tmp_path / "out"
    run(hub, out, "--normalize")
    items = {c["item_id"]: c for c in read(out / "combined_test.csv")}
    assert list(next(iter(items.values()))) == prep.COMBINED_FIELDS
    assert items["test1_nyt_mj-0000"]["text"] == "ai caption 1"
    assert items["test1_nyt_mj-0000"]["image_path"] == "images/test1_nyt_mj/0.png"
    assert items["test1_nyt_mj-0002"]["text"] == "real caption 2"
    assert (items["test1_nyt_mj-0002"]["is_ai"], items["test1_nyt_mj-0002"]["is_mismatch"]) == ("0", "0")
    assert items["test1_nyt_mj-0003"]["normalized_path"] == "normalized/test1_nyt_mj/3.jpg"


def test_rerun_writes_nothing_new(hub, tmp_path, capsys):
    out = tmp_path / "out"
    run(hub, out, "--normalize")
    capsys.readouterr()
    run(hub, out, "--normalize")
    assert "새로 저장 0장" in capsys.readouterr().out


def test_cross_table_assembles_text_and_photo(hub, tmp_path):
    out = tmp_path / "out"
    run(hub, out, "--normalize")
    items = {c["item_id"]: c for c in read(out / "cross_test.csv")}
    assert list(next(iter(items.values()))) == prep.CROSS_FIELDS
    real_fake = items["cross-test1_nyt_mj-0001"]  # 진짜 사진 + 지어낸 글
    assert real_fake["image_path"] == "images/test1_nyt_mj/2.jpg"
    assert real_fake["text"] == "ai caption 0"
    assert (real_fake["is_ai"], real_fake["text_fake"]) == ("0", "1")
    ai_true = items["cross-test1_nyt_mj-0002"]  # AI 사진 + 기사 글
    assert ai_true["image_path"] == "images/test1_nyt_mj/0.png"
    assert ai_true["text"] == "real caption 2"
    assert (ai_true["is_ai"], ai_true["text_fake"]) == ("1", "0")


def test_different_revisions_stop(hub, tmp_path):
    meta = hub["cross"].with_suffix(".meta.json")
    meta.write_text(json.dumps({"revision": "rev2"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        run(hub, tmp_path / "out")
