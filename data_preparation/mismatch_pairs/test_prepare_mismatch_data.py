"""가짜 parquet 파일로 prepare_mismatch_data.py를 시험한다. 실행: python -m pytest -q data_preparation"""
import csv
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

import prepare_mismatch_data as prep


def image_bytes(color, fmt):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format=fmt)
    return buf.getvalue()


def write_parquet(path, rows):
    """MiRAGeNews와 같은 모양(image{bytes,path}, label, text)의 parquet을 만든다."""
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
    """train은 파일 2개(행 0~2, 3~5), test1은 파일 1개. 이름이 비슷한 다른 파일도 넣어 둔다."""
    train = [(image_bytes((i * 40, 0, 0), "JPEG" if i % 2 else "PNG"), 0 if i >= 2 else 1, f"train text {i}")
             for i in range(6)]
    test1 = [(image_bytes((0, i * 60, 0), "JPEG"), 0, f"nyt text {i}") for i in range(3)]
    root = tmp_path / "hub"
    write_parquet(root / "data/train-00000-of-00002.parquet", train[:3])
    write_parquet(root / "data/train-00001-of-00002.parquet", train[3:])
    write_parquet(root / "data/test1_nyt_mj-00000-of-00001.parquet", test1)
    write_parquet(root / "data/test-00000-of-00001.parquet", [(image_bytes((9, 9, 9), "PNG"), 0, "decoy")])
    files = [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.parquet")]

    pairs = [  # pair_id, set, group, image_split, image_row, text_split, text_row, label, text_sim
        ("easy-train-00000", "easy", "train", "train", 4, "train", 2, 1, 0.0),
        ("easy-train-00001", "easy", "train", "train", 2, "train", 2, 0, 1.0),
        ("easy-train-00002", "easy", "train", "train", 2, "train", 4, 1, 0.0),
        ("easy-train-00003", "easy", "train", "train", 4, "train", 4, 0, 1.0),
        ("hard-train-00000", "hard", "train", "train", 4, "train", 5, 1, 0.2),
        ("easy-test_nyt-00000", "easy", "test_nyt", "test1_nyt_mj", 1, "test1_nyt_mj", 1, 0, 1.0),
        ("easy-test_nyt-00001", "easy", "test_nyt", "test1_nyt_mj", 1, "test1_nyt_mj", 2, 1, 0.0),
    ]
    pairs_path = tmp_path / "mismatch_pairs.csv"
    with open(pairs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "set", "group", "image_split", "image_row",
                    "text_split", "text_row", "label", "text_sim"])
        w.writerows(pairs)
    (tmp_path / "mismatch_pairs.meta.json").write_text(json.dumps({"revision": "rev1"}), encoding="utf-8")
    source = {"train": train, "test1_nyt_mj": test1}
    return {
        "root": root, "pairs": pairs_path, "source": source,
        "list_files": lambda revision: files,
        "download": lambda revision, name: str(root / name),
    }


def run(hub, out, *extra):
    return prep.main(["--pairs", str(hub["pairs"]), "--out", str(out), *extra],
                     list_files=hub["list_files"], download=hub["download"])


def read(out, set_name):
    with open(out / f"{set_name}.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_export_matches_source_rows(hub, tmp_path):
    out = tmp_path / "out"
    assert run(hub, out) == 0
    easy = read(out, "easy")
    assert list(easy[0]) == prep.OUT_FIELDS  # 행 번호·유사도 칸은 없다
    expected = {
        "easy-train-00000": ("train", 4, "train text 2", "1"),
        "easy-train-00001": ("train", 2, "train text 2", "0"),
        "easy-train-00002": ("train", 2, "train text 4", "1"),
        "easy-train-00003": ("train", 4, "train text 4", "0"),
        "easy-test_nyt-00000": ("test1_nyt_mj", 1, "nyt text 1", "0"),
        "easy-test_nyt-00001": ("test1_nyt_mj", 1, "nyt text 2", "1"),
    }
    assert {r["pair_id"] for r in easy} == set(expected)
    for r in easy:
        split, row, text, label = expected[r["pair_id"]]
        assert (out / r["image_path"]).read_bytes() == hub["source"][split][row][0]  # 두 번째 파일의 행 번호도 맞는지
        assert r["text"] == text and r["label"] == label
    assert [r["pair_id"] for r in read(out, "hard")] == ["hard-train-00000"]
    assert read(out, "hard")[0]["text"] == "train text 5"


def test_extension_follows_image_format(hub, tmp_path):
    out = tmp_path / "out"
    run(hub, out)
    assert (out / "images/train/2.png").exists()  # 짝수 행은 PNG
    assert (out / "images/train/4.png").exists()
    assert (out / "images/test1_nyt_mj/1.jpg").exists()


def test_rerun_writes_nothing_new(hub, tmp_path, capsys):
    out = tmp_path / "out"
    run(hub, out)
    capsys.readouterr()
    run(hub, out)
    assert "새로 저장 0장" in capsys.readouterr().out


def test_groups_option_limits_output(hub, tmp_path):
    out = tmp_path / "out"
    run(hub, out, "--groups", "test_nyt", "--sets", "easy")
    assert {r["group"] for r in read(out, "easy")} == {"test_nyt"}
    assert not (out / "images" / "train").exists()
    assert not (out / "hard.csv").exists()


def test_missing_row_stops_with_message(hub, tmp_path):
    with open(hub["pairs"], "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["easy-train-00009", "easy", "train", "train", 99, "train", 2, 1, 0.0])
    with pytest.raises(SystemExit):
        run(hub, tmp_path / "out")


def test_image_ext():
    assert prep.image_ext(b"\xff\xd8\xff\xe0rest") == "jpg"
    assert prep.image_ext(b"\x89PNG\r\n\x1a\nrest") == "png"
    assert prep.image_ext(b"RIFF1234WEBPrest") == "webp"
    assert prep.image_ext(b"unknown", "photo.tif") == "tif"
