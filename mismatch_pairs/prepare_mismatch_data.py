#!/usr/bin/env python3
"""짝 표(mismatch_pairs.csv)대로 MiRAGeNews 사진과 글을 받아, 바로 쓸 수 있는 폴더로 만든다.

실행 (저장소 맨 위 폴더에서):
    pip install huggingface_hub pyarrow pillow
    python mismatch_pairs/prepare_mismatch_data.py                                    # 전체
    python mismatch_pairs/prepare_mismatch_data.py --groups test_nyt test_bbc test_cnn  # 시험용만

결과 (data/ 폴더는 .gitignore에 들어 있어 깃에 올라가지 않는다):
    data/mismatch/images/<split>/<행 번호>.<확장자>    진짜 사진
    data/mismatch/easy.csv, data/mismatch/hard.csv     쌍 번호, 묶음, 사진 경로, 글, 정답
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = "anson-huang/mirage-news"
GROUPS = ["train", "validation", "test_nyt", "test_bbc", "test_cnn"]
SETS = ["easy", "hard"]
OUT_FIELDS = ["pair_id", "group", "image_path", "text", "label"]


def image_ext(data: bytes, path: str | None = None) -> str:
    """사진 바이트의 앞부분을 보고 확장자를 정한다. 원본 바이트는 그대로 저장한다."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if path and "." in path:
        return path.rsplit(".", 1)[1].lower()
    return "img"


def hub_list_files(revision: str) -> list[str]:
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(DATASET, repo_type="dataset", revision=revision)


def hub_download(revision: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(DATASET, filename, repo_type="dataset", revision=revision)


def split_files(all_files: list[str], split: str) -> list[str]:
    prefix = f"data/{split}-"
    files = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".parquet"))
    if not files:
        raise SystemExit(f"{split}: 데이터 파일을 찾지 못했어요.")
    return files


def export_split(local_files: list[str], need_images: set[int], need_texts: set[int],
                 image_dir: Path, rel_prefix: str):
    """split의 parquet 파일을 이름 순서대로 읽으며 필요한 사진은 저장하고 필요한 글은 모은다.

    행 번호 = 파일들을 이름 순서로 이어 붙였을 때의 순서 (datasets.load_dataset의 행 순서와 같음).
    """
    import pyarrow.parquet as pq

    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths, texts, written, offset = {}, {}, 0, 0
    for path in local_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["image", "text"]):
            text_col = batch.column("text").to_pylist()
            image_col = batch.column("image")
            image_bytes, image_names = image_col.field("bytes"), image_col.field("path")
            for k in range(batch.num_rows):
                row = offset + k
                if row in need_texts:
                    texts[row] = text_col[k]
                if row in need_images:
                    data = image_bytes[k].as_py()
                    if not data:
                        raise SystemExit(f"{path} {row}번 행: 사진 데이터가 비어 있어요.")
                    name = f"{row}.{image_ext(data, image_names[k].as_py())}"
                    target = image_dir / name
                    if not (target.exists() and target.stat().st_size == len(data)):
                        target.write_bytes(data)
                        written += 1
                    image_paths[row] = f"{rel_prefix}/{name}"
            offset += batch.num_rows
    missing_images = sorted(need_images - image_paths.keys())
    missing_texts = sorted(need_texts - texts.keys())
    if missing_images or missing_texts:
        raise SystemExit(f"데이터에 없는 행이 있어요: 사진 {missing_images[:5]}, 글 {missing_texts[:5]}")
    return image_paths, texts, written


def check_images(out: Path, rel_paths: list[str]) -> int:
    """저장한 사진이 모두 열리는지 확인한다. 확인한 장 수를 돌려준다."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow가 없어 사진 열기 확인은 건너뛰었어요. (pip install pillow)")
        return 0
    bad = []
    for rel in rel_paths:
        try:
            with Image.open(out / rel) as im:
                im.verify()
        except Exception:  # noqa: BLE001 - 어떤 이유든 열리지 않으면 알려 준다
            bad.append(rel)
    if bad:
        raise SystemExit(f"열리지 않는 사진이 {len(bad)}장 있어요: {bad[:5]}")
    return len(rel_paths)


def main(argv: list[str] | None = None, list_files=hub_list_files, download=hub_download) -> int:
    parser = argparse.ArgumentParser(description="짝 데이터용 MiRAGeNews 사진·글 내려받기")
    parser.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS,
                        help="받을 묶음 (기본: 전부)")
    parser.add_argument("--sets", nargs="+", default=SETS, choices=SETS, help="만들 세트 (기본: easy hard)")
    parser.add_argument("--pairs", default=str(HERE / "mismatch_pairs.csv"), help="짝 표 경로")
    parser.add_argument("--out", default=str(HERE.parent / "data" / "mismatch"), help="저장 폴더")
    args = parser.parse_args(argv)

    started = time.time()
    pairs_path = Path(args.pairs)
    meta = json.loads(pairs_path.with_name(pairs_path.stem + ".meta.json").read_text(encoding="utf-8"))
    revision = meta["revision"]
    with open(pairs_path, newline="", encoding="utf-8") as f:
        pairs = [r for r in csv.DictReader(f) if r["group"] in args.groups and r["set"] in args.sets]

    need_images, need_texts = defaultdict(set), defaultdict(set)
    for r in pairs:
        need_images[r["image_split"]].add(int(r["image_row"]))
        need_texts[r["text_split"]].add(int(r["text_row"]))

    out = Path(args.out)
    all_files = list_files(revision)
    image_paths, texts, written = {}, {}, 0
    for split in sorted(set(need_images) | set(need_texts)):
        files = split_files(all_files, split)
        print(f"{split}: 파일 {len(files)}개 받는 중...", flush=True)
        local = [download(revision, name) for name in files]
        paths, split_texts, n = export_split(local, need_images[split], need_texts[split],
                                             out / "images" / split, f"images/{split}")
        image_paths.update({(split, row): p for row, p in paths.items()})
        texts.update({(split, row): t for row, t in split_texts.items()})
        written += n

    for set_name in args.sets:
        with open(out / f"{set_name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
            writer.writeheader()
            for r in pairs:
                if r["set"] != set_name:
                    continue
                writer.writerow({
                    "pair_id": r["pair_id"],
                    "group": r["group"],
                    "image_path": image_paths[(r["image_split"], int(r["image_row"]))],
                    "text": texts[(r["text_split"], int(r["text_row"]))],
                    "label": r["label"],
                })

    checked = check_images(out, list(image_paths.values()))
    size_mb = sum((out / p).stat().st_size for p in image_paths.values()) / 1e6
    print(f"\n완료 ({time.time() - started:.0f}초) → {out}")
    print(f"- 사진 {len(image_paths)}장, {size_mb:.0f}MB (이번에 새로 저장 {written}장, 열기 확인 {checked}장)")
    for set_name in args.sets:
        n = sum(1 for r in pairs if r["set"] == set_name)
        print(f"- {set_name}.csv: {n}쌍 (묶음: {', '.join(args.groups)})")
    print("- 정답(label): 0 = 맞음, 1 = 안 맞음. 판정할 때는 사진과 글만 쓰고, label은 마지막 채점에만 써요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
