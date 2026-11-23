#!/usr/bin/env python3
"""MiRAGeNews 사진(진짜 + AI)을 받아 AI 사진 판정용 폴더로 만든다. 통합 시험 세트도 함께 조립한다.

실행 (저장소 맨 위 폴더에서):
    pip install huggingface_hub pyarrow pillow
    python ai_photos/prepare_ai_photos.py --normalize
    python ai_photos/prepare_ai_photos.py --normalize --splits test1_nyt_mj test2_bbc_dalle   # 일부만

결과 (data/ 폴더는 .gitignore에 들어 있어 깃에 올라가지 않는다):
    data/ai_photos/images/<split>/<행 번호>.<확장자>   원본 사진 (받은 그대로)
    data/ai_photos/normalized/<split>/<행 번호>.jpg    --normalize: 형식·크기를 맞춘 사진 (가운데 256x256 + 같은 품질 JPEG)
    data/ai_photos/ai_photos.csv                       사진마다 split, 정답, 출처, 생성기, 형식, 크기, 경로
    data/ai_photos/combined_test.csv                   통합 시험 세트 (사진 경로, 글, AI 여부, 글 불일치 여부)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = "anson-huang/mirage-news"
SPLITS = ["train", "validation", "test1_nyt_mj", "test2_bbc_dalle", "test3_cnn_dalle",
          "test4_bbc_sdxl", "test5_cnn_sdxl"]
# split별 뉴스 출처와 AI 사진 생성기 (MiRAGeNews 설명 기준)
SOURCE = {
    "train": ("NYT", "Midjourney"), "validation": ("NYT", "Midjourney"),
    "test1_nyt_mj": ("NYT", "Midjourney"),
    "test2_bbc_dalle": ("BBC", "DALL-E 3"), "test3_cnn_dalle": ("CNN", "DALL-E 3"),
    "test4_bbc_sdxl": ("BBC", "SDXL"), "test5_cnn_sdxl": ("CNN", "SDXL"),
}
CROP = 256          # 형식·크기 맞추기: 줄이거나 늘리지 않고 가운데를 이 크기로 자른다
JPEG_QUALITY = 95   # 형식·크기 맞추기: 모든 사진을 같은 품질의 JPEG로 저장
PHOTO_FIELDS = ["image_id", "split", "label", "source", "generator", "image_path",
                "format", "width", "height", "normalized_path"]
COMBINED_FIELDS = ["item_id", "split", "image_path", "text", "is_ai", "is_mismatch", "normalized_path"]


def image_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
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


def normalize_image(data: bytes, target: Path) -> bool:
    """가운데를 CROP x CROP으로 잘라(크기 변경 없음) JPEG로 저장한다. 사진이 더 작으면 False."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        if min(w, h) < CROP:
            return False
        left, top = (w - CROP) // 2, (h - CROP) // 2
        im.convert("RGB").crop((left, top, left + CROP, top + CROP)).save(
            target, format="JPEG", quality=JPEG_QUALITY)
    return True


def export_split(local_files: list[str], split: str, out: Path, normalize: bool, need_texts: set[int]):
    """split의 parquet 파일을 이름 순서대로 읽으며 사진을 저장하고, 사진 정보와 필요한 글을 돌려준다.

    행 번호 = 파일들을 이름 순서로 이어 붙였을 때의 순서 (datasets.load_dataset의 행 순서와 같음).
    """
    import pyarrow.parquet as pq
    from PIL import Image

    source, generator = SOURCE[split]
    raw_dir, norm_dir = out / "images" / split, out / "normalized" / split
    raw_dir.mkdir(parents=True, exist_ok=True)
    if normalize:
        norm_dir.mkdir(parents=True, exist_ok=True)
    photos, texts, written, too_small, offset = [], {}, 0, 0, 0
    for path in local_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image", "label", "text"]):
            labels = batch.column("label").to_pylist()
            text_col = batch.column("text").to_pylist()
            image_bytes = batch.column("image").field("bytes")
            for k, label in enumerate(labels):
                row = offset + k
                if row in need_texts:
                    texts[row] = text_col[k]
                data = image_bytes[k].as_py()
                if not data:
                    raise SystemExit(f"{path} {row}번 행: 사진 데이터가 비어 있어요.")
                ext = image_ext(data)
                with Image.open(io.BytesIO(data)) as im:
                    width, height = im.size
                    im.verify()  # 깨진 사진이면 여기서 멈춘다
                target = raw_dir / f"{row}.{ext}"
                if not (target.exists() and target.stat().st_size == len(data)):
                    target.write_bytes(data)
                    written += 1
                normalized_path = ""
                if normalize:
                    norm_target = norm_dir / f"{row}.jpg"
                    if norm_target.exists() or normalize_image(data, norm_target):
                        normalized_path = f"normalized/{split}/{row}.jpg"
                    else:
                        too_small += 1
                photos.append({
                    "image_id": f"{split}-{row:05d}", "split": split, "label": int(label),
                    "source": source, "generator": generator if label == 1 else "",
                    "image_path": f"images/{split}/{row}.{ext}", "format": ext,
                    "width": width, "height": height, "normalized_path": normalized_path,
                })
            offset += batch.num_rows
    return photos, texts, written, too_small


def main(argv: list[str] | None = None, list_files=hub_list_files, download=hub_download) -> int:
    parser = argparse.ArgumentParser(description="AI 사진 판정용 MiRAGeNews 사진 내려받기")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS, help="받을 split (기본: 전부)")
    parser.add_argument("--normalize", action="store_true",
                        help=f"가운데 {CROP}x{CROP} 자르기 + JPEG {JPEG_QUALITY} 버전도 만들기")
    parser.add_argument("--combined", default=str(HERE / "combined_test_pairs.csv"), help="통합 시험 세트 표")
    parser.add_argument("--out", default=str(HERE.parent / "data" / "ai_photos"), help="저장 폴더")
    args = parser.parse_args(argv)

    started = time.time()
    combined_path = Path(args.combined)
    meta = json.loads(combined_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    revision = meta["revision"]
    with open(combined_path, newline="", encoding="utf-8") as f:
        combined = [r for r in csv.DictReader(f) if r["split"] in args.splits]
    need_texts = defaultdict(set)
    for r in combined:
        need_texts[r["split"]].add(int(r["text_row"]))

    out = Path(args.out)
    all_files = list_files(revision)
    photos, texts, written, too_small = [], {}, 0, 0
    for split in args.splits:
        files = split_files(all_files, split)
        print(f"{split}: 파일 {len(files)}개 받는 중...", flush=True)
        local = [download(revision, name) for name in files]
        split_photos, split_texts, n_written, n_small = export_split(
            local, split, out, args.normalize, need_texts[split])
        photos += split_photos
        texts.update({(split, row): t for row, t in split_texts.items()})
        written += n_written
        too_small += n_small

    with open(out / "ai_photos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PHOTO_FIELDS)
        writer.writeheader()
        writer.writerows(photos)

    by_id = {p["image_id"]: p for p in photos}
    with open(out / "combined_test.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_FIELDS)
        writer.writeheader()
        for r in combined:
            photo = by_id[f"{r['split']}-{int(r['image_row']):05d}"]
            writer.writerow({
                "item_id": r["item_id"], "split": r["split"], "image_path": photo["image_path"],
                "text": texts[(r["split"], int(r["text_row"]))], "is_ai": r["is_ai"],
                "is_mismatch": r["is_mismatch"], "normalized_path": photo["normalized_path"],
            })

    size_mb = sum((out / p["image_path"]).stat().st_size for p in photos) / 1e6
    counts = Counter((p["split"], p["label"]) for p in photos)
    print(f"\n완료 ({time.time() - started:.0f}초) → {out}")
    print(f"- 원본 사진 {len(photos)}장, {size_mb:.0f}MB (이번에 새로 저장 {written}장, 모두 열림 확인)")
    for split in args.splits:
        print(f"  {split:16s} 진짜 {counts[(split, 0)]:5d}  AI {counts[(split, 1)]:5d}")
    if args.normalize:
        print(f"- 형식·크기를 맞춘 사진 {len(photos) - too_small}장 ({CROP}x{CROP} JPEG), "
              f"{CROP}px보다 작아 뺀 사진 {too_small}장 (normalized_path가 빈칸)")
    print(f"- 통합 시험 세트 combined_test.csv: {len(combined)}문항")
    print("- 정답: label 0 = 진짜, 1 = AI / is_mismatch 0 = 글 맞음, 1 = 글 안 맞음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
