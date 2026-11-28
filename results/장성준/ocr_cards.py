#!/usr/bin/env python3
"""카드뉴스 이미지에서 글자를 읽어낸다 (OCR, Tesseract). 카드 편집 실험의 글 쪽 준비.

실제 카드뉴스는 글도 그림 속 글자라서, 판정하려면 먼저 읽어내야 한다. 이때 생기는 손해 두 가지를 잰다.
- 잘림: 카드에 다 못 들어가 "…"로 잘린 글
- 읽기 실수: OCR이 글자를 잘못 읽은 것

읽는 곳: 템플릿마다 글이 있는 자리만 잘라서 읽는다 (브랜드 이름·쪽 번호·안내 문구는 뺀다).
- 사진 위에 글을 덮은 템플릿(photo_full): 거의 흰 글자만 남기고 읽은 뒤, 자신 없는 줄(평균 확신도 70 미만)은 버린다.
- 글 많은 템플릿(text_heavy): 강조 막대 아래 본문만 읽는다 (위의 제목은 본문 앞부분과 같아서).

실행 (저장소 맨 위 폴더에서, make_cards.py로 카드를 만든 뒤. Tesseract가 설치돼 있어야 한다):
    pip install pillow numpy rapidfuzz matplotlib
    python results/장성준/ocr_cards.py
결과: data/cards/ocr.csv (카드마다 읽어낸 글, 잘렸는지, 읽기 오류율)
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "data_preparation" / "cards"))
import make_cards as mc  # noqa: E402  (카드 크기·템플릿·글꼴·줄바꿈을 같이 쓴다)

M, W, H = mc.MARGIN, mc.W, mc.H
# 템플릿마다 글이 있는 자리 (x1, y1, x2, y2)
REGIONS = {
    "photo_full": (M - 20, int(H * 0.45), W - M + 20, H - 90),
    "photo_top": (M - 20, 868, W - M + 20, 1250),
    "title_top": (M - 20, 130, W - M + 20, 465),
    "text_heavy": (M - 20, 130, W - M + 20, 855),
}
# 카드에 그린 글 (글꼴 종류, 크기, 최대 줄 수) — 카드에 실제로 보이는 글을 계산할 때 쓴다
TEXT_STYLE = {"photo_full": ("bold", 58, 5), "photo_top": ("bold", 54, 5),
              "title_top": ("bold", 52, 4), "text_heavy": ("regular", 36, 8)}
FIELDS = ["card_id", "template", "truncated", "ocr_text", "visible_text", "cer_visible", "cer_original",
          "same_as_original"]


def norm(text: str) -> str:
    """비교용: 둥근 따옴표·말줄임표·대시를 같은 글자로 맞추고 띄어쓰기를 정리한다."""
    for a, b in (("…", "..."), ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("—", "-"), ("–", "-")):
        text = text.replace(a, b)
    return " ".join(text.split())


def visible_text(text: str, template: str, fonts: tuple[str, str]) -> str:
    """카드에 실제로 그려진 글 (넘치면 "…"로 잘린 글)."""
    kind, size, max_lines = TEXT_STYLE[template]
    fnt = mc.font(fonts[0] if kind == "bold" else fonts[1], size)
    return " ".join(mc.wrap(" ".join(text.split()), fnt, W - 2 * M, max_lines))


def bar_top(card: Image.Image) -> int | None:
    """text_heavy 카드의 강조 막대(왼쪽 여백의 한 색 가로 막대) 위치."""
    a = np.asarray(card.convert("RGB"), dtype=int)
    for y in range(150, 520):
        seg = a[y, M + 5:M + 85]
        if seg.std(axis=0).max() < 4 and np.abs(seg.mean(axis=0) - a[y, W - 20]).sum() > 60:
            return y
    return None


def tesseract_lines(img: Image.Image) -> list[tuple[float, str]]:
    """Tesseract로 읽고 줄마다 (평균 확신도, 글)을 돌려준다."""
    buf = io.BytesIO()
    img.save(buf, "PNG")
    out = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6", "-l", "eng", "tsv"],
                         input=buf.getvalue(), capture_output=True, check=True).stdout.decode("utf-8")
    lines = defaultdict(list)
    for row in out.splitlines()[1:]:
        r = row.split("\t")
        if len(r) == 12 and r[11].strip():
            lines[(int(r[2]), int(r[3]), int(r[4]))].append((float(r[10]), r[11]))
    return [(float(np.mean([c for c, _ in ws])), " ".join(w for _, w in ws)) for _, ws in sorted(lines.items())]


def read_card(card: Image.Image, template: str) -> str:
    x1, y1, x2, y2 = REGIONS[template]
    if template == "text_heavy":
        bar = bar_top(card)
        if bar is not None:
            y1 = bar + 30
    region = ImageOps.grayscale(card.convert("RGB").crop((x1, y1, x2, y2)))
    if template == "photo_full":
        a = np.asarray(region)
        img = Image.fromarray(np.where(a >= 230, 0, 255).astype(np.uint8))  # 거의 흰 글자만 남긴다
        return " ".join(t for conf, t in tesseract_lines(img) if conf >= 70)
    if np.asarray(region).mean() < 128:  # 어두운 바탕의 밝은 글자는 뒤집는다
        region = ImageOps.invert(region)
    return " ".join(t for _, t in tesseract_lines(region))


def cer(got: str, want: str) -> float:
    from rapidfuzz.distance import Levenshtein
    want = norm(want)
    return Levenshtein.distance(norm(got), want) / max(1, len(want))


def process(job: dict) -> dict:
    card_path, template, text, fonts = job["card_abs"], job["template"], job["text"], job["fonts"]
    with Image.open(card_path) as card:
        got = read_card(card, template)
    vis = visible_text(text, template, fonts)
    return {"card_id": job["card_id"], "template": template, "truncated": int(vis.endswith("…")),
            "ocr_text": got, "visible_text": vis, "cer_visible": round(cer(got, vis), 4),
            "cer_original": round(cer(got, text), 4), "same_as_original": int(norm(got) == norm(text))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="카드뉴스 이미지에서 글자 읽기 (Tesseract)")
    parser.add_argument("--cards", default=str(REPO_ROOT / "data" / "cards"), help="make_cards.py 결과 폴더")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0, help="빠른 시험용: 이만큼만")
    args = parser.parse_args(argv)
    if shutil.which("tesseract") is None:
        raise SystemExit("Tesseract가 없어요. https://github.com/tesseract-ocr/tesseract 에서 설치해 주세요.")
    folder = Path(args.cards)
    if not (folder / "cards.csv").exists():
        raise SystemExit("카드가 없어요. 먼저 python data_preparation/cards/make_cards.py 를 실행해 주세요.")
    with open(folder / "cards.csv", newline="", encoding="utf-8") as f:
        cards = list(csv.DictReader(f))
    if args.limit:
        cards = cards[: args.limit]
    fonts = mc.default_fonts()
    jobs = [{"card_id": c["card_id"], "template": c["template"], "text": c["text"], "fonts": fonts,
             "card_abs": str(folder / c["card_path"])} for c in cards]
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # Tesseract 하나가 코어 하나만 쓰게
    started = time.time()
    print(f"카드 {len(jobs)}장 읽는 중...", flush=True)
    with Pool(args.workers) as pool:
        rows = []
        for k, row in enumerate(pool.imap(process, jobs, chunksize=8), start=1):
            rows.append(row)
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}장 ({(time.time() - started) / 60:.1f}분)", flush=True)
    with open(folder / "ocr.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"완료 ({(time.time() - started) / 60:.1f}분) → {folder / 'ocr.csv'}")
    print(summary(rows))
    return 0


def summary(rows: list[dict]) -> str:
    lines = ["| 템플릿 | 카드 | 글이 잘린 카드 | 보이는 글을 그대로 읽은 카드 | 읽기 오류율 (글자 기준) |",
             "|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in rows:
        groups[r["template"]].append(r)
    for name in list(sorted(groups)) + ["전체"]:
        g = rows if name == "전체" else groups[name]
        bold = "**" if name == "전체" else ""
        exact = sum(1 for r in g if float(r["cer_visible"]) == 0)
        lines.append(f"| {bold}{name}{bold} | {len(g):,} | {sum(int(r['truncated']) for r in g) / len(g) * 100:.1f}% | "
                     f"{exact / len(g) * 100:.1f}% | {np.mean([float(r['cer_visible']) for r in g]) * 100:.2f}% |")
    same = sum(int(r["same_as_original"]) for r in rows)
    lines.append("")
    lines.append(f"원래 글과 똑같이 읽힌 카드 {same:,}장 ({same / len(rows) * 100:.1f}%), 달라진 카드 {len(rows) - same:,}장")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
