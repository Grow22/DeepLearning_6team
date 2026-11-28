#!/usr/bin/env python3
"""섞어서 만든 시험 문항(교차 시험 + 통합 시험)을 카드뉴스 이미지로 만든다 (카드 편집 실험용).

카드는 원래 문항과 사진·글·정답이 똑같고, 카드로 편집됐다는 것만 다르다.
- 교차 시험 5,000문항은 모두 카드가 된다.
- 통합 시험 2,500문항 중 글이 맞는 1,250문항은 교차 시험에 같은 문항(같은 사진 + 같은 글)이 있어서
  그 카드를 같이 쓰고, 글이 안 맞는 1,250문항만 새로 만든다. 그래서 카드는 모두 6,250장이다.
- 같은 사건의 사진이 들어간 카드는 모두 같은 템플릿·색을 쓴다. 그래서 같은 글의 진짜·AI 카드는 사진만 다르고,
  같은 사진의 카드끼리는 글만 다르다. 진짜 카드와 AI 카드의 템플릿·색 수도 split마다 똑같이 맞춘다.
  (꾸밈새나 카드 속 글이 진짜·AI와 엮이면, 글자를 읽는 모델이 사진이 아닌 다른 것을 보고 맞힐 수 있어서다.)

실행 (저장소 맨 위 폴더에서, prepare_all.py로 데이터를 먼저 받은 뒤):
    pip install pillow matplotlib
    python data_preparation/cards/make_cards.py

결과 (data/ 폴더는 깃에 올라가지 않는다):
    data/cards/cards/<split>/<사진 행>-<글 행>.jpg   카드 (1080x1350, JPEG 품질 85)
    data/cards/crops/<split>/<사진 행>-<글 행>.png   (--save-crops) 카드에서 사진 부분만 잘라낸 것
    data/cards/cards.csv                            카드마다 원래 문항 번호, 정답, 템플릿, 사진 위치

잘라낸 사진은 용량이 커서(장당 약 1MB) 기본으로는 저장하지 않는다.
cards.csv의 사진 위치(box_x1~box_y2)로 카드에서 바로 잘라 쓰면 같은 결과가 나온다 (crop_from_card).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TABLES = HERE.parent / "ai_photos"  # 깃에 있는 시험 세트 표 (행 번호와 정답)

TEST_SPLITS = ["test1_nyt_mj", "test2_bbc_dalle", "test3_cnn_dalle", "test4_bbc_sdxl", "test5_cnn_sdxl"]
W, H = 1080, 1350          # 카드 크기 (인스타그램 세로형 4:5)
MARGIN = 70
JPEG_QUALITY = 85          # SNS에 올릴 때 흔히 쓰는 품질
BRAND = "NEWS CARD"        # 가상의 이름 (실제 언론사·기관을 흉내 내지 않음)
FIELDS = ["card_id", "split", "event", "image_id", "cross_item_id", "combined_item_id",
          "is_ai", "is_mismatch", "text_fake", "template", "palette",
          "card_path", "crop_path", "photo_path", "box_x1", "box_y1", "box_x2", "box_y2",
          "photo_scale", "text_over_photo", "text"]

# 템플릿: 사진이 들어갈 자리 (x1, y1, x2, y2)
TEMPLATES = {
    "photo_top": (0, 0, W, 780),                          # 사진 위, 글 아래
    "photo_full": (0, 0, W, H),                           # 사진을 꽉 채우고 아래에 글을 덮음
    "title_top": (MARGIN, 470, W - MARGIN, 1130),         # 제목 위, 사진 가운데
    "text_heavy": (MARGIN, 860, W - MARGIN, 1280),        # 글이 많고 사진은 아래에 작게
}
PALETTES = {  # 배경색, 글자색, 강조색
    "navy": ((31, 42, 68), (255, 255, 255), (255, 196, 0)),
    "red": ((139, 30, 45), (255, 255, 255), (255, 214, 196)),
    "teal": ((15, 92, 92), (255, 255, 255), (178, 238, 218)),
    "beige": ((243, 237, 226), (30, 30, 30), (178, 58, 40)),
    "white": ((250, 250, 250), (25, 25, 25), (20, 90, 200)),
}


# ---------------------------------------------------------------------------
# 글꼴과 글자 배치
# ---------------------------------------------------------------------------
def default_fonts() -> tuple[str, str]:
    """matplotlib에 들어 있는 DejaVu Sans를 쓴다. 누구 컴퓨터에서나 같은 글꼴이 되게 하려고."""
    try:
        import matplotlib
    except ImportError as e:
        raise SystemExit("글꼴을 찾으려면 matplotlib이 필요해요: pip install matplotlib") from e
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return str(base / "DejaVuSans-Bold.ttf"), str(base / "DejaVuSans.ttf")


_FONT_CACHE: dict = {}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def wrap(text: str, fnt: ImageFont.FreeTypeFont, width: int, max_lines: int) -> list[str]:
    """단어 단위로 줄을 나눈다. 줄이 넘치면 마지막 줄 끝을 '…'로 줄인다."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if fnt.getlength(trial) <= width:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = word
        while fnt.getlength(cur) > width and len(cur) > 1:  # 한 단어가 한 줄보다 긴 경우
            cut = len(cur) - 1
            while cut > 1 and fnt.getlength(cur[:cut]) > width:
                cut -= 1
            lines.append(cur[:cut])
            cur = cur[cut:]
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and fnt.getlength(last + "…") > width:
            last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
        lines[-1] = last.rstrip(" ,.;:") + "…"
    return lines


def draw_lines(draw: ImageDraw.ImageDraw, lines: list[str], fnt, xy: tuple[int, int],
               fill, spacing: float = 1.25, **kwargs) -> int:
    """여러 줄을 쓰고, 마지막 줄 아래 y 좌표를 돌려준다."""
    x, y = xy
    step = int(fnt.size * spacing)
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill, **kwargs)
        y += step
    return y


def short_headline(text: str, max_words: int = 8) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


# ---------------------------------------------------------------------------
# 카드 그리기
# ---------------------------------------------------------------------------
def place_photo(card: Image.Image, photo: Image.Image, box: tuple[int, int, int, int]) -> float:
    """사진을 칸에 꽉 차게(넘치는 부분은 가운데 기준으로 잘라) 넣고, 확대·축소 비율을 돌려준다."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    w, h = photo.size
    scale = max(bw / w, bh / h)
    nw, nh = max(bw, round(w * scale)), max(bh, round(h * scale))
    resized = photo.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - bw) // 2, (nh - bh) // 2
    card.paste(resized.crop((left, top, left + bw, top + bh)), (x1, y1))
    return scale


def bottom_gradient(card: Image.Image, start: int, max_alpha: int = 215) -> Image.Image:
    """사진 아래쪽을 점점 어둡게 덮어 글자가 잘 보이게 한다."""
    overlay = Image.new("RGBA", card.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(start, card.height):
        alpha = int(max_alpha * ((y - start) / (card.height - start)) ** 1.1)
        draw.line([(0, y), (card.width, y)], fill=(0, 0, 0, alpha))
    return Image.alpha_composite(card.convert("RGBA"), overlay).convert("RGB")


def render_card(photo: Image.Image, text: str, template: str, palette: str,
                fonts: tuple[str, str]) -> tuple[Image.Image, float]:
    bold, regular = fonts
    text = " ".join(text.split())  # 원문의 겹친 띄어쓰기·줄바꿈은 한 칸으로
    bg, fg, accent = PALETTES[palette]
    box = TEMPLATES[template]
    card = Image.new("RGB", (W, H), bg)
    scale = place_photo(card, photo.convert("RGB"), box)
    inner = W - 2 * MARGIN

    if template == "photo_full":
        card = bottom_gradient(card, start=int(H * 0.45))
        draw = ImageDraw.Draw(card)
        draw.text((MARGIN, 60), BRAND, font=font(bold, 30), fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((W - MARGIN, 60), "1/5", font=font(regular, 28), fill=(255, 255, 255),
                  anchor="ra", stroke_width=2, stroke_fill=(0, 0, 0))
        lines = wrap(text, font(bold, 58), inner, max_lines=5)
        top = H - 110 - len(lines) * int(58 * 1.25)
        draw_lines(draw, lines, font(bold, 58), (MARGIN, top), (255, 255, 255))
        return card, scale

    draw = ImageDraw.Draw(card)
    if template == "photo_top":
        draw.text((MARGIN, 820), BRAND, font=font(bold, 30), fill=accent)
        draw_lines(draw, wrap(text, font(bold, 54), inner, max_lines=5), font(bold, 54), (MARGIN, 875), fg)
        draw.text((W - MARGIN, H - 60), "1/5", font=font(regular, 28), fill=fg, anchor="rs")
    elif template == "title_top":
        draw.text((MARGIN, 60), BRAND, font=font(bold, 30), fill=accent)
        draw.text((W - MARGIN, 60), "1/5", font=font(regular, 28), fill=fg, anchor="ra")
        draw_lines(draw, wrap(text, font(bold, 52), inner, max_lines=4), font(bold, 52), (MARGIN, 140), fg)
        draw.rectangle([MARGIN, 1170, MARGIN + 90, 1178], fill=accent)
        draw.text((MARGIN, 1200), "Swipe for more", font=font(regular, 30), fill=fg)
    elif template == "text_heavy":
        draw.text((MARGIN, 60), BRAND, font=font(bold, 30), fill=accent)
        draw.text((W - MARGIN, 60), "1/5", font=font(regular, 28), fill=fg, anchor="ra")
        y = draw_lines(draw, wrap(short_headline(text), font(bold, 50), inner, max_lines=2),
                       font(bold, 50), (MARGIN, 140), fg)
        draw.rectangle([MARGIN, y + 20, MARGIN + 90, y + 28], fill=accent)
        draw_lines(draw, wrap(text, font(regular, 36), inner, max_lines=8), font(regular, 36),
                   (MARGIN, y + 60), fg, spacing=1.4)
    else:
        raise ValueError(f"모르는 템플릿이에요: {template}")
    return card, scale


def make_one(job: dict) -> dict:
    """카드 한 장을 만들어 저장하고, 표에 들어갈 한 줄을 돌려준다."""
    out = Path(job["out"])
    name = f"{job['split']}/{job['image_row']}-{job['text_row']}"
    with Image.open(job["photo_abs"]) as photo:
        card, scale = render_card(photo, job["text"], job["template"], job["palette"], job["fonts"])
    card_rel = f"cards/{name}.jpg"
    (out / card_rel).parent.mkdir(parents=True, exist_ok=True)
    card.save(out / card_rel, format="JPEG", quality=JPEG_QUALITY, subsampling=2, optimize=False)
    box = TEMPLATES[job["template"]]
    crop_rel = ""
    if job["save_crops"]:
        crop_rel = f"crops/{name}.png"
        (out / crop_rel).parent.mkdir(parents=True, exist_ok=True)
        crop_from_card(out / card_rel, box).save(out / crop_rel, format="PNG")
    return {
        "card_id": f"card-{job['split']}-{job['image_row']}-{job['text_row']}", "split": job["split"],
        "event": job["event"], "image_id": f"{job['split']}-{job['image_row']:05d}",
        "cross_item_id": job["cross_item_id"], "combined_item_id": job["combined_item_id"],
        "is_ai": job["is_ai"], "is_mismatch": job["is_mismatch"], "text_fake": job["text_fake"],
        "template": job["template"], "palette": job["palette"], "card_path": card_rel, "crop_path": crop_rel,
        "photo_path": job["photo_rel"], "box_x1": box[0], "box_y1": box[1], "box_x2": box[2], "box_y2": box[3],
        "photo_scale": round(scale, 4), "text_over_photo": int(job["template"] == "photo_full"),
        "text": job["text"],
    }


def crop_from_card(card_path, box) -> Image.Image:
    """저장된(압축된) 카드에서 사진 부분만 잘라낸다. box = (x1, y1, x2, y2)."""
    with Image.open(card_path) as card:
        return card.convert("RGB").crop(tuple(int(v) for v in box))


# ---------------------------------------------------------------------------
# 문항을 카드로 묶기
# ---------------------------------------------------------------------------
def plan_split(split: str, labels: list[int], items: list[dict], seed: int) -> list[dict]:
    """한 split의 시험 문항을 카드로 묶고 템플릿·색을 정한다.

    labels: 행마다 정답 (1 = AI, 0 = 진짜). 앞 절반이 AI, 뒤 절반이 진짜이고 같은 순번끼리 같은 사건이다.
    items: 문항 [{"test": "cross" 또는 "combined", "item_id", "image_row", "text_row", "text", "is_ai",
                  "text_fake"(교차 시험) 또는 "is_mismatch"(통합 시험)}]
    사진과 글이 같은 문항은 카드 한 장으로 합친다.
    """
    n = len(labels) // 2
    if n == 0 or len(labels) != 2 * n or labels[:n] != [1] * n or labels[n:] != [0] * n:
        raise SystemExit(f"{split}: 앞 절반이 AI, 뒤 절반이 진짜인 구조가 아니에요.")

    def event(row: int) -> int:
        return row if row < n else row - n

    cards = {}
    for it in items:
        img, txt = it["image_row"], it["text_row"]
        if not (0 <= img < 2 * n and 0 <= txt < 2 * n):
            raise SystemExit(f"{split} {it['item_id']}: 행 번호가 범위를 벗어났어요.")
        truth = {"is_ai": labels[img], "is_mismatch": int(event(img) != event(txt)), "text_fake": labels[txt]}
        for key, value in truth.items():
            if key in it and int(it[key]) != value:
                raise SystemExit(f"{split} {it['item_id']}: 표의 {key} 정답이 원본과 달라요.")
        if it["test"] == "cross" and truth["is_mismatch"]:
            raise SystemExit(f"{split} {it['item_id']}: 교차 시험인데 다른 사건의 글이 붙었어요.")
        card = cards.setdefault((img, txt), {
            "split": split, "event": event(img), "image_row": img, "text_row": txt, **truth,
            "cross_item_id": "", "combined_item_id": "", "text": it["text"]})
        if card["text"] != it["text"]:
            raise SystemExit(f"{split} {it['item_id']}: 같은 글 행인데 글이 달라요.")
        field = f"{it['test']}_item_id"
        if card[field]:
            raise SystemExit(f"{split} {it['item_id']}: 같은 사진·글 문항이 두 번 있어요.")
        card[field] = it["item_id"]

    # 템플릿·색은 사건(진짜·AI 사진 한 쌍)마다 정한다.
    # 안 맞는 글 카드가 어느 사진에 있는지(없음 / 둘 다 / 진짜만 / AI만)로 사건을 나누고, 묶음 안에서 돌아가며 배정한다.
    # '진짜만'과 'AI만' 묶음은 크기가 같으면(통합 시험이 그렇다) 같은 자리에서 시작해 템플릿·색 수가 똑같아진다.
    kind = {}
    for c in cards.values():
        real_mis, ai_mis = kind.get(c["event"], (0, 0))
        if c["is_mismatch"]:
            real_mis, ai_mis = (real_mis, 1) if c["is_ai"] else (1, ai_mis)
        kind[c["event"]] = (real_mis, ai_mis)
    rng = random.Random(f"{seed}-{split}")
    names = {"template": list(TEMPLATES), "palette": list(PALETTES)}
    offset, saved, style = {"template": 0, "palette": 0}, {}, {}
    for group in [(0, 0), (1, 1), (1, 0), (0, 1)]:
        if group == (1, 0):
            saved = dict(offset)
        elif group == (0, 1):
            offset = dict(saved)
        events = sorted(e for e, k in kind.items() if k == group)
        for field in ("template", "palette"):
            rng.shuffle(events)
            for k, e in enumerate(events):
                style.setdefault(e, {})[field] = names[field][(offset[field] + k) % len(names[field])]
            offset[field] = (offset[field] + len(events)) % len(names[field])

    order = sorted(cards.values(), key=lambda c: (c["event"], c["is_ai"], c["is_mismatch"], c["text_fake"],
                                                  c["text_row"]))
    return [{**c, **style[c["event"]]} for c in order]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_items(splits: list[str], photos_dir: Path, cross_table: Path, combined_table: Path) -> dict:
    """깃에 있는 표(행 번호·정답)와 prepare_all.py로 받은 시험 세트(글·사진 경로)를 문항 번호로 잇는다."""
    items = {split: [] for split in splits}
    for test, table, local_name in (("cross", cross_table, "cross_test.csv"),
                                    ("combined", combined_table, "combined_test.csv")):
        local_path = photos_dir / local_name
        if not local_path.exists():
            raise SystemExit(f"{local_name}이 없어요. python data_preparation/prepare_all.py 를 다시 실행해 주세요.")
        local = {r["item_id"]: r for r in read_csv(local_path)}
        for r in read_csv(table):
            if r["split"] not in items:
                continue
            got = local.get(r["item_id"])
            if (got is None or got["split"] != r["split"] or got["is_ai"] != r["is_ai"]
                    or Path(got["image_path"]).stem != r["image_row"]):
                raise SystemExit(f"{local_name}이 깃의 표와 달라요 ({r['item_id']}). "
                                 "python data_preparation/prepare_all.py 를 다시 실행해 주세요.")
            item = {"test": test, "item_id": r["item_id"], "image_row": int(r["image_row"]),
                    "text_row": int(r["text_row"]), "is_ai": int(r["is_ai"]), "text": got["text"]}
            for key in ("text_fake", "is_mismatch"):
                if key in r:
                    item[key] = int(r[key])
            items[r["split"]].append(item)
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="카드뉴스 이미지 만들기 (카드 편집 실험용)")
    parser.add_argument("--splits", nargs="+", default=TEST_SPLITS, choices=TEST_SPLITS)
    parser.add_argument("--photos", default=str(REPO_ROOT / "data" / "ai_photos"),
                        help="prepare_all.py로 받은 사진·시험 세트 폴더")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "cards"), help="저장 폴더")
    parser.add_argument("--cross", default=str(TABLES / "cross_test_pairs.csv"), help="교차 시험 세트 표")
    parser.add_argument("--combined", default=str(TABLES / "combined_test_pairs.csv"), help="통합 시험 세트 표")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--save-crops", action="store_true", help="잘라낸 사진도 PNG로 저장 (장당 약 1MB)")
    parser.add_argument("--font-bold", help="굵은 글꼴 파일 (기본: matplotlib의 DejaVu Sans Bold)")
    parser.add_argument("--font-regular", help="보통 글꼴 파일 (기본: matplotlib의 DejaVu Sans)")
    args = parser.parse_args(argv)

    started = time.time()
    photos_dir, out = Path(args.photos), Path(args.out)
    if not (photos_dir / "ai_photos.csv").exists():
        raise SystemExit("사진이 없어요. 먼저 python data_preparation/prepare_all.py 를 실행해 주세요.")
    photos = read_csv(photos_dir / "ai_photos.csv")
    photo_paths = {r["image_id"]: r["image_path"] for r in photos}
    labels = {split: {} for split in args.splits}
    for r in photos:
        if r["split"] in labels:
            labels[r["split"]][int(r["image_id"].rsplit("-", 1)[1])] = int(r["label"])
    for split, by_row in labels.items():
        if not by_row or sorted(by_row) != list(range(len(by_row))):
            raise SystemExit(f"{split} 사진이 없거나 빠졌어요. python data_preparation/prepare_all.py 를 실행해 주세요.")
        labels[split] = [by_row[row] for row in range(len(by_row))]
    items = load_items(args.splits, photos_dir, Path(args.cross), Path(args.combined))
    fonts = (args.font_bold or default_fonts()[0], args.font_regular or default_fonts()[1])

    jobs = []
    for split in args.splits:
        for p in plan_split(split, labels[split], items[split], args.seed):
            photo_rel = photo_paths[f"{split}-{p['image_row']:05d}"]
            jobs.append({**p, "photo_rel": photo_rel, "photo_abs": str(photos_dir / photo_rel), "out": str(out),
                         "fonts": fonts, "save_crops": args.save_crops})

    print(f"카드 {len(jobs)}장 만드는 중 (split {len(args.splits)}개)...", flush=True)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(make_one, jobs, chunksize=16))
    else:
        results = [make_one(j) for j in jobs]

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cards.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    n_cross = sum(1 for r in results if r["cross_item_id"])
    n_combined = sum(1 for r in results if r["combined_item_id"])
    n_both = sum(1 for r in results if r["cross_item_id"] and r["combined_item_id"])
    size_mb = sum((out / r["card_path"]).stat().st_size for r in results) / 1e6
    crops = f", 잘라낸 사진 {len(results)}장 (crops/)" if args.save_crops else ""
    print(f"완료 ({time.time() - started:.0f}초) → {out}")
    print(f"- 카드 {len(results)}장 {size_mb:.0f}MB (cards/){crops}, 표 cards.csv")
    print(f"- 교차 시험 {n_cross}문항 + 통합 시험 {n_combined}문항 (두 시험에 같이 있는 {n_both}문항은 카드 한 장)")
    print("- 정답: is_ai(사진이 AI인지), is_mismatch(글이 사진과 안 맞는지), text_fake(글이 거짓인지)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
