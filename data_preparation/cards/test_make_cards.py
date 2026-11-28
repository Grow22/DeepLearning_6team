"""가짜 데이터로 make_cards.py를 시험한다. 실행: python -m pytest -q data_preparation"""
import csv
from collections import Counter

import numpy as np
import pytest
from PIL import Image

import make_cards as mc

SPLIT = "test1_nyt_mj"
N = 8  # 사건 수. 0~7행은 AI 사진(지어낸 글), 8~15행은 진짜 사진(기사 글), 같은 순번끼리 같은 사건


def photo(size, color):
    rng = np.random.default_rng(sum(color) + size[0])
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    arr[..., 0] = color[0]  # 사진마다 다르게 보이도록 빨강 채널을 고정
    return Image.fromarray(arr)


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def setup(tmp_path):
    """prepare_all.py로 받은 것처럼 사진·시험 세트를 만들고, 깃에 있는 것처럼 문항 표를 만든다."""
    texts = [f"Fabricated scandal number {i} shocks the city." for i in range(N)] + \
            [f"Officials met on Tuesday to discuss plan number {i}." for i in range(N)]
    texts[N + 3] = "Officials  met on Tuesday\nto discuss plan number 3."  # 띄어쓰기·줄바꿈이 섞인 원문
    labels = [1] * N + [0] * N

    photos = tmp_path / "ai_photos"
    (photos / "images" / SPLIT).mkdir(parents=True)
    sizes = [(600, 330), (1024, 1024), (300, 900), (1200, 200)]
    photo_rows = []
    for row in range(2 * N):
        rel = f"images/{SPLIT}/{row}.jpg"
        photo(sizes[row % len(sizes)], (row * 15 % 256, 0, 0)).save(photos / rel, quality=95)
        photo_rows.append({"image_id": f"{SPLIT}-{row:05d}", "split": SPLIT, "label": labels[row], "image_path": rel})
    write_csv(photos / "ai_photos.csv", ["image_id", "split", "label", "image_path"], photo_rows)

    # 교차 시험: 사건마다 진짜/AI 사진 x 기사 글/지어낸 글
    cross = []
    for e in range(N):
        for img, txt in ((N + e, N + e), (N + e, e), (e, N + e), (e, e)):
            cross.append({"item_id": f"cross-{SPLIT}-{len(cross):04d}", "split": SPLIT, "image_row": img,
                          "text_row": txt, "is_ai": labels[img], "text_fake": labels[txt]})
    # 통합 시험: 사진마다 한 번. 절반은 자기 글, 절반은 같은 종류의 다른 사진 글
    pairs = [(r, r) for r in (4, 5, 6, 7, 9, 11, 13, 15)]
    pairs += [(0, 1), (1, 2), (2, 3), (3, 0), (8, 10), (10, 12), (12, 14), (14, 8)]
    combined = [{"item_id": f"{SPLIT}-{k:04d}", "split": SPLIT, "image_row": img, "text_row": txt,
                 "is_ai": labels[img], "is_mismatch": int(img != txt)} for k, (img, txt) in enumerate(pairs)]

    tables = tmp_path / "tables"
    tables.mkdir()
    write_csv(tables / "cross_test_pairs.csv", list(cross[0]), cross)
    write_csv(tables / "combined_test_pairs.csv", list(combined[0]), combined)

    def local(rows, label_key):
        return [{"item_id": r["item_id"], "split": SPLIT, "image_path": f"images/{SPLIT}/{r['image_row']}.jpg",
                 "text": texts[r["text_row"]], "is_ai": r["is_ai"], label_key: r[label_key], "normalized_path": ""}
                for r in rows]
    write_csv(photos / "cross_test.csv", ["item_id", "split", "image_path", "text", "is_ai", "text_fake",
                                          "normalized_path"], local(cross, "text_fake"))
    write_csv(photos / "combined_test.csv", ["item_id", "split", "image_path", "text", "is_ai", "is_mismatch",
                                             "normalized_path"], local(combined, "is_mismatch"))
    return {"texts": texts, "labels": labels, "photos": photos, "cross": cross, "combined": combined,
            "cross_table": tables / "cross_test_pairs.csv", "combined_table": tables / "combined_test_pairs.csv"}


def items_of(setup):
    return mc.load_items([SPLIT], setup["photos"], setup["cross_table"], setup["combined_table"])[SPLIT]


def run(setup, out, *extra):
    return mc.main(["--splits", SPLIT, "--photos", str(setup["photos"]), "--out", str(out),
                    "--cross", str(setup["cross_table"]), "--combined", str(setup["combined_table"]),
                    "--workers", "1", *extra])


def read(out):
    with open(out / "cards.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_wrap_fits_width_and_lines():
    fnt = mc.font(mc.default_fonts()[0], 40)
    text = "word " * 60 + "Supercalifragilisticexpialidocious" * 4
    lines = mc.wrap(text, fnt, 400, max_lines=3)
    assert len(lines) == 3 and lines[-1].endswith("…")
    assert all(fnt.getlength(line) <= 400 for line in lines)
    assert mc.wrap("short text", fnt, 400, max_lines=3) == ["short text"]


@pytest.mark.parametrize("template", list(mc.TEMPLATES))
@pytest.mark.parametrize("size", [(10, 10), (3000, 100), (100, 3000)])
def test_render_every_template_and_shape(template, size):
    card, scale = mc.render_card(photo(size, (200, 0, 0)), "A long caption " * 20, template, "beige",
                                 mc.default_fonts())
    assert card.size == (mc.W, mc.H) and card.mode == "RGB" and scale > 0
    x1, y1, x2, y2 = mc.TEMPLATES[template]
    region = np.asarray(card.crop((x1, y1, x2, min(y2, y1 + 200))))
    assert np.abs(region[..., 0].astype(int) - 200).mean() < 30  # 사진 칸에 사진이 들어감


def test_every_item_becomes_exactly_one_card(setup):
    plan = mc.plan_split(SPLIT, setup["labels"], items_of(setup), seed=42)
    assert len(plan) == 4 * N + N  # 교차 시험 32문항 + 통합 시험의 안 맞는 글 8문항
    assert sorted(p["cross_item_id"] for p in plan if p["cross_item_id"]) == [r["item_id"] for r in setup["cross"]]
    assert sorted(p["combined_item_id"] for p in plan if p["combined_item_id"]) == \
        [r["item_id"] for r in setup["combined"]]
    for p in plan:
        img, txt = p["image_row"], p["text_row"]
        assert p["text"] == setup["texts"][txt]  # 원문 그대로
        assert (p["is_ai"], p["text_fake"], p["is_mismatch"]) == \
            (setup["labels"][img], setup["labels"][txt], int(img % N != txt % N))
        if p["combined_item_id"] and not p["is_mismatch"]:
            assert p["cross_item_id"]  # 글이 맞는 통합 문항은 교차 문항과 같은 카드


def test_same_event_same_style_and_real_ai_balanced(setup):
    plan = mc.plan_split(SPLIT, setup["labels"], items_of(setup), seed=42)
    style = {}
    for p in plan:
        assert style.setdefault(p["event"], (p["template"], p["palette"])) == (p["template"], p["palette"])
    for field in ("template", "palette"):
        real = Counter(p[field] for p in plan if p["is_ai"] == 0)
        ai = Counter(p[field] for p in plan if p["is_ai"] == 1)
        assert real == ai  # 진짜·AI 카드의 템플릿·색 수가 똑같음
    templates = Counter(t for t, _ in style.values())
    assert len(templates) == len(mc.TEMPLATES) and max(templates.values()) - min(templates.values()) <= 2
    assert mc.plan_split(SPLIT, setup["labels"], items_of(setup), seed=42) == plan
    assert mc.plan_split(SPLIT, setup["labels"], items_of(setup), seed=7) != plan


def test_wrong_answers_or_structure_stop(setup):
    with pytest.raises(SystemExit):
        mc.plan_split("x", [0, 1], [], seed=1)
    items = items_of(setup)
    items[0]["is_ai"] = 1 - items[0]["is_ai"]
    with pytest.raises(SystemExit):
        mc.plan_split(SPLIT, setup["labels"], items, seed=1)


def test_cards_hold_their_own_photo_and_text(setup, tmp_path):
    out = tmp_path / "cards"
    assert run(setup, out, "--save-crops") == 0
    rows = read(out)
    assert len(rows) == 5 * N and list(rows[0]) == mc.FIELDS
    with open(setup["photos"] / "cross_test.csv", newline="", encoding="utf-8") as f:
        local_text = {r["item_id"]: r["text"] for r in csv.DictReader(f)}
    for r in rows:
        if r["cross_item_id"]:
            assert r["text"] == local_text[r["cross_item_id"]]  # 받은 시험 세트의 글과 똑같음
        box = (int(r["box_x1"]), int(r["box_y1"]), int(r["box_x2"]), int(r["box_y2"]))
        with Image.open(out / r["card_path"]) as card:
            assert (card.format, card.size) == ("JPEG", (mc.W, mc.H))
        crop = mc.crop_from_card(out / r["card_path"], box)
        with Image.open(out / r["crop_path"]) as saved:
            assert np.array_equal(np.asarray(saved), np.asarray(crop))  # 저장한 것과 바로 자른 것이 같음
        canvas = Image.new("RGB", (mc.W, mc.H))
        with Image.open(setup["photos"] / r["photo_path"]) as p:
            mc.place_photo(canvas, p.convert("RGB"), box)
        h = (box[3] - box[1]) if r["text_over_photo"] == "0" else int((box[3] - box[1]) * 0.4)
        diff = np.abs(np.asarray(crop, dtype=float)[:h] - np.asarray(canvas.crop(box), dtype=float)[:h]).mean()
        assert diff < 12  # JPEG 압축 차이 정도만 남음


def test_rerun_gives_same_cards(setup, tmp_path):
    run(setup, tmp_path / "a")
    run(setup, tmp_path / "b")
    a, b = read(tmp_path / "a"), read(tmp_path / "b")
    assert a == b
    for r in a:
        assert (tmp_path / "a" / r["card_path"]).read_bytes() == (tmp_path / "b" / r["card_path"]).read_bytes()
    assert all(r["crop_path"] == "" for r in a)  # --save-crops 없이는 잘라낸 사진을 저장하지 않음


def test_missing_or_old_data_stops(setup, tmp_path):
    with open(setup["photos"] / "combined_test.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["image_path"] = rows[1]["image_path"]  # 깃의 표와 다른 옛날 시험 세트
    write_csv(setup["photos"] / "combined_test.csv", list(rows[0]), rows)
    with pytest.raises(SystemExit):
        run(setup, tmp_path / "out")
    (setup["photos"] / "cross_test.csv").unlink()  # 교차 시험이 생기기 전에 받은 경우
    with pytest.raises(SystemExit):
        run(setup, tmp_path / "out")
    (setup["photos"] / "ai_photos.csv").unlink()
    with pytest.raises(SystemExit):
        run(setup, tmp_path / "out")
