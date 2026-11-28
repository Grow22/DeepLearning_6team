"""ocr_cards.py 시험: 글을 아는 카드를 직접 그려서 읽어 본다. 실행: python -m pytest -q results"""
import shutil

import numpy as np
import pytest
from PIL import Image

import ocr_cards as oc

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract가 없음")
TEXT = ("Officials met on Tuesday in Brussels to discuss a new plan for 12 regional bus routes, "
        "according to a statement released after the meeting.")


def photo(seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(30, 40, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((800, 600), Image.BILINEAR)  # 사진처럼 부드러운 무늬


def test_norm_and_cer():
    assert oc.norm("It’s  a “test” — ok…") == "It's a \"test\" - ok..."
    assert oc.cer("abcd", "abcd") == 0 and oc.cer("abxd", "abcd") == 0.25


@pytest.mark.parametrize("template,palette", [("photo_top", "navy"), ("title_top", "beige"),
                                              ("text_heavy", "white"), ("photo_full", "teal")])
def test_reads_what_is_on_the_card(tmp_path, template, palette):
    fonts = oc.mc.default_fonts()
    card, _ = oc.mc.render_card(photo(3), TEXT, template, palette, fonts)
    card.save(tmp_path / "card.jpg", quality=85)  # 실제 카드처럼 JPEG로 저장한 뒤 읽는다
    with Image.open(tmp_path / "card.jpg") as saved:
        got = oc.read_card(saved, template)
    want = oc.visible_text(TEXT, template, fonts)
    assert oc.cer(got, want) < 0.02, (got, want)
    assert "NEWS CARD" not in got and "1/5" not in got  # 브랜드·쪽 번호는 빼고 읽음


def test_long_text_is_marked_truncated():
    fonts = oc.mc.default_fonts()
    long_text = " ".join([TEXT] * 4)
    assert oc.visible_text(long_text, "title_top", fonts).endswith("…")
    assert not oc.visible_text("Short caption.", "title_top", fonts).endswith("…")
