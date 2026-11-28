#!/usr/bin/env python3
"""편집 강도 실험: 사진에 편집을 한 가지씩 점점 세게 하면서 AI 사진 판정(UnivFD 방식)이 어떻게 바뀌는지 본다.

카드 편집 실험에서는 카드마다 압축·크기 바꾸기·글 덮기가 한꺼번에 들어가서 무엇이 판정을 망가뜨리는지
따로 볼 수 없었다. 여기서는 원본 사진에 편집을 하나씩만 한다.
    jpeg    JPEG 품질 95 → 30으로 다시 저장 (카드는 85로 저장했다)
    shrink  가로·세로를 75% → 25%로 줄이기
    cover   사진 아래쪽을 어두운 띠 + 흰 글씨로 10% → 50% 가리기 (글은 그 사건의 사실 글)

판정은 univfd.py가 학습한 분류기를 그대로 쓰고, 기준(0.5)도 바꾸지 않는다.
시험 사진은 test1(학습 때 본 생성기)과 SDXL(test4·5) 1,500장이다. DALL-E 3(test2·3)는 원본부터
찍기 수준(53~56%)이라 편집의 영향을 볼 수 없어서 뺐다.

실행 (저장소 맨 위 폴더에서, univfd.py를 먼저 실행한 뒤):
    python results/장성준/edit_strength.py
결과 (data/ 폴더는 깃에 올라가지 않는다):
    data/edit_strength/features/          편집한 사진의 특징 (중간에 멈춰도 다시 실행하면 이어서 계산)
    data/edit_strength/scores.csv         사진마다 AI일 확률
    data/edit_strength/summary.csv        편집·단계·묶음마다 정확도, AUC, 진짜를 진짜로, AI를 AI로
    data/edit_strength/edit_strength.png  그래프 (test1)
"""
from __future__ import annotations

import argparse
import csv
import io
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "data_preparation" / "cards"))
import final_verdict  # noqa: E402  (univfd.py가 저장해 둔 원본 특징을 지우지 않고 읽기)
import make_cards as mc  # noqa: E402  (카드와 같은 글꼴·줄바꿈)
import match_clip  # noqa: E402  (새로 계산할 사진이 있을 때만 CLIP 불러오기)
import score_cards  # noqa: E402
import univfd  # noqa: E402

SPLITS = ["test1_nyt_mj", "test4_bbc_sdxl", "test5_cnn_sdxl"]
GROUPS = {"test1": ["test1_nyt_mj"], "sdxl": ["test4_bbc_sdxl", "test5_cnn_sdxl"]}
GROUP_NAMES = {"test1": "test1 (NYT · Midjourney)", "sdxl": "SDXL (BBC · CNN)"}
LEVELS = {"jpeg": [95, 85, 75, 50, 30], "shrink": [0.75, 0.5, 0.35, 0.25], "cover": [0.1, 0.2, 0.3, 0.4, 0.5]}
EDIT_NAMES = {"jpeg": "JPEG 품질", "shrink": "사진 축소 (가로·세로)", "cover": "글로 가린 비율 (아래부터)"}
THRESHOLD = 0.5
TEXT_SIZE = 0.05   # 글자 크기: 사진 짧은 변의 5% (카드에서 사진 위에 덮은 글과 비슷한 크기)
BAND_ALPHA = 215   # 띠 어둡기: 카드에서 사진 위에 글을 덮을 때 가장 어두운 곳과 같다
SUMMARY_FIELDS = ["edit", "level", "group", "n", "acc", "auc", "real_acc", "ai_acc"]
KOREAN_FONTS = ["Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "NanumGothic"]


# ---------------------------------------------------------------------------
# 편집
# ---------------------------------------------------------------------------
def default_font() -> str:
    return mc.default_fonts()[0]  # 카드 제목과 같은 DejaVu Sans Bold


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    """JPEG 품질 quality로 다시 저장한 사진 (카드와 같은 4:2:0 색 압축)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=2)
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB")


def shrink(img: Image.Image, ratio: float) -> Image.Image:
    """가로·세로를 ratio배로 줄인 사진 (카드처럼 LANCZOS). 1픽셀보다 작아지지는 않는다."""
    w, h = img.size
    return img.convert("RGB").resize((max(1, round(w * ratio)), max(1, round(h * ratio))), Image.LANCZOS)


def cover(img: Image.Image, ratio: float, text: str, font_path: str) -> Image.Image:
    """사진 아래쪽 ratio만큼을 어두운 띠로 덮고, 띠 안에 흰 글씨로 text를 들어가는 만큼 쓴다."""
    img = img.convert("RGB")
    w, h = img.size
    band = round(h * ratio)
    if band <= 0:
        return img
    top = h - band
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([0, top, w, h], fill=(0, 0, 0, BAND_ALPHA))
    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    size = max(8, round(min(w, h) * TEXT_SIZE))
    pad, step = round(size * 0.25), int(size * 1.25)
    max_lines = (band - 2 * pad) // step
    if text.strip() and max_lines > 0:
        fnt, margin = mc.font(font_path, size), round(w * 0.05)
        lines = mc.wrap(text, fnt, w - 2 * margin, max_lines)
        mc.draw_lines(ImageDraw.Draw(out), lines, fnt, (margin, top + pad), fill=(255, 255, 255))
    return out


def apply_edit(img: Image.Image, edit: str, level, text: str, font_path: str) -> Image.Image:
    if edit == "jpeg":
        return jpeg(img, int(level))
    if edit == "shrink":
        return shrink(img, float(level))
    if edit == "cover":
        return cover(img, float(level), text, font_path)
    raise ValueError(f"모르는 편집: {edit}")


# ---------------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------------
def summarize_groups(rows) -> dict:
    """rows: (split, 정답 1=AI, 점수) → 묶음마다 n, 정확도, AUC, 진짜를 진짜로, AI를 AI로 (기준 0.5)."""
    out = {}
    for group, splits in GROUPS.items():
        items = [(None, float(score), {"is_ai": int(label)}) for split, label, score in rows if split in splits]
        if items:
            out[group] = score_cards.summarize(items, THRESHOLD)
    return out


def load_photos(data: Path, limit: int, seed: int) -> dict:
    """시험 사진 (split → 행 목록). 글로 가리기에 쓸 사실 글을 교차 시험 세트에서 붙인다."""
    photos_csv, cross_csv = data / "ai_photos" / "ai_photos.csv", data / "ai_photos" / "cross_test.csv"
    if not photos_csv.exists() or not cross_csv.exists():
        raise SystemExit("사진이 없어요. 먼저 python data_preparation/prepare_all.py 를 실행해 주세요.")
    with open(cross_csv, newline="", encoding="utf-8") as f:
        true_text = {r["image_path"]: r["text"] for r in csv.DictReader(f) if r["text_fake"] == "0"}
    rows = defaultdict(list)
    with open(photos_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] in SPLITS:
                if r["image_path"] not in true_text:
                    raise SystemExit(f"{r['image_id']}의 사실 글이 교차 시험 세트에 없어요. prepare_all.py를 다시 실행해 주세요.")
                rows[r["split"]].append({**r, "text": true_text[r["image_path"]]})
    return {s: univfd.balanced(rows[s], limit, seed) for s in SPLITS}


def level_label(edit: str, level: str) -> str:
    if edit == "jpeg":
        return level
    if level == "원본":
        return "100%" if edit == "shrink" else "0%"
    return f"{float(level) * 100:g}%"


# ---------------------------------------------------------------------------
# 그래프
# ---------------------------------------------------------------------------
def plot(summary: list[dict], path: Path, group: str = "test1") -> None:
    """편집마다 한 칸: 정확도, AI를 AI로, 진짜를 진짜로 (같은 0~100% 눈금)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in KOREAN_FONTS if f in names), None)
    if family:
        plt.rcParams["font.family"] = family
    ink, ink2, grid, axis = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
    series = [("acc", "정확도", "#257de0"), ("ai_acc", "AI를 AI로", "#f47230"), ("real_acc", "진짜를 진짜로", "#00b894")]
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(11, 4.1), sharey=True)
    for ax, edit in zip(axes, LEVELS):
        rows = [r for r in summary if r["edit"] == edit and r["group"] == group]
        x = list(range(len(rows)))
        ax.axhline(50, color=axis, lw=1, zorder=1)
        for key, name, color in series:
            ax.plot(x, [float(r[key]) * 100 for r in rows], color=color, lw=2, marker="o", ms=7, mfc=color,
                    mec="white", mew=2, label=name, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ticks = [level_label(edit, r["level"]) + ("\n(카드)" if (edit, r["level"]) == ("jpeg", "85") else "")
                 for r in rows]
        ax.set_xticks(x, ticks)
        ax.set_title(EDIT_NAMES[edit], loc="left", fontsize=12, color=ink, pad=10)
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
        ax.grid(axis="y", color=grid, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(axis)
        ax.tick_params(colors=ink2, length=0, labelsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=10, labelcolor=ink2,
               bbox_to_anchor=(0.99, 0.99))
    title = GROUP_NAMES[group].replace(" · ", ", ")  # 한글 글꼴에서는 가운뎃점이 넓게 보여서
    fig.text(0.01, 0.955, f"{title} 사진에 편집을 한 가지씩 세게 했을 때", fontsize=13, color=ink,
             ha="left", va="center")
    fig.text(0.01, 0.885, "왼쪽 끝이 원본, 회색 가로선이 찍기 수준(50%)", fontsize=10, color=ink2, ha="left", va="center")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None, model=None) -> int:
    parser = argparse.ArgumentParser(description="편집 강도 실험 (AI 사진 판정, UnivFD 방식)")
    parser.add_argument("--data", default=str(REPO_ROOT / "data"), help="prepare_all.py로 받은 data 폴더")
    parser.add_argument("--out", default=None, help="결과 폴더 (기본: data/edit_strength)")
    parser.add_argument("--backend", default="auto", choices=["auto", "openvino", "onnxruntime"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="빠른 시험용: split마다 이만큼만 쓰기 (0 = 전부)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    data = Path(args.data)
    out = Path(args.out) if args.out else data / "edit_strength"
    head_path = data / "univfd" / "head.pkl"
    if not head_path.exists():
        raise SystemExit("분류기가 없어요. 먼저 python results/장성준/univfd.py 를 실행해 주세요.")
    head = pickle.loads(head_path.read_bytes())
    photos = load_photos(data, args.limit, args.seed)
    model = model or match_clip.LazyImageModel(args.backend)
    font_path, feat_dir, started = default_font(), out / "features", time.time()

    scored = []  # (편집, 단계, split, image_id, 정답, 점수)
    for split in SPLITS:
        try:
            cached = final_verdict.load_cached(data / "univfd" / "features", f"photo-{split}")
        except SystemExit:
            raise SystemExit(f"photo-{split} 특징이 없어요. 먼저 python results/장성준/univfd.py 를 실행해 주세요.") from None
        rows = photos[split]
        prob = head.predict_proba(univfd.l2(np.stack([cached[r["image_id"]] for r in rows])))[:, 1]
        scored += [("none", "원본", split, r["image_id"], int(r["label"]), p) for r, p in zip(rows, prob)]
    for edit, levels in LEVELS.items():
        for level in levels:
            for split in SPLITS:
                rows = photos[split]
                feats = univfd.embed(
                    f"{edit}-{level}-{split}", [r["image_id"] for r in rows],
                    lambda k, rows=rows, edit=edit, level=level: apply_edit(
                        univfd.open_rgb(data / "ai_photos" / rows[k]["image_path"]), edit, level, rows[k]["text"],
                        font_path),
                    model, feat_dir, args.batch)
                prob = head.predict_proba(univfd.l2(feats))[:, 1]
                scored += [(edit, str(level), split, r["image_id"], int(r["label"]), p) for r, p in zip(rows, prob)]

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["edit", "level", "split", "id", "score"])
        writer.writerows([e, lv, s, i, f"{p:.6f}"] for e, lv, s, i, _, p in scored)
    summary = []
    for edit, levels in LEVELS.items():
        for level in ["원본", *map(str, levels)]:
            source = "none" if level == "원본" else edit
            picked = [(s, y, p) for e, lv, s, _, y, p in scored if e == source and lv == level]
            for group, m in summarize_groups(picked).items():
                summary.append({"edit": edit, "level": level, "group": group, "n": m["n"],
                                **{k: f"{m[k]:.6f}" for k in ("acc", "auc", "real_acc", "ai_acc")}})
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    plot(summary, out / "edit_strength.png")

    for group in GROUPS:
        print(f"\n{GROUP_NAMES[group]}: 정확도 (AUC) / 진짜를 진짜로 / AI를 AI로")
        for r in (r for r in summary if r["group"] == group):
            print(f"  {EDIT_NAMES[r['edit']]:<16} {level_label(r['edit'], r['level']):>5}: "
                  f"{float(r['acc']):.1%} ({float(r['auc']):.3f}) / {float(r['real_acc']):.1%} / {float(r['ai_acc']):.1%}")
    print(f"\n완료 ({(time.time() - started) / 60:.1f}분) → {out / 'summary.csv'}, {out / 'edit_strength.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
