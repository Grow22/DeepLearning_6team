#!/usr/bin/env python3
"""세 판정을 합친 최종 판정: 카드뉴스 한 장이 허위인지 정한다.

판정 순서 (각 판정은 확률 0.5 이상이면 "그렇다"):
1. 글이 사실이 아니면             → 허위 (글이 거짓)         [factcheck.py]
2. 글은 사실인데 사진과 안 맞으면 → 허위 (글-사진 불일치)    [match_clip.py]
3. 내용은 괜찮은데 사진이 AI면    → 주의 (AI 사진)           [univfd.py]  AI 사진이라고 허위로 치지 않는다
4. 셋 다 아니면                   → 정상
정답도 같은 순서로 정한다 (text_fake → is_mismatch → is_ai).

같은 카드 6,250장을 두 가지로 판정해 비교한다.
- 원본: 원래 사진 + 원래 글
- 카드: 카드에서 잘라낸 사진 + 카드에서 읽어낸 글 (카드 안의 사진 위치는 안다고 가정)

실행 (univfd.py → ocr_cards.py → match_clip.py → factcheck.py, factcheck.py --set cards 를 먼저 실행한 뒤):
    python results/장성준/final_verdict.py
결과: data/final/final.csv (카드마다 정답·판정·세 확률), data/final/report.md
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import factcheck  # noqa: E402
import match_clip  # noqa: E402
import univfd  # noqa: E402

CATS = ["허위 (글이 거짓)", "허위 (글-사진 불일치)", "주의 (AI 사진)", "정상"]
FALSE_NEWS = {0, 1}  # 허위로 치는 판정
FIELDS = ["card_id", "template", "truth", "pred_original", "pred_card",
          "p_fake_original", "p_mismatch_original", "p_ai_original", "p_fake_card", "p_mismatch_card", "p_ai_card"]


def decide(fake: bool, mismatch: bool, ai: bool) -> int:
    """판정 순서대로 하나를 고른다 (CATS의 번호)."""
    return 0 if fake else 1 if mismatch else 2 if ai else 3


def load_cached(feat_dir: Path, name: str) -> dict:
    """univfd.py·match_clip.py가 저장해 둔 사진 특징을 그대로 읽는다 (새로 계산하거나 지우지 않는다)."""
    folder = feat_dir / name
    if not (folder / "ids.txt").exists():
        raise SystemExit(f"{name} 특징이 없어요. univfd.py와 match_clip.py를 먼저 실행해 주세요.")
    ids = (folder / "ids.txt").read_text(encoding="utf-8").split("\n")
    parts = sorted(folder.glob("*.npy"))
    feats = np.concatenate([np.load(p) for p in parts]) if parts else np.zeros((0, 768), np.float32)
    if len(feats) != len(ids):
        raise SystemExit(f"{name} 특징이 다 계산되지 않았어요. univfd.py·match_clip.py를 다시 실행해 주세요.")
    return dict(zip(ids, feats))


class LazyTextEncoder:
    """저장해 두지 않은 글이 있을 때만 CLIP 글 모델을 불러온다."""

    def __init__(self, backend: str):
        self.backend, self.encode = backend, None

    def __call__(self, texts):
        if self.encode is None:
            self.encode = match_clip.load_text_encoder(self.backend)
        return self.encode(texts)


def evaluate(truth: np.ndarray, pred: np.ndarray) -> dict:
    """4가지 판정이 정확히 맞은 비율, 허위를 잡는 성능, 판정마다 맞힌 비율."""
    t_false, p_false = np.isin(truth, list(FALSE_NEWS)), np.isin(pred, list(FALSE_NEWS))
    tp = int((t_false & p_false).sum())
    return {"n": len(truth), "acc4": float((truth == pred).mean()), "acc2": float((t_false == p_false).mean()),
            "precision": tp / max(int(p_false.sum()), 1), "recall": tp / max(int(t_false.sum()), 1),
            "per_cat": [float((pred[truth == k] == k).mean()) if (truth == k).any() else float("nan")
                        for k in range(len(CATS))]}


def pct(x: float) -> str:
    return "-" if x != x else f"{x * 100:.1f}%"


def report(truth, pred_o, pred_c, templates) -> str:
    o, c = evaluate(truth, pred_o), evaluate(truth, pred_c)
    lines = ["## 세 판정을 합친 최종 판정", "",
             "판정 순서: 글이 거짓 → 허위 / 글-사진 불일치 → 허위 / 사진만 AI → 주의 / 나머지 → 정상 (각 판정 확률 0.5 기준)", "",
             "| 지표 | 원본 (사진 + 원래 글) | 카드 (잘라낸 사진 + 읽어낸 글) |", "|---|---|---|",
             f"| 4가지 판정이 정확히 맞은 비율 | {pct(o['acc4'])} | {pct(c['acc4'])} |",
             f"| 허위인지 아닌지 맞은 비율 | {pct(o['acc2'])} | {pct(c['acc2'])} |",
             f"| 허위를 잡은 비율 (진짜 허위 중) | {pct(o['recall'])} | {pct(c['recall'])} |",
             f"| 허위 판정이 맞은 비율 (허위라고 한 것 중) | {pct(o['precision'])} | {pct(c['precision'])} |", "",
             "**정답별로 맞힌 비율**", "", "| 정답 | 카드 수 | 원본 | 카드 |", "|---|---|---|---|"]
    counts = Counter(truth.tolist())
    for k, name in enumerate(CATS):
        lines.append(f"| {name} | {counts[k]:,} | {pct(o['per_cat'][k])} | {pct(c['per_cat'][k])} |")
    lines += ["", "**카드에서 판정이 어떻게 나왔나 (줄: 정답, 칸: 판정)**", "",
              "| 정답 \\ 판정 | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
    for k, name in enumerate(CATS):
        row = [int(((truth == k) & (pred_c == j)).sum()) for j in range(len(CATS))]
        lines.append(f"| {name} | " + " | ".join(f"{v:,}" for v in row) + " |")
    lines += ["", "**템플릿별 (카드)**", "", "| 템플릿 | 카드 수 | 4가지 판정 정확도 | 허위를 잡은 비율 | 원본 4가지 판정 정확도 |",
              "|---|---|---|---|---|"]
    for t, name in factcheck.TEMPLATE_KO.items():
        m = templates == t
        if m.any():
            em, eo = evaluate(truth[m], pred_c[m]), evaluate(truth[m], pred_o[m])
            lines.append(f"| {name} | {int(m.sum()):,} | {pct(em['acc4'])} | {pct(em['recall'])} | {pct(eo['acc4'])} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="세 판정을 합친 최종 판정")
    parser.add_argument("--data", default=str(REPO_ROOT / "data"))
    parser.add_argument("--model", default=factcheck.MODEL, help="가짜 글 판정에 쓴 모델")
    parser.add_argument("--backend", default="auto", choices=["auto", "openvino", "onnxruntime"])
    args = parser.parse_args(argv)
    data = Path(args.data)
    feat_dir = data / "univfd" / "features"
    for path in (data / "univfd" / "head.pkl", data / "match" / "head.pkl"):
        if not path.exists():
            raise SystemExit(f"{path}가 없어요. univfd.py와 match_clip.py를 먼저 실행해 주세요.")
    ai_head = pickle.loads((data / "univfd" / "head.pkl").read_bytes())
    match_head = pickle.loads((data / "match" / "head.pkl").read_bytes())

    cards = factcheck.read_csv(data / "cards" / "cards.csv")
    ocr = {r["card_id"]: r for r in factcheck.read_csv(data / "cards" / "ocr.csv")}
    splits = sorted({c["split"] for c in cards})
    photo, crop = {}, {}
    for split in splits:
        photo.update(load_cached(feat_dir, f"photo-{split}"))
        crop.update(load_cached(feat_dir, f"crop-combined-{split}"))
        crop.update(load_cached(feat_dir, f"crop-{split}"))  # 교차 시험 카드 (univfd.py에서 계산)

    test = factcheck.load_records(data / "factcheck" / f"{args.model}-test.jsonl")
    read_back = factcheck.load_records(data / "factcheck" / f"{args.model}-cards.jsonl")
    rows, missing = [], []
    for c in cards:
        o = ocr[c["card_id"]]
        same = o["same_as_original"] == "1"
        before = test.get(factcheck.text_id(c["text"]))
        after = before if same else read_back.get(factcheck.text_id(o["ocr_text"]))
        if not before or not after or c["image_id"] not in photo or c["card_id"] not in crop:
            missing.append(c["card_id"])
            continue
        rows.append({"card": c, "ocr_text": c["text"] if same else o["ocr_text"],
                     "p_fake_original": before["p_false"], "p_fake_card": after["p_false"]})
    if missing:
        raise SystemExit(f"판정이나 특징이 없는 카드가 {len(missing)}장 있어요 (예: {missing[0]}). 앞 단계를 먼저 실행해 주세요.")

    encode = LazyTextEncoder(args.backend)
    texts = match_clip.embed_texts([r["card"]["text"] for r in rows] + [r["ocr_text"] for r in rows], encode,
                                   data / "match" / "text_features.npz")
    n = len(rows)
    img_o = np.stack([photo[r["card"]["image_id"]] for r in rows])
    img_c = np.stack([crop[r["card"]["card_id"]] for r in rows])
    p_ai_o = ai_head.predict_proba(univfd.l2(img_o))[:, 1]
    p_ai_c = ai_head.predict_proba(univfd.l2(img_c))[:, 1]
    p_mis_o = match_head.predict_proba(match_clip.pair_features(img_o, texts[:n]))[:, 1]
    p_mis_c = match_head.predict_proba(match_clip.pair_features(img_c, texts[n:]))[:, 1]
    p_fake_o = np.array([r["p_fake_original"] for r in rows])
    p_fake_c = np.array([r["p_fake_card"] for r in rows])

    truth = np.array([decide(int(r["card"]["text_fake"]), int(r["card"]["is_mismatch"]), int(r["card"]["is_ai"]))
                      for r in rows])
    pred_o = np.array([decide(f >= 0.5, m >= 0.5, a >= 0.5) for f, m, a in zip(p_fake_o, p_mis_o, p_ai_o)])
    pred_c = np.array([decide(f >= 0.5, m >= 0.5, a >= 0.5) for f, m, a in zip(p_fake_c, p_mis_c, p_ai_c)])
    templates = np.array([r["card"]["template"] for r in rows])

    out = data / "final"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "final.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({"card_id": r["card"]["card_id"], "template": templates[i], "truth": CATS[truth[i]],
                             "pred_original": CATS[pred_o[i]], "pred_card": CATS[pred_c[i]],
                             "p_fake_original": f"{p_fake_o[i]:.4f}", "p_mismatch_original": f"{p_mis_o[i]:.4f}",
                             "p_ai_original": f"{p_ai_o[i]:.4f}", "p_fake_card": f"{p_fake_c[i]:.4f}",
                             "p_mismatch_card": f"{p_mis_c[i]:.4f}", "p_ai_card": f"{p_ai_c[i]:.4f}"})
    text = report(truth, pred_o, pred_c, templates)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"결과: {out / 'final.csv'} ({n:,}장), {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
