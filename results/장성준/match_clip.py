#!/usr/bin/env python3
"""글-사진 짝 판정 도구 (CLIP ViT-L/14 + 선형 분류기): 글이 사진과 맞는지 판정한다.

UnivFD 방식처럼 CLIP은 그대로 두고, 그 위에 "사진과 글이 맞는지" 가르는 선형 분류기만 학습한다.
- 특징: 사진 특징과 글 특징을 칸마다 곱한 값 768개 + 두 특징의 코사인 유사도
- 학습: 글-사진 짝 데이터의 train (easy + hard). 규제 세기(C)는 validation으로 고른다.
- 시험: ① 글-사진 짝 test (easy, hard)
        ② 통합 시험 원본 (사진 + 원래 글)
        ③ 통합 시험 카드 (카드에서 잘라낸 사진 + 카드에서 읽어낸 글)
사진 특징은 univfd.py와 같은 것을 쓴다 (이미 계산했으면 다시 계산하지 않는다).

실행 (저장소 맨 위 폴더에서, prepare_all.py → make_cards.py → univfd.py → ocr_cards.py를 실행한 뒤):
    pip install onnxruntime huggingface_hub tokenizers scikit-learn pillow numpy
    python results/장성준/match_clip.py
결과 (data/ 폴더는 깃에 올라가지 않는다):
    data/match/scores.csv   문항마다 "안 맞을" 확률
    data/match/report.md    결과 표
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import univfd  # noqa: E402  (사진 특징: 같은 CLIP ViT-L/14, 같은 전처리, 같은 저장 방식)

TEXT_FILE = "onnx/text_model.onnx"
MAX_TOKENS = 77  # CLIP이 읽는 글 길이 (넘으면 뒷부분은 잘린다)
PAD_ID = 49407
C_GRID = [0.001, 0.01, 0.1, 1.0]
TEST_GROUPS = ["test_nyt", "test_bbc", "test_cnn"]
SCORE_FIELDS = ["set", "id", "label", "score", "cosine"]


def norm(text: str) -> str:
    return " ".join(text.split())


def text_key(text: str) -> str:
    return hashlib.sha1(norm(text).encode("utf-8")).hexdigest()


def image_id_from_path(path: str) -> str:
    """images/<split>/<행>.<확장자> → ai_photos.csv의 image_id (<split>-<행 5자리>)."""
    parts = Path(path).parts
    return f"{parts[-2]}-{int(Path(path).stem):05d}"


# ---------------------------------------------------------------------------
# 특징
# ---------------------------------------------------------------------------
def load_text_encoder(backend: str):
    """글 묶음 → CLIP 글 특징 (N, 768) 함수."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(hf_hub_download(univfd.MODEL_REPO, "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_TOKENS)
    tok.enable_padding(length=MAX_TOKENS, pad_id=PAD_ID, pad_token="<|endoftext|>")
    path = hf_hub_download(univfd.MODEL_REPO, TEXT_FILE)
    if univfd.pick_backend(backend) == "openvino":
        import openvino as ov
        compiled = ov.Core().compile_model(path, "CPU", {"PERFORMANCE_HINT": "LATENCY"})

        def run(ids):
            return np.asarray(compiled({"input_ids": ids})[0])
    else:
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        session = ort.InferenceSession(path, providers=providers)

        def run(ids):
            return session.run(["text_embeds"], {"input_ids": ids})[0]

    def encode(texts: list[str]) -> np.ndarray:
        enc = tok.encode_batch([norm(t) for t in texts])
        return run(np.array([e.ids for e in enc], dtype=np.int64)).astype(np.float32)
    return encode


def embed_texts(texts: list[str], encode, cache_path: Path, batch: int = 32) -> np.ndarray:
    """글 특징을 계산한다. 한 번 계산한 글은 cache_path에 저장해 두고 다시 쓴다."""
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    need = sorted({norm(t) for t in texts if text_key(t) not in cache})
    started = time.time()
    for b in range(0, len(need), batch):
        chunk = need[b:b + batch]
        cache.update(zip(map(text_key, chunk), encode(chunk)))
        if (b // batch + 1) % 50 == 0 or b + batch >= len(need):
            print(f"  글 특징 {min(b + batch, len(need))}/{len(need)}개 ({(time.time() - started) / 60:.1f}분)", flush=True)
    if need:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, keys=np.array(list(cache)), vecs=np.stack(list(cache.values())))
    return np.stack([cache[text_key(t)] for t in texts]) if texts else np.zeros((0, 768), np.float32)


def pair_features(img: np.ndarray, txt: np.ndarray) -> np.ndarray:
    """사진·글 특징 → (칸마다 곱한 값 768개, 코사인 유사도 1개)."""
    img, txt = univfd.l2(img), univfd.l2(txt)
    prod = img * txt
    return np.hstack([prod, prod.sum(axis=1, keepdims=True)])


class LazyImageModel:
    """사진 특징을 새로 계산해야 할 때만 CLIP을 불러온다 (이미 계산해 둔 특징은 그대로 읽는다)."""

    def __init__(self, backend: str):
        self.backend, self.model = backend, None

    def __call__(self, x):
        if self.model is None:
            self.model = univfd.load_model(self.backend)
        return self.model(x)


# ---------------------------------------------------------------------------
# 학습과 평가
# ---------------------------------------------------------------------------
def fit_head(x_train, y_train, x_val, y_val):
    """C를 바꿔 가며 학습하고 validation AUC가 가장 높은 것을 고른다."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    best = None
    for c in C_GRID:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, prob)
        print(f"  C={c}: validation AUC {auc:.4f}, 정확도 {((prob >= 0.5) == y_val).mean():.4f}", flush=True)
        if best is None or auc > best["val_auc"]:
            best = {"C": c, "val_auc": float(auc), "val_acc": float(((prob >= 0.5) == y_val).mean()), "model": clf}
    return best


def summarize(y: np.ndarray, prob: np.ndarray, cos: np.ndarray) -> dict:
    import score_cards
    pred = prob >= 0.5
    return {"n": len(y), "acc": float((pred == y).mean()), "auc": score_cards.auc(y, prob),
            "auc_cos": score_cards.auc(y, -cos),  # CLIP 유사도만 쓸 때 (유사도가 낮을수록 안 맞음)
            "match": float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan"),
            "mismatch": float(pred[y == 1].mean()) if (y == 1).any() else float("nan")}


def report(results: dict, head: dict) -> str:
    def pct(x):
        return "-" if x != x else f"{x * 100:.1f}%"
    lines = ["## 글-사진 짝 판정 결과 (CLIP ViT-L/14 + 선형 분류기)", "",
             f"학습: 글-사진 짝 train · 고른 설정 C={head['C']} (validation AUC {head['val_auc']:.3f}) · "
             "안 맞을 확률 0.5 이상이면 안 맞음", "",
             "| 시험 | 문항 | 정확도 | AUC | 맞는 짝을 맞다고 | 안 맞는 짝을 안 맞다고 | 참고: CLIP 유사도만 AUC |",
             "|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['n']:,} | {pct(m['acc'])} | {m['auc']:.3f} | {pct(m['match'])} | "
                     f"{pct(m['mismatch'])} | {m['auc_cos']:.3f} |")
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None, image_model=None, text_encoder=None) -> int:
    parser = argparse.ArgumentParser(description="글-사진 짝 판정 (CLIP ViT-L/14 + 선형 분류기)")
    parser.add_argument("--data", default=str(REPO_ROOT / "data"))
    parser.add_argument("--backend", default="auto", choices=["auto", "openvino", "onnxruntime"])
    args = parser.parse_args(argv)
    data = Path(args.data)
    need = {"사진": data / "ai_photos" / "ai_photos.csv", "글-사진 짝": data / "mismatch" / "easy.csv",
            "카드": data / "cards" / "cards.csv", "카드 글자": data / "cards" / "ocr.csv"}
    for name, path in need.items():
        if not path.exists():
            raise SystemExit(f"{name} 데이터가 없어요 ({path}). 실행 순서: prepare_all.py → make_cards.py → ocr_cards.py")
    started = time.time()
    image_model = image_model or LazyImageModel(args.backend)
    feat_dir, out = data / "univfd" / "features", data / "match"

    # 사진 특징: univfd.py와 같은 이름·순서로 부르면 저장해 둔 것을 그대로 읽는다
    photos = {}
    for r in read_csv(data / "ai_photos" / "ai_photos.csv"):
        photos.setdefault(r["split"], []).append(r)
    print("[1/4] 사진 특징 (CLIP ViT-L/14)", flush=True)
    image_vec = {}
    for split, rows in photos.items():
        feats = univfd.embed(f"photo-{split}", [r["image_id"] for r in rows],
                             lambda k, rows=rows: univfd.open_rgb(data / "ai_photos" / rows[k]["image_path"]),
                             image_model, feat_dir)
        image_vec.update(zip((r["image_id"] for r in rows), feats))
    cards = [c for c in read_csv(data / "cards" / "cards.csv") if c["combined_item_id"]]
    ocr = {r["card_id"]: r["ocr_text"] for r in read_csv(data / "cards" / "ocr.csv")}
    crop_vec = {}
    for split in sorted({c["split"] for c in cards}):
        rows = [c for c in cards if c["split"] == split]
        boxes = [tuple(int(c[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2")) for c in rows]
        feats = univfd.embed(f"crop-combined-{split}", [c["card_id"] for c in rows],
                             lambda k, rows=rows, boxes=boxes: univfd.open_rgb(data / "cards" / rows[k]["card_path"], boxes[k]),
                             image_model, feat_dir)
        crop_vec.update(zip((c["card_id"] for c in rows), feats))

    # 문항: (이름, id, 사진 특징, 글, 정답 1 = 안 맞음)
    pairs = []
    for level in ("easy", "hard"):
        for r in read_csv(data / "mismatch" / f"{level}.csv"):
            pairs.append({"set": f"{level}-{r['group']}", "id": r["pair_id"], "img": image_vec[image_id_from_path(r["image_path"])],
                          "text": r["text"], "label": int(r["label"])})
    for r in read_csv(data / "ai_photos" / "combined_test.csv"):
        pairs.append({"set": "combined", "id": r["item_id"], "img": image_vec[image_id_from_path(r["image_path"])],
                      "text": r["text"], "label": int(r["is_mismatch"])})
    for c in cards:
        if c["card_id"] not in ocr:
            raise SystemExit("ocr.csv가 카드와 맞지 않아요. ocr_cards.py를 다시 실행해 주세요.")
        pairs.append({"set": "card", "id": c["card_id"], "img": crop_vec[c["card_id"]], "text": ocr[c["card_id"]],
                      "label": int(c["is_mismatch"])})

    print("[2/4] 글 특징 (CLIP ViT-L/14)", flush=True)
    encode = text_encoder or load_text_encoder(args.backend)
    txt = embed_texts([p["text"] for p in pairs], encode, out / "text_features.npz")
    x = pair_features(np.stack([p["img"] for p in pairs]), txt)
    y = np.array([p["label"] for p in pairs])
    sets = np.array([p["set"] for p in pairs])

    print("[3/4] 선형 분류기 학습", flush=True)
    train = np.isin(sets, ["easy-train", "hard-train"])
    val = np.isin(sets, ["easy-validation", "hard-validation"])
    head = fit_head(x[train], y[train], x[val], y[val])

    print("[4/4] 시험", flush=True)
    prob = head["model"].predict_proba(x)[:, 1]
    groups = {
        "글-사진 짝 easy (test)": np.isin(sets, [f"easy-{g}" for g in TEST_GROUPS]),
        "글-사진 짝 hard (test)": np.isin(sets, [f"hard-{g}" for g in TEST_GROUPS]),
        "통합 시험 원본 (사진 + 원래 글)": sets == "combined",
        "통합 시험 카드 (잘라낸 사진 + 읽어낸 글)": sets == "card",
    }
    results = {name: summarize(y[m], prob[m], x[m, -1]) for name, m in groups.items() if m.any()}
    out.mkdir(parents=True, exist_ok=True)
    tested = np.any(np.stack(list(groups.values())), axis=0)
    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for p, s, c, t in zip(pairs, prob, x[:, -1], tested):
            if t:
                writer.writerow({"set": p["set"], "id": p["id"], "label": p["label"], "score": f"{s:.6f}",
                                 "cosine": f"{c:.6f}"})
    text = report(results, head)
    (out / "report.md").write_text(text, encoding="utf-8")
    info = {k: v for k, v in head.items() if k != "model"}
    (out / "head.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(text)
    print(f"완료 ({(time.time() - started) / 60:.1f}분) → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
