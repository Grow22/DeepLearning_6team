#!/usr/bin/env python3
"""UnivFD 방식 AI 사진 판정 도구: CLIP ViT-L/14 특징 + 선형 분류기.

UnivFD(Ojha 외, CVPR 2023)처럼 CLIP은 그대로 두고, 그 위의 선형 분류기만 학습한다.
1. CLIP ViT-L/14가 사진 한 장을 숫자 768개(특징)로 바꾼다.
2. train 사진(진짜 5,000 + AI 5,000)의 특징으로 선형 분류기(로지스틱 회귀)를 학습하고,
   validation 사진 2,500장으로 규제 세기(C)를 고른다.
3. 테스트 사진을 세 가지로 넣어 AI일 확률(0~1)을 낸다.
   ① 원본 사진 2,500장  ② 카드 전체  ③ 카드에서 잘라낸 사진 (② ③은 교차 시험 카드 5,000장)

실행 (저장소 맨 위 폴더에서, prepare_all.py와 cards/make_cards.py를 먼저 실행한 뒤):
    pip install onnxruntime huggingface_hub scikit-learn pillow numpy
    pip install openvino      # 선택: 인텔 CPU에서 몇 배 빨라짐
    python results/장성준/univfd.py
    python results/장성준/score_cards.py data/univfd/scores.csv

결과 (data/ 폴더는 깃에 올라가지 않는다):
    data/univfd/features/   사진 특징 (중간에 멈춰도 다시 실행하면 이어서 계산)
    data/univfd/scores.csv  입력마다 AI일 확률
    data/univfd/head.json   고른 설정과 validation 점수
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MODEL_REPO = "Xenova/clip-vit-large-patch14"  # OpenAI CLIP ViT-L/14(UnivFD와 같은 모델)를 ONNX로 바꾼 것
MODEL_FILE = "onnx/vision_model.onnx"
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)  # CLIP 전처리 값
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
TEST_SPLITS = ["test1_nyt_mj", "test2_bbc_dalle", "test3_cnn_dalle", "test4_bbc_sdxl", "test5_cnn_sdxl"]
CHUNK = 500  # 이만큼 계산할 때마다 저장한다
C_GRID = [0.001, 0.01, 0.1, 1.0]
SCORE_FIELDS = ["input", "id", "split", "score"]


def preprocess(img: Image.Image) -> np.ndarray:
    """CLIP 기본 전처리: 짧은 변을 224로 줄이고 가운데 224x224를 잘라 정규화한다. 결과 (3, 224, 224)."""
    img = img.convert("RGB")
    w, h = img.size
    s = SIZE / min(w, h)
    img = img.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    arr = np.asarray(img.crop((left, top, left + SIZE, top + SIZE)), dtype=np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(2, 0, 1)


def pick_backend(backend: str) -> str:
    """auto: GPU용 onnxruntime이 있으면 그것, 없으면 openvino, 그것도 없으면 onnxruntime CPU."""
    if backend != "auto":
        return backend
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "onnxruntime"
    except ImportError:
        pass
    try:
        import openvino  # noqa: F401
        return "openvino"
    except ImportError:
        return "onnxruntime"


def load_model(backend: str):
    """사진 묶음 (N, 3, 224, 224) → 특징 (N, 768) 함수를 돌려준다."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    backend = pick_backend(backend)
    if backend == "openvino":
        import openvino as ov
        compiled = ov.Core().compile_model(path, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
        print(f"계산: OpenVINO CPU ({compiled.get_property('INFERENCE_PRECISION_HINT')})", flush=True)
        return lambda x: np.asarray(compiled(x)[0])
    import onnxruntime as ort
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    session = ort.InferenceSession(path, providers=providers)
    print(f"계산: onnxruntime {session.get_providers()[0]}", flush=True)
    return lambda x: session.run(["image_embeds"], {"pixel_values": x})[0]


def embed(name: str, ids: list[str], load, model, feat_dir: Path, batch: int = 16) -> np.ndarray:
    """ids 순서대로 특징을 계산한다. load(k) → k번째 PIL 이미지.

    CHUNK장마다 저장하고, 다시 실행하면 저장된 조각은 건너뛴다. ids가 바뀌면 처음부터 다시 계산한다.
    """
    folder = feat_dir / name
    id_file = folder / "ids.txt"
    if folder.exists() and (not id_file.exists() or id_file.read_text(encoding="utf-8").split("\n") != ids):
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    id_file.write_text("\n".join(ids), encoding="utf-8")
    parts, started, computed = [], time.time(), 0
    for k, start in enumerate(range(0, len(ids), CHUNK)):
        part = folder / f"{k:04d}.npy"
        end = min(start + CHUNK, len(ids))
        if not part.exists():
            feats = []
            for b in range(start, end, batch):
                x = np.stack([preprocess(load(i)) for i in range(b, min(b + batch, end))])
                feats.append(np.asarray(model(x), dtype=np.float32))
            np.save(part, np.concatenate(feats))
            computed += end - start
            spent = time.time() - started
            left = spent / computed * (len(ids) - end)
            print(f"  {name}: {end}/{len(ids)}장 ({spent / 60:.1f}분, 남은 시간 약 {left / 60:.0f}분)", flush=True)
        parts.append(np.load(part))
    feats = np.concatenate(parts) if parts else np.zeros((0, 768), dtype=np.float32)
    if len(feats) != len(ids):
        raise SystemExit(f"{name}: 특징 수({len(feats)})가 사진 수({len(ids)})와 달라요. {folder}를 지우고 다시 실행해 주세요.")
    return feats


def open_rgb(path: Path, box=None) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
    return im.crop(box) if box else im


def fit_head(x_train, y_train, x_val, y_val):
    """C를 바꿔 가며 로지스틱 회귀를 학습하고, validation AUC가 가장 높은 것을 고른다."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    best = None
    for c in C_GRID:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_val)[:, 1]
        auc, acc = roc_auc_score(y_val, prob), float(((prob >= 0.5) == y_val).mean())
        print(f"  C={c}: validation AUC {auc:.4f}, 정확도 {acc:.4f}", flush=True)
        if best is None or auc > best["val_auc"]:
            best = {"C": c, "val_auc": float(auc), "val_acc": acc, "model": clf}
    return best


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def balanced(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """빠른 시험용: 진짜·AI를 반반 섞어 limit개만 고른다 (0이면 전부)."""
    if not limit or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    pick = []
    for label in ("0", "1"):
        group = [r for r in rows if r["label"] == label]
        pick += rng.sample(group, min(len(group), limit // 2))
    return sorted(pick, key=lambda r: r["image_id"])


def main(argv: list[str] | None = None, model=None) -> int:
    parser = argparse.ArgumentParser(description="UnivFD 방식 AI 사진 판정 (CLIP ViT-L/14 + 선형 분류기)")
    parser.add_argument("--data", default=str(REPO_ROOT / "data"), help="prepare_all.py·make_cards.py로 받은 data 폴더")
    parser.add_argument("--out", default=None, help="결과 폴더 (기본: data/univfd)")
    parser.add_argument("--backend", default="auto", choices=["auto", "openvino", "onnxruntime"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="빠른 시험용: 묶음마다 이만큼만 쓰기 (0 = 전부)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    data = Path(args.data)
    out = Path(args.out) if args.out else data / "univfd"
    photos_csv, cards_csv = data / "ai_photos" / "ai_photos.csv", data / "cards" / "cards.csv"
    if not photos_csv.exists():
        raise SystemExit("사진이 없어요. 먼저 python data_preparation/prepare_all.py 를 실행해 주세요.")
    if not cards_csv.exists():
        raise SystemExit("카드가 없어요. 먼저 python data_preparation/cards/make_cards.py 를 실행해 주세요.")
    with open(photos_csv, newline="", encoding="utf-8") as f:
        photos = defaultdict(list)
        for r in csv.DictReader(f):
            photos[r["split"]].append(r)
    with open(cards_csv, newline="", encoding="utf-8") as f:
        cards = defaultdict(list)
        for r in csv.DictReader(f):
            if r["cross_item_id"]:  # AI 사진 판정은 교차 시험 카드로 (사진마다 사실·거짓 글 카드가 하나씩)
                cards[r["split"]].append(r)
    for split in ["train", "validation", *TEST_SPLITS]:
        if not photos[split]:
            raise SystemExit(f"{split} 사진이 없어요. python data_preparation/prepare_all.py 를 실행해 주세요.")

    started = time.time()
    model = model or load_model(args.backend)
    feat_dir = out / "features"
    feats, rows_of = {}, {}

    def photo_set(split):
        rows = balanced(photos[split], args.limit, args.seed)
        rows_of[f"photo-{split}"] = rows
        feats[f"photo-{split}"] = embed(
            f"photo-{split}", [r["image_id"] for r in rows],
            lambda k: open_rgb(data / "ai_photos" / rows[k]["image_path"]), model, feat_dir, args.batch)

    def card_sets(split):
        rows = cards[split][: args.limit] if args.limit else cards[split]
        boxes = [tuple(int(r[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2")) for r in rows]
        ids = [r["card_id"] for r in rows]
        for kind in ("card", "crop"):
            rows_of[f"{kind}-{split}"] = rows
            feats[f"{kind}-{split}"] = embed(
                f"{kind}-{split}", ids,
                lambda k, kind=kind: open_rgb(data / "cards" / rows[k]["card_path"], boxes[k] if kind == "crop" else None),
                model, feat_dir, args.batch)

    print("[1/3] 특징 계산 (CLIP ViT-L/14)", flush=True)
    for split in ["train", "validation", *TEST_SPLITS]:
        photo_set(split)
    for split in TEST_SPLITS:
        card_sets(split)

    print("[2/3] 선형 분류기 학습", flush=True)
    y = {s: np.array([int(r["label"]) for r in rows_of[f"photo-{s}"]]) for s in ("train", "validation")}
    head = fit_head(l2(feats["photo-train"]), y["train"], l2(feats["photo-validation"]), y["validation"])
    print(f"  고른 설정: C={head['C']} (validation AUC {head['val_auc']:.4f}, 정확도 {head['val_acc']:.4f})", flush=True)

    print("[3/3] 테스트 점수 내기", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for split in TEST_SPLITS:
        for kind, name, id_key in (("original", f"photo-{split}", "image_id"), ("card", f"card-{split}", "card_id"),
                                   ("crop", f"crop-{split}", "card_id")):
            if not len(feats[name]):
                continue
            prob = head["model"].predict_proba(l2(feats[name]))[:, 1]
            rows_out += [{"input": kind, "id": r[id_key], "split": split, "score": f"{p:.6f}"}
                         for r, p in zip(rows_of[name], prob)]
    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    info = {k: v for k, v in head.items() if k != "model"}
    info.update({"model": MODEL_REPO, "train": len(y["train"]), "validation": len(y["validation"]),
                 "limit": args.limit, "seed": args.seed})
    (out / "head.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료 ({(time.time() - started) / 60:.1f}분) → {out / 'scores.csv'} ({len(rows_out)}줄)", flush=True)
    print(f"채점: python results/{HERE.name}/score_cards.py {out / 'scores.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
