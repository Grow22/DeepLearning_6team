"""MobileNetV2로 실제/AI 사진과 사진·글 짝을 팀 공통 데이터에서 평가한다."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, normalize


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
MODEL_REPO = "onnxmodelzoo/mobilenetv2-12"
MODEL_REV = "23a2a25eaaa6fcd61865414c8a1acf13dda66b7a"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def image_encoder(weights, threads):
    # 고정한 ONNX 가중치에서 ImageNet 분류기 직전의 평균 풀링 출력을 읽는다.
    model = onnx.load(weights)
    pool = next(n for n in model.graph.node if n.op_type == "GlobalAveragePool")
    del model.graph.output[:]
    model.graph.output.append(onnx.helper.make_tensor_value_info(
        pool.output[0], onnx.TensorProto.FLOAT, ["batch_size", 1280, 1, 1]
    ))
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def prepare_image(path):
    with Image.open(path) as image:
        crop = ImageOps.fit(image.convert("RGB"), (224, 224), method=Image.Resampling.BILINEAR)
        pixels = np.asarray(crop, dtype=np.float32) / 255
    return ((pixels - MEAN) / STD).transpose(2, 0, 1)


def encode_photos(rows, path_column, root, session, cache_dir, batch_size):
    blocks = []
    input_name = session.get_inputs()[0].name
    for split, group in rows.groupby("split", sort=False):
        ids = group["image_id"].to_numpy(dtype=str)
        cache = cache_dir / f"{split}-{path_column}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if np.array_equal(saved["ids"], ids):
                    blocks.append(pd.DataFrame(saved["features"], index=ids))
                    continue
        paths = group[path_column].tolist()
        features = []
        for start in range(0, len(paths), batch_size):
            batch = np.stack([prepare_image(root / p) for p in paths[start:start + batch_size]])
            pooled = session.run(None, {input_name: batch})[0]
            features.append(pooled.reshape(len(batch), -1))
        vectors = np.concatenate(features)
        with NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as temp:
            np.savez(temp, ids=ids, features=vectors)
        Path(temp.name).replace(cache)
        blocks.append(pd.DataFrame(vectors, index=ids))
        print(f"{split} {path_column}: {len(ids)}장", flush=True)
    return pd.concat(blocks)


def choose_threshold(labels, scores):
    # validation에서 balanced accuracy 최대. 동률이면 0.5에 가까운 기준을 택한다.
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    gain = tpr[finite] - fpr[finite]
    best = thresholds[finite][np.isclose(gain, gain.max(), atol=1e-12, rtol=0)]
    return float(best[np.argmin(np.abs(best - 0.5))])


def record_metrics(frame, task, split, threshold):
    y = frame["label"].to_numpy()
    score = frame["score"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, score >= threshold, labels=[0, 1]).ravel()
    return dict(task=task, split=split, n=len(y), accuracy=(tn + tp) / len(y),
                auc=roc_auc_score(y, score), fpr=fp / (tn + fp), tn=tn, fp=fp,
                fn=fn, tp=tp, threshold=threshold)


def summarize(frame, task, threshold):
    rows = [record_metrics(frame, task, "all", threshold)]
    rows.extend(record_metrics(group, task, split, threshold)
                for split, group in frame.groupby("split", sort=True))
    return rows


def photo_experiment(photos, original, normalized):
    train = photos[photos["split"] == "train"]
    val = photos[photos["split"] == "validation"]
    test = photos[photos["split"].str.startswith("test")].copy()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    model.fit(original.loc[train["image_id"]].to_numpy(), train["label"])
    val_score = model.predict_proba(original.loc[val["image_id"]].to_numpy())[:, 1]
    threshold = choose_threshold(val["label"], val_score)
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    weight = classifier.coef_[0].astype(np.float64) / scaler.scale_
    head = {
        "encoder_revision": MODEL_REV, "positive_class": "AI", "threshold": threshold,
        "intercept": float(classifier.intercept_[0] - weight @ scaler.mean_),
        "weight": weight.tolist()
    }
    results, metrics = [], []
    for view, features in (("original", original), ("normalized", normalized)):
        part = test[test["image_id"].isin(features.index)][["image_id", "split", "label"]].copy()
        part = part.rename(columns={"image_id": "id"})
        part["input"] = view
        part["score"] = model.predict_proba(features.loc[part["id"]].to_numpy())[:, 1]
        part["prediction"] = (part["score"] >= threshold).astype(int)
        results.append(part)
        metrics.extend(summarize(part, "photo_" + view, threshold))
    return pd.concat(results, ignore_index=True), metrics, head


def predict_photo(path, session, head_path):
    with head_path.open(encoding="utf-8") as handle:
        head = json.load(handle)
    if head["encoder_revision"] != MODEL_REV:
        raise ValueError("분류기와 인코더의 버전이 다릅니다.")
    batch = prepare_image(path)[None, ...]
    features = session.run(None, {session.get_inputs()[0].name: batch})[0].ravel()
    score = float(expit(features @ np.asarray(head["weight"]) + head["intercept"]))
    return {"is_ai": score >= head["threshold"], "ai_score": score, "threshold": head["threshold"]}


def ids_from_paths(paths):
    identifiers = []
    for raw in paths:
        path = Path(raw)
        identifiers.append(f"{path.parent.name}-{int(path.stem):05d}")
    return identifiers


def pair_experiment(data, original, combined):
    levels = []
    for level in ("easy", "hard"):
        frame = pd.read_csv(data / "mismatch" / (level + ".csv"))
        frame["level"] = level
        levels.append(frame.rename(columns={"pair_id": "id", "group": "split"}))
    pairs = pd.concat(levels, ignore_index=True)
    pairs["image_id"] = ids_from_paths(pairs["image_path"])

    # 동일 사진의 정답 짝이 easy/hard에 반복되므로 학습에는 한 벌만 사용한다.
    matched = pairs[(pairs["split"] == "train") & (pairs["level"] == "easy") & (pairs["label"] == 0)]
    text_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=14000,
                        stop_words="english", sublinear_tf=True),
        TruncatedSVD(n_components=128, random_state=42), Normalizer()
    )
    caption_vectors = text_model.fit_transform(matched["text"])
    image_to_text = make_pipeline(StandardScaler(), Normalizer(), Ridge(alpha=2.0, solver="cholesky"))
    image_to_text.fit(original.loc[matched["image_id"]].to_numpy(), caption_vectors)
    projected = pd.DataFrame(normalize(image_to_text.predict(original.to_numpy())), index=original.index)

    # 각 쌍의 사진과 주어진 글로 점수를 계산한다. ID는 해당 사진의 특징을 찾는 데만 쓴다.
    held_out = pairs[pairs["split"] != "train"].copy()
    joined = combined.rename(columns={"item_id": "id", "is_mismatch": "label"}).copy()
    joined["image_id"] = ids_from_paths(joined["image_path"])
    joined["level"] = "combined"
    evaluated = pd.concat([held_out, joined], ignore_index=True)
    unique_text, indices = np.unique(evaluated["text"].to_numpy(dtype=str), return_inverse=True)
    text_vectors = text_model.transform(unique_text)
    image_vectors = projected.loc[evaluated["image_id"]].to_numpy()
    cosine = np.einsum("ij,ij->i", image_vectors, text_vectors[indices])
    evaluated["score"] = (1 - np.clip(cosine, -1, 1)) / 2
    val = evaluated[evaluated["split"] == "validation"]
    threshold = choose_threshold(val["label"], val["score"])
    test = evaluated[evaluated["split"].str.startswith("test")].copy()
    test["prediction"] = (test["score"] >= threshold).astype(int)
    metrics = []
    for level, group in test.groupby("level", sort=False):
        metrics.extend(summarize(group, "pair_" + level, threshold))
    test["set"] = np.where(test["level"] == "combined", "combined", test["level"] + "-" + test["split"])
    return test[["set", "id", "split", "label", "score", "prediction"]], metrics


def combined_results(source, photo_scores, pair_scores):
    out = source[["item_id", "split", "is_ai", "is_mismatch"]].copy()
    source_ids = ids_from_paths(source["image_path"])
    image = photo_scores[photo_scores["input"] == "original"].set_index("id").loc[source_ids]
    pair = pair_scores[pair_scores["set"] == "combined"].set_index("id").loc[out["item_id"]]
    if not (np.array_equal(image["label"], out["is_ai"])
            and np.array_equal(pair["label"], out["is_mismatch"])):
        raise ValueError("통합 시험의 정답과 개별 시험 ID 매핑을 확인하세요.")
    out["ai_score"] = image["score"].to_numpy()
    out["ai_pred"] = image["prediction"].to_numpy()
    out["mismatch_score"] = pair["score"].to_numpy()
    out["mismatch_pred"] = pair["prediction"].to_numpy()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--image", type=Path, help="저장된 분류기로 사진 한 장을 판정한다.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size와 --threads는 양수여야 합니다.")
    weights = hf_hub_download(MODEL_REPO, "mobilenetv2-12.onnx", revision=MODEL_REV)
    session = image_encoder(weights, args.threads)
    if args.image is not None:
        print(json.dumps(predict_photo(args.image, session, args.out / "image_head.json")))
        return
    root = args.data / "ai_photos"
    photos = pd.read_csv(root / "ai_photos.csv")
    required_splits = {"train", "validation", "test1_nyt_mj", "test2_bbc_dalle",
                       "test3_cnn_dalle", "test4_bbc_sdxl", "test5_cnn_sdxl"}
    if not required_splits.issubset(set(photos["split"])):
        parser.error("먼저 저장소 루트에서 python data_preparation/prepare_all.py 를 실행하세요.")
    cache = args.data / "youngrok_mobilenet" / (MODEL_REV[:12] + "-rgb224-v1")
    cache.mkdir(parents=True, exist_ok=True)
    original = encode_photos(photos, "image_path", root, session, cache, args.batch_size)
    test = photos[photos["split"].str.startswith("test")]
    if test["normalized_path"].isna().any():
        parser.error("정규화된 테스트 사진이 누락됐습니다. prepare_all.py를 다시 실행하세요.")
    normalized = encode_photos(test, "normalized_path", root, session, cache, args.batch_size)
    combined = pd.read_csv(root / "combined_test.csv")
    image_scores, metrics, head = photo_experiment(photos, original, normalized)
    pair_scores, pair_metrics = pair_experiment(args.data, original, combined)
    tables = {
        "image_scores.csv": image_scores, "match_scores.csv": pair_scores,
        "combined_scores.csv": combined_results(combined, image_scores, pair_scores),
        "metrics.csv": pd.DataFrame(metrics + pair_metrics)
    }
    args.out.mkdir(exist_ok=True, parents=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.out, delete=False) as tmp:
        json.dump(head, tmp)
        tmp.write("\n")
    Path(tmp.name).replace(args.out / "image_head.json")
    for filename, table in tables.items():
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=args.out, delete=False) as tmp:
            table.to_csv(tmp, index=False)
        Path(tmp.name).replace(args.out / filename)
    print(tables["metrics.csv"].query("split == 'all'").to_string(index=False))


if __name__ == "__main__":
    main()
