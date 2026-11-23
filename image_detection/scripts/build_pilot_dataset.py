"""파일럿 데이터셋(진짜 50 + AI) 구성 스크립트.

1) Wikimedia Commons 후보 목록(all.json)에서 손으로 고른 사진 50장을 받는다.
2) 바탕화면 ai_images/{chatgpt,gemini}의 AI 이미지를 이름 규칙에 맞춰 복사한다.
3) data/meta.csv를 만든다.

원본 포맷은 그대로 둔다. 512px/JPEG q95/EXIF 제거는 Step 2 전처리에서 한다.

사용법:
  python build_pilot_dataset.py <all.json> <ai_images 폴더> <image_detection 폴더>
"""
import csv, json, os, re, shutil, sys, time
import requests
from PIL import Image

H = {"User-Agent": "dataset-collector/0.1 (research project)"}

# all.json의 idx -> 카테고리. 미리보기 시트를 보고 손으로 고름.
# 기준: 카테고리 장면과 맞는 것, 거의 같은 사진 중복 제외, 흑백/워터마크/문서 사진 제외.
PICKS = {
    "crowd":  [3, 23, 25, 30, 31, 42, 46, 51, 55, 207],
    "street": [78, 79, 85, 91, 123, 125, 126, 127, 128, 129],
    "speech": [0, 47, 48, 93, 105, 148, 150, 154, 168, 173],
    "vote":   [56, 58, 59, 60, 61, 62, 63, 69, 71, 90],
    "city":   [88, 89, 95, 106, 113, 146, 187, 190, 197, 206],
}
# 실존 인물 얼굴이 크게 나온 사진 (분석용 표시)
FACE_CLOSEUP = {25, 154}

FIELDS = ["img_id", "source_id", "label", "generator", "category", "file", "orig_format",
          "width", "height", "face_closeup", "origin_url", "license", "author", "note", "split"]


def get(url):
    for k in range(8):
        r = requests.get(url, headers=H, timeout=60)
        if r.status_code == 429:
            time.sleep(10 + 5 * k); continue
        r.raise_for_status(); return r.content
    raise RuntimeError("rate limited: " + url)


def main(all_json, ai_dir, out):
    cands = {c["idx"]: c for c in json.load(open(all_json, encoding="utf-8"))}
    real_dir = os.path.join(out, "data", "raw", "real"); os.makedirs(real_dir, exist_ok=True)
    ai_out = os.path.join(out, "data", "raw", "ai"); os.makedirs(ai_out, exist_ok=True)
    rows = []

    for cat, idxs in PICKS.items():
        for n, i in enumerate(idxs, 1):
            c = cands[i]; img_id = f"real_{cat}_{n:03d}"
            path = os.path.join(real_dir, img_id + ".jpg")
            if not os.path.exists(path):
                open(path, "wb").write(get(c["thumb"])); time.sleep(1)
            w, h = Image.open(path).size
            rows.append({"img_id": img_id, "source_id": img_id, "label": "real", "generator": "",
                         "category": cat, "file": f"data/raw/real/{img_id}.jpg", "orig_format": "JPEG",
                         "width": w, "height": h, "face_closeup": "yes" if i in FACE_CLOSEUP else "no",
                         "origin_url": c["page"], "license": c["license"], "author": c["artist"],
                         "note": "Wikimedia Commons thumbnail (1024px wide)", "split": ""})
            print(img_id, w, h, flush=True)

    for gen in ("chatgpt", "gemini"):
        d = os.path.join(ai_dir, gen)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            m = re.match(rf"ai_{gen}_([a-z]+)_+(\d+)\.(\w+)$", f)
            if not m:
                print("skip (name)", f); continue
            cat, num, ext = m.group(1), int(m.group(2)), m.group(3).lower()
            img_id = f"ai_{gen}_{cat}_{num:03d}"
            dst = os.path.join(ai_out, f"{img_id}.{ext}")
            shutil.copy2(os.path.join(d, f), dst)
            im = Image.open(dst)
            rows.append({"img_id": img_id, "source_id": img_id, "label": "ai", "generator": gen,
                         "category": cat, "file": f"data/raw/ai/{img_id}.{ext}", "orig_format": im.format,
                         "width": im.size[0], "height": im.size[1], "face_closeup": "no",
                         "origin_url": "", "license": "self-generated",
                         "author": "", "note": "Gemini sparkle watermark bottom-right; crop before use" if gen == "gemini" else "",
                         "split": ""})

    with open(os.path.join(out, "data", "meta.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, FIELDS); w.writeheader(); w.writerows(rows)
    print("rows", len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:4])
