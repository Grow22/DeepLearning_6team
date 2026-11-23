# AI 사진 판정 데이터 + 통합 시험 세트 (MiRAGeNews)

- **AI 사진 판정용**: MiRAGeNews의 진짜 사진과 AI 사진 15,000장을 정답(진짜/AI)과 함께 그대로 받아요.
- **통합 시험 세트**: 시험용 사진으로 "진짜/AI 사진 × 맞는/안 맞는 글" 네 경우를 섞은 2,500문항이에요. 두 판정을 합친 전체 시스템을 평가할 때 써요.

## 1. 데이터 받기

저장소 맨 위 폴더에서 실행해요.

```bash
pip install huggingface_hub pyarrow pillow
python ai_photos/prepare_ai_photos.py --normalize
```

- 처음 실행할 때 MiRAGeNews 원본 약 2.1GB를 받아요. 글-사진 짝 데이터를 이미 받았다면 다시 받지 않아요.
- 일부 split만 받으려면 `--splits test1_nyt_mj test2_bbc_dalle`처럼 붙여요.

결과는 `data/ai_photos/`에 생겨요 (깃에 올라가지 않아요).

```
data/ai_photos/
├── images/<split>/<번호>.<확장자>   원본 사진 15,000장 (약 2.5GB)
├── normalized/<split>/<번호>.jpg    정규화 사진 13,802장 (256×256 JPEG)
├── ai_photos.csv                    AI 사진 판정용 표
└── combined_test.csv                통합 시험 세트 표
```

**ai_photos.csv**

| 칸 | 뜻 |
|---|---|
| `image_id` | 사진 번호 (`<split>-<행 번호>`) |
| `split` | `train` / `validation` / `test1_nyt_mj` / `test2_bbc_dalle` / `test3_cnn_dalle` / `test4_bbc_sdxl` / `test5_cnn_sdxl` |
| `label` | 정답: 0 = 진짜, 1 = AI |
| `source`, `generator` | 뉴스 출처, AI 사진 생성기 (진짜 사진은 빈칸) |
| `image_path` | 원본 사진 위치 (`data/ai_photos/` 기준) |
| `format`, `width`, `height` | 원본 형식과 크기 |
| `normalized_path` | 정규화 사진 위치 (256px보다 작아 뺀 사진은 빈칸) |

**combined_test.csv**

| 칸 | 뜻 |
|---|---|
| `item_id`, `split` | 문항 번호, split |
| `image_path`, `normalized_path` | 사진 위치 |
| `text` | 사진에 붙인 글 |
| `is_ai` | 정답 1: 0 = 진짜 사진, 1 = AI 사진 |
| `is_mismatch` | 정답 2: 0 = 글이 맞음, 1 = 글이 안 맞음 |

## 2. 정규화를 꼭 같이 보세요

시험 세트 일부는 진짜 사진과 AI 사진의 겉모습이 달라서, 원본 그대로 쓰면 사진 내용이 아니라 겉모습만 보고도 AI를 맞힐 수 있어요.

| 시험 세트 | 진짜 사진 | AI 사진 |
|---|---|---|
| BBC (test2, test4) | 가로로 긴 JPEG, 왼쪽 아래에 BBC 로고가 있는 사진이 많음 | 정사각형 1024×1024 (DALL-E는 PNG) |
| CNN (test3, test5) | 가로로 긴 JPEG | 정사각형 1024×1024 (DALL-E는 PNG) |
| NYT (train, validation, test1) | JPEG | 진짜와 같은 크기의 JPEG (차이 없음) |

`--normalize`는 모든 사진의 가운데를 256×256으로 자르고(크기 변경 없음) 같은 품질의 JPEG로 저장해요. 그래서 형식, 모양, 구석의 로고 차이가 없어져요. 결과는 원본과 정규화 두 가지로 같이 내는 걸 추천해요.

256px보다 작은 사진(train 1,042장, validation 156장)은 정규화에서 빠져요. 진짜와 AI가 같은 수만큼 빠져서 반반 균형은 그대로예요.

## 3. 구성

| split | 진짜 | AI | 출처 + 생성기 |
|---|---|---|---|
| train | 5,000 | 5,000 | NYT + Midjourney |
| validation | 1,250 | 1,250 | NYT + Midjourney |
| test1_nyt_mj | 250 | 250 | NYT + Midjourney |
| test2_bbc_dalle | 250 | 250 | BBC + DALL-E 3 |
| test3_cnn_dalle | 250 | 250 | CNN + DALL-E 3 |
| test4_bbc_sdxl | 250 | 250 | BBC + SDXL |
| test5_cnn_sdxl | 250 | 250 | CNN + SDXL |

통합 시험 세트는 시험 split마다 네 경우가 125문항씩이에요 (5개 split × 500 = 2,500문항).

| | 글이 맞음 | 글이 안 맞음 |
|---|---|---|
| 진짜 사진 | 625 | 625 |
| AI 사진 | 625 | 625 |

- 사진과 글은 split 안에서 한 번씩만 나와요. 안 맞는 글은 같은 종류(진짜끼리, AI끼리) 안에서만 바꿨어요.
- AI 사진의 "맞는 글"은 그 사진을 만들 때 쓴 글이에요. 같은 사건으로 보이는 글(유사도 0.3 이상)은 안 맞음에 쓰지 않았어요.

## 4. 채점 예시

```python
import pandas as pd
from PIL import Image

df = pd.read_csv("data/ai_photos/ai_photos.csv")
test = df[df["split"].str.startswith("test") & (df["normalized_path"].notna())].copy()

test["pred"] = [my_detector(Image.open("data/ai_photos/" + p)) for p in test["normalized_path"]]  # 0 진짜 / 1 AI
test["ok"] = test["pred"] == test["label"]
print(test.groupby("split")["ok"].mean())  # split별 정확도
```

통합 시험 세트는 `is_ai`와 `is_mismatch`를 각각 채점하고, 둘 중 하나라도 1이면 "문제 있는 카드"로 보고 함께 채점하면 돼요.

## 5. 지킬 것

- 판정에는 사진(과 글)만 쓰고, 정답 칸은 마지막 채점에만 써요.
- 학습은 `train`, 기준값 정하기는 `validation`, 최종 평가는 `test*`로 해요.
- `data/` 폴더(사진)는 깃에 올리지 않아요. 원본 데이터셋에 라이선스 표시가 없어요.

## 6. 파일

| 파일 | 내용 |
|---|---|
| `prepare_ai_photos.py` | 사진·표 받기 (팀원이 실행) |
| `combined_test_pairs.csv`, `.meta.json` | 통합 시험 세트를 섞는 방법이 적힌 표 (행 번호와 정답) |
| `make_combined_test.py` | 통합 시험 세트 표를 새로 만들 때만 씀 (`pip install datasets scikit-learn numpy`) |
| `test_*.py` | 시험 코드 (`python -m pytest -q ai_photos`) |
