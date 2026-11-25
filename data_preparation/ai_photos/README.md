# AI 사진 판정 데이터 + 통합·교차 시험 세트 (MiRAGeNews)

- **AI 사진 판정용**: MiRAGeNews의 진짜 사진과 AI 사진 15,000장을 정답(진짜/AI)과 함께 받아요.
- **통합 시험 세트**: 테스트 사진으로 만든 2,500문항이에요. 문항마다 사진(진짜/AI)과 글(맞음/안 맞음)이 섞여 있어서, AI 판정과 짝 판정을 한 번에 평가할 때 써요.
- **교차 시험 세트**: 같은 사건의 사진과 글을 바꿔 붙인 5,000문항이에요. "진짜 사진 + 거짓 글", "AI 사진 + 사실 글"까지 들어 있어서, 사진이 AI인지와 글이 거짓인지를 따로 볼 때 써요.

## 1. 데이터 받기

저장소 맨 위 폴더에서 실행해요.

```bash
pip install huggingface_hub pyarrow pillow
python data_preparation/prepare_all.py                               # 데이터 전부 한 번에
python data_preparation/ai_photos/prepare_ai_photos.py --normalize   # 이 데이터만
```

- 처음 실행할 때 MiRAGeNews 원본 약 2.1GB를 받아요. 글-사진 짝 데이터를 이미 받았다면 다시 받지 않아요.
- 일부 split만 받으려면 `--splits test1_nyt_mj test2_bbc_dalle`처럼 붙여요.

결과는 `data/ai_photos/`에 생겨요 (깃에 올라가지 않아요).

```
data/ai_photos/
├── images/<split>/<번호>.<확장자>   원본 사진 15,000장 (약 2.5GB)
├── normalized/<split>/<번호>.jpg    형식·크기를 맞춘 사진 13,802장 (256×256 JPEG)
├── ai_photos.csv                    AI 사진 판정용 표
├── combined_test.csv                통합 시험 세트 표
└── cross_test.csv                   교차 시험 세트 표
```

**ai_photos.csv**

| 칸 | 뜻 |
|---|---|
| `image_id` | 사진 번호 (`<split>-<행 번호>`) |
| `split` | `train` / `validation` / `test1_nyt_mj` / `test2_bbc_dalle` / `test3_cnn_dalle` / `test4_bbc_sdxl` / `test5_cnn_sdxl` |
| `label` | 정답: 0 = 진짜, 1 = AI |
| `source`, `generator` | 뉴스 출처, AI 사진을 만든 모델 (진짜 사진은 빈칸) |
| `image_path` | 원본 사진 위치 (`data/ai_photos/` 기준) |
| `format`, `width`, `height` | 원본 형식과 크기 |
| `normalized_path` | 형식·크기를 맞춘 사진 위치 (256px보다 작은 사진은 빈칸) |

**combined_test.csv**

| 칸 | 뜻 |
|---|---|
| `item_id`, `split` | 문항 번호, split |
| `image_path`, `normalized_path` | 사진 위치 (원본, 맞춘 사진) |
| `text` | 사진에 붙인 글 |
| `is_ai` | 정답 1: 0 = 진짜 사진, 1 = AI 사진 |
| `is_mismatch` | 정답 2: 0 = 글이 맞음, 1 = 글이 안 맞음 |

**cross_test.csv**

| 칸 | 뜻 |
|---|---|
| `item_id`, `split` | 문항 번호, split |
| `image_path`, `normalized_path` | 사진 위치 (원본, 맞춘 사진) |
| `text` | 사진에 붙인 글 (모두 같은 사건의 글) |
| `is_ai` | 정답 1: 0 = 진짜 사진, 1 = AI 사진 |
| `text_fake` | 정답 2: 0 = 기사 글(사실), 1 = 지어낸 글(거짓) |

## 2. 형식·크기를 맞춘 사진도 꼭 같이 보세요

BBC·CNN 테스트 세트는 진짜 사진과 AI 사진의 겉모습이 달라요. 원본 그대로 쓰면 사진 내용이 아니라 겉모습만 보고도 AI를 맞힐 수 있어요.

| 테스트 세트 | 진짜 사진 | AI 사진 |
|---|---|---|
| BBC (test2, test4) | 가로로 긴 JPEG, 왼쪽 아래에 BBC 로고가 있는 사진이 많음 | 정사각형 1024×1024 (DALL-E는 PNG) |
| CNN (test3, test5) | 가로로 긴 JPEG | 정사각형 1024×1024 (DALL-E는 PNG) |
| NYT (train, validation, test1) | JPEG | 진짜와 같은 크기의 JPEG (차이 없음) |

`--normalize`를 붙이면 모든 사진을 줄이거나 늘리지 않고 가운데 256×256만 잘라서, 같은 품질의 JPEG로 저장해요. 그러면 형식, 모양, 구석 로고의 차이가 사라져요. 결과는 원본 사진과 맞춘 사진 두 가지로 같이 내 주세요.

256px보다 작은 사진(train 1,042장, validation 156장)은 맞춘 사진에서 빠져요. 진짜와 AI가 같은 수만큼 빠져서 반반은 그대로예요.

## 3. 구성

| split | 진짜 | AI | 출처 + AI 사진을 만든 모델 |
|---|---|---|---|
| train | 5,000 | 5,000 | NYT + Midjourney |
| validation | 1,250 | 1,250 | NYT + Midjourney |
| test1_nyt_mj | 250 | 250 | NYT + Midjourney |
| test2_bbc_dalle | 250 | 250 | BBC + DALL-E 3 |
| test3_cnn_dalle | 250 | 250 | CNN + DALL-E 3 |
| test4_bbc_sdxl | 250 | 250 | BBC + SDXL |
| test5_cnn_sdxl | 250 | 250 | CNN + SDXL |

## 4. 채점하기

### AI 사진 판정

```python
import pandas as pd
from PIL import Image

df = pd.read_csv("data/ai_photos/ai_photos.csv")
test = df[df["split"].str.startswith("test") & (df["normalized_path"].notna())].copy()

test["pred"] = [my_detector(Image.open("data/ai_photos/" + p)) for p in test["normalized_path"]]  # 0 진짜 / 1 AI
test["ok"] = test["pred"] == test["label"]
print(test.groupby("split")["ok"].mean())  # split별 정확도
```

### 통합 시험 세트

문항 하나가 카드 한 장처럼 사진과 글로 되어 있어요. 같은 문항에 AI 판정과 짝 판정을 둘 다 돌려요.

| 경우 | 사진 | 글 | `is_ai` | `is_mismatch` | 종합 정답 | 문항 수 |
|---|---|---|---|---|---|---|
| 1 | 진짜 | 맞음 | 0 | 0 | 문제 없음 | 625 |
| 2 | 진짜 | 안 맞음 | 0 | 1 | 문제 있음 | 625 |
| 3 | AI | 맞음 | 1 | 0 | 문제 있음 | 625 |
| 4 | AI | 안 맞음 | 1 | 1 | 문제 있음 | 625 |

- 테스트 split마다 네 경우가 125문항씩이에요. 사진과 글은 split 안에서 한 번씩만 나와요.
- AI 사진의 "맞는 글"은 그 사진을 만들 때 쓴 글이에요. 안 맞는 글은 진짜는 진짜끼리, AI는 AI끼리 바꿨어요.

채점은 세 가지로 해요.

1. **AI 판정**: 사진 판정 결과를 `is_ai`와 비교
2. **짝 판정**: 글-사진 판정 결과를 `is_mismatch`와 비교
3. **종합**: 둘 중 하나라도 걸리면 "문제 있음"으로 보고 종합 정답과 비교

```python
df = pd.read_csv("data/ai_photos/combined_test.csv")
df["ai_pred"] = ...        # 각자 AI 판정 (0 진짜 / 1 AI)
df["mismatch_pred"] = ...  # 각자 짝 판정 (0 맞음 / 1 안 맞음)

df["problem"] = df["is_ai"] | df["is_mismatch"]
df["problem_pred"] = df["ai_pred"] | df["mismatch_pred"]
df["case"] = df["is_ai"].map({0: "진짜", 1: "AI"}) + " + 글 " + df["is_mismatch"].map({0: "맞음", 1: "안 맞음"})

print("AI 판정 정확도:", (df["ai_pred"] == df["is_ai"]).mean())
print("짝 판정 정확도:", (df["mismatch_pred"] == df["is_mismatch"]).mean())
print((df["problem_pred"] == df["problem"]).groupby(df["case"]).mean())  # 경우별 종합 정확도
```

> **주의** 4문항 중 3문항이 "문제 있음"이라서, 전부 "문제 있음"이라고 찍어도 종합 정확도가 75%예요. 종합 점수만 보지 말고 **경우별 정확도**를 같이 보세요. 특히 "진짜 + 글 맞음"을 문제 없다고 맞히는지가 중요해요.

> **주의** 이 세트에서는 진짜 사진에 항상 기사 글(사실), AI 사진에 항상 지어낸 글(거짓)이 붙어 있어요. 그래서 글만 보는 간단한 모델도 AI 사진을 82.6% 맞혀요. 사진 종류와 글의 사실 여부를 떼어서 보려면 아래 교차 시험 세트를 쓰세요.

### 교차 시험 세트

MiRAGeNews의 지어낸 글은 GPT-4가 기사 글을 일부러 오해를 부르게 바꿔 쓴 글이고, AI 사진은 그 지어낸 글로 만든 사진이에요. 같은 사건의 조각을 바꿔 붙여서 네 칸을 다 채웠어요.

| | 기사 글 (사실) | 지어낸 글 (거짓) |
|---|---|---|
| 진짜 사진 | 원래 짝 | **바꿔 붙임** |
| AI 사진 | **바꿔 붙임** | 원래 짝 |

- 예: 아테네 시위 사진에 "시위대와 경찰이 충돌했다"(사실) 또는 "경찰이 평화 시위를 잔혹하게 진압했다"(거짓)를 붙여요.
- 테스트 split마다 250사건 × 4칸 = 1,000문항, 모두 5,000문항이에요.
- 사진 하나가 사실 글·거짓 글과 한 번씩, 글 하나가 진짜 사진·AI 사진과 한 번씩 나와요. 그래서 사진 정답으로 글 정답을 맞히면 정확히 50%예요.
- 글은 모두 같은 사건의 것이라 사진과 맞는 글이에요. 짝 판정을 돌리면 정답은 전부 "맞음"이에요.

채점은 `is_ai`(사진이 AI인가)와 `text_fake`(글이 거짓인가)를 칸별로 봐요.

```python
df = pd.read_csv("data/ai_photos/cross_test.csv")
df["ai_pred"] = ...  # 각자 AI 판정 (0 진짜 / 1 AI)
df["case"] = df["is_ai"].map({0: "진짜 사진", 1: "AI 사진"}) + " + " + df["text_fake"].map({0: "사실 글", 1: "거짓 글"})

print((df["ai_pred"] == df["is_ai"]).groupby(df["case"]).mean())  # 칸별 AI 판정 정확도
# 글이 거짓인지도 판정했다면 같은 방식으로 text_fake와 비교
```

> **참고** 이 데이터에는 사실 확인에 쓸 근거 자료가 들어 있지 않아요. 근거 없이 글만 보는 방법은 결국 말투로 판단하게 돼요(글만 보는 간단한 모델로 약 83%). 비교할 때 이 점을 같이 적어 주세요.

결과는 저장소 맨 위 `results/<이름>/` 폴더에 올려요. 올리는 방법은 [`results/README.md`](../../results/README.md)에 있어요.

## 5. 지킬 것

- 판정에는 사진과 글만 쓰고, 정답 칸은 마지막 채점에만 써요.
- 교차 시험 세트는 같은 사진과 같은 글이 두 번씩 나와요. 문항마다 따로 판정하고, 문항끼리 비교하지 않아요.
- `train`은 학습, `validation`은 기준값(진짜·AI를 가르는 값) 정하기, `test*`는 최종 평가에 써요.
- `data/` 폴더(사진)는 깃에 올리지 않아요. 원본 데이터셋에 라이선스 표시가 없어요.

## 6. 파일

| 파일 | 내용 |
|---|---|
| `prepare_ai_photos.py` | 사진·표 받기 (팀원이 실행) |
| `combined_test_pairs.csv`, `.meta.json` | 통합 시험 세트를 섞는 방법이 적힌 표 (행 번호와 정답) |
| `make_combined_test.py` | 통합 시험 세트 표를 새로 만들 때만 씀 (`pip install datasets scikit-learn numpy`) |
| `cross_test_pairs.csv`, `.meta.json` | 교차 시험 세트를 섞는 방법이 적힌 표 (행 번호와 정답) |
| `make_cross_test.py` | 교차 시험 세트 표를 새로 만들 때만 씀 (`pip install datasets scikit-learn numpy`) |
| `test_*.py` | 코드 점검용 (`python -m pytest -q data_preparation`) |
