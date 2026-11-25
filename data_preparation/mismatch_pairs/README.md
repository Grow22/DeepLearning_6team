# 글-사진 짝 데이터 (MiRAGeNews)

MiRAGeNews의 진짜 사진과 글을 섞어 만든 **"글이 사진과 맞는지" 판정용 데이터**예요.
깃에는 섞는 방법이 적힌 표만 있어요. 스크립트를 실행하면 MiRAGeNews에서 사진과 글을 받아 표대로 섞어 줘요. 누가 실행해도 똑같은 데이터가 나와요.

## 1. 데이터 받기

저장소 맨 위 폴더에서 실행해요.

```bash
pip install huggingface_hub pyarrow pillow
python data_preparation/prepare_all.py                          # 데이터 전부 한 번에
python data_preparation/mismatch_pairs/prepare_mismatch_data.py  # 이 데이터만
```

- 처음 실행할 때 MiRAGeNews 원본 약 2.1GB를 받아요 (인터넷 속도에 따라 몇 분).
- 테스트용만 필요하면 `--groups test_nyt test_bbc test_cnn`을 붙여요 (원본 약 1.3GB).
- 다시 실행해도 이미 받은 사진은 다시 받지 않아요.

결과는 `data/mismatch/`에 생겨요. `data/` 폴더는 깃에 올라가지 않아요.

```
data/mismatch/
├── images/<split>/<번호>.jpg   진짜 사진 7,076장 (557MB)
├── easy.csv                    안 맞는 글을 아무거나 고른 세트
└── hard.csv                    안 맞는 글을 비슷한 내용으로 고른 세트
```

| csv 칸 | 뜻 |
|---|---|
| `pair_id` | 쌍 번호 |
| `group` | 묶음: `train` / `validation` / `test_nyt` / `test_bbc` / `test_cnn` |
| `image_path` | 사진 위치 (`data/mismatch/` 기준) |
| `text` | 사진에 붙인 글 |
| `label` | 정답: 0 = 맞음, 1 = 안 맞음 |

## 2. 채점하기

```python
import pandas as pd
from PIL import Image

df = pd.read_csv("data/mismatch/easy.csv")
test = df[df["group"].str.startswith("test")].copy()

preds = []
for r in test.itertuples():
    image = Image.open("data/mismatch/" + r.image_path)
    preds.append(my_method(image, r.text))  # 각자 방법: 0 = 맞음, 1 = 안 맞음

test["pred"] = preds
test["ok"] = test["pred"] == test["label"]
print("정확도:", test["ok"].mean())
print(test.groupby("group")["ok"].mean())  # 묶음별 정확도
```

결과는 저장소 맨 위 `results/<이름>/` 폴더에 올려요. 올리는 방법은 [`results/README.md`](../../results/README.md)에 있어요.

## 3. 지킬 것

- 판정에는 사진과 글만 쓰고, `label`은 마지막 채점에만 써요.
- 쌍마다 따로 판정해요. 같은 사진이 맞음과 안 맞음에 한 번씩 나오니 쌍끼리 비교하지 않아요.
- `train`은 학습, `validation`은 기준값(맞음·안 맞음을 가르는 값) 정하기, `test_*`는 최종 평가에 써요.
- `data/` 폴더(사진)는 깃에 올리지 않아요. 원본 데이터셋에 라이선스 표시가 없어요.

## 4. 구성

| 묶음 | 출처 | 진짜 사진 | 세트당 쌍 (맞음 + 안 맞음) |
|---|---|---|---|
| train | NYT | 4,997 | 9,994 |
| validation | NYT | 1,249 | 2,498 |
| test_nyt | NYT (학습과 같은 출처) | 250 | 500 |
| test_bbc | BBC | 291 | 582 |
| test_cnn | CNN | 289 | 578 |
| **합계** | | **7,076** | **14,152** |

## 5. 어떻게 섞었나

- 진짜 사진만 써요. 사진마다 원래 글을 붙인 "맞음" 1쌍과, 같은 묶음의 다른 사진 글을 붙인 "안 맞음" 1쌍을 만들어요.
- 그래서 같은 사진과 같은 글이 맞음과 안 맞음에 한 번씩 나와요. 한쪽만 보고는 맞힐 수 없고(한쪽으로 찍으면 50%), 사진과 글을 비교해야 점수가 올라가요.
- AI 사진은 넣지 않았어요. 섞으면 "AI 사진"을 잡은 건지 "안 맞는 글"을 잡은 건지 알 수 없어요. AI 사진까지 섞은 테스트는 `ai_photos/`의 통합 시험 세트에 있어요.
- easy는 아무 글이나, hard는 겹치는 단어가 많은 비슷한 글을 붙여요. 너무 비슷해서 같은 사건으로 보이는 글은 쓰지 않아요.
- hard에는 "같은 인물, 다른 사건" 쌍이 많아서 사람이 봐도 헷갈리는 쌍이 조금 있어요.
- 글은 같은 묶음 안에서만 바꿔요. BBC·CNN 진짜 사진은 MiRAGeNews 테스트 세트 두 곳에 겹쳐 있어서 한 번만 썼어요.

## 6. 파일

| 파일 | 내용 |
|---|---|
| `prepare_mismatch_data.py` | 사진·글을 받아 표대로 섞기 (팀원이 실행) |
| `mismatch_pairs.csv` | 섞는 방법이 적힌 표 (행 번호와 정답) |
| `mismatch_pairs.meta.json` | 데이터 버전과 개수 |
| `make_mismatch_pairs.py` | 표를 새로 만들 때만 씀 (`pip install datasets scikit-learn numpy`) |
| `test_*.py` | 코드 점검용 (`python -m pytest -q data_preparation`) |
