# 글-사진 짝 데이터 (MiRAGeNews)

MiRAGeNews의 진짜 사진과 글을 섞어 만든 **"글이 사진과 맞는지" 판정용 데이터**예요.
깃에는 섞는 방법이 적힌 표만 있고, 사진과 글은 스크립트가 MiRAGeNews에서 받아 표대로 섞어 줘요.
누가 실행해도 똑같이 섞인 데이터가 나와요.

## 1. 데이터 받기

저장소 맨 위 폴더에서 실행해요.

```bash
pip install huggingface_hub pyarrow pillow
python mismatch_pairs/prepare_mismatch_data.py
```

- 처음 실행할 때 MiRAGeNews 원본 약 2.1GB를 받아요 (인터넷 속도에 따라 몇 분).
- 시험용만 필요하면 `--groups test_nyt test_bbc test_cnn`을 붙여요 (원본 약 1.3GB).
- 다시 실행해도 이미 받은 사진은 그대로 둬요.

결과는 `data/mismatch/`에 생겨요. `data/` 폴더는 깃에 올라가지 않아요.

```
data/mismatch/
├── images/<split>/<번호>.jpg   진짜 사진 7,076장 (557MB)
├── easy.csv                    안 맞는 글을 무작위로 고른 세트
└── hard.csv                    안 맞는 글을 내용이 비슷한 것으로 고른 세트
```

| csv 칸 | 뜻 |
|---|---|
| `pair_id` | 쌍 번호 |
| `group` | `train` / `validation` / `test_nyt` / `test_bbc` / `test_cnn` |
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

## 3. 지킬 것

- 판정에는 사진과 글만 쓰고, `label`은 마지막 채점에만 써요.
- 쌍마다 따로 판정해요. 같은 사진이 맞음 1번, 안 맞음 1번 나오니 쌍끼리 비교하지 않아요.
- 학습은 `train`, 기준값 정하기는 `validation`, 최종 평가는 `test_*` 묶음으로 해요.
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

- 진짜 쌍(MiRAGeNews label 0)만 써요. 사진마다 원래 글을 붙인 "맞음" 1쌍과, 같은 묶음의 다른 사진 글을 붙인 "안 맞음" 1쌍을 만들어요.
- easy는 무작위 글, hard는 내용이 비슷한 글(TF-IDF 유사도)을 붙여요. 같은 사건으로 보이는 글(유사도 0.3 이상)은 빼요.
- 글은 같은 묶음 안에서만 바꿔요. BBC·CNN 사진은 두 시험 세트에 겹쳐 있어서 한 번만 썼어요.
- hard에는 "같은 인물·다른 사건" 쌍이 많아서, 사람이 봐도 헷갈리는 쌍이 일부 있어요.

## 6. 파일

| 파일 | 내용 |
|---|---|
| `prepare_mismatch_data.py` | 사진·글을 받아 표대로 섞기 (팀원이 실행) |
| `mismatch_pairs.csv` | 섞는 방법이 적힌 표 (행 번호와 정답) |
| `mismatch_pairs.meta.json` | 데이터 버전, 시드, 개수 |
| `make_mismatch_pairs.py` | 표를 새로 만들 때만 씀 (`pip install datasets scikit-learn numpy`) |
| `test_*.py` | 시험 코드 (`python -m pytest -q mismatch_pairs`) |
