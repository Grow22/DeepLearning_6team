# 카드뉴스 이미지 (카드 편집 실험용)

섞어서 만든 시험 문항(교차 시험 + 통합 시험)을 실제 카드뉴스처럼 한 장의 이미지로 만들어요.
원래 문항과 사진·글·정답이 똑같고, 카드로 편집됐다는 것만 달라요. 그래서 각자 만든 방법을 카드에도 돌려 보고, 원래 문항의 결과와 바로 비교할 수 있어요.

## 만들기

`prepare_all.py`로 데이터를 먼저 받은 뒤, 저장소 맨 위 폴더에서 실행해요.
교차 시험이 생기기 전에 데이터를 받았다면 `prepare_all.py`를 한 번 더 실행해요 (사진은 다시 받지 않아요).

```bash
pip install pillow matplotlib
python data_preparation/cards/make_cards.py
```

- 카드 6,250장을 만들어요 (약 1GB, 몇 분).
- 결과는 `data/cards/`에 생기고, 깃에는 올라가지 않아요.

## 어떤 문항이 카드가 되나

| 시험 세트 | 문항 | 카드 |
|---|---|---|
| 교차 시험 | 5,000 | 전부 카드로 만들어요 |
| 통합 시험 | 2,500 | 글이 맞는 1,250문항은 교차 시험과 같은 문항이라 그 카드를 같이 써요. 글이 안 맞는 1,250문항은 새로 만들어요 |
| 합계 | | 6,250장 |

- 교차 시험의 카드 버전: `cross_item_id`가 있는 카드 5,000장
- 통합 시험의 카드 버전: `combined_item_id`가 있는 카드 2,500장

## 카드

| 항목 | 내용 |
|---|---|
| 크기·저장 | 1080×1350, JPEG 품질 85 (SNS 업로드 가정) |
| 템플릿 4종 | 사진 위·글 아래 / 사진 위에 글 덮기 / 제목 위·사진 가운데 / 글 많고 사진 작게 |
| 템플릿·색 | 같은 사건의 사진이 들어간 카드는 템플릿·색이 모두 같아요. 그래서 같은 글의 진짜·AI 카드는 사진만 다르고, 같은 사진의 카드끼리는 글만 달라요 |
| 균형 | 진짜 카드와 AI 카드의 템플릿·색 수가 split마다 똑같아요 |

**cards.csv**

| 칸 | 뜻 |
|---|---|
| `card_id`, `split`, `event` | 카드 번호, split, 사진의 사건 번호 |
| `image_id` | 사진 번호 (`ai_photos.csv`와 같음) |
| `cross_item_id`, `combined_item_id` | 원래 문항 번호 (`cross_test.csv`, `combined_test.csv`의 `item_id`). 그 시험에 없는 문항이면 빈칸 |
| `is_ai` | 사진이 AI인지: 0 = 진짜, 1 = AI |
| `is_mismatch` | 글이 사진과 안 맞는지: 0 = 맞음, 1 = 안 맞음 |
| `text_fake` | 글이 거짓인지: 0 = 사실(기사 글), 1 = 거짓(지어낸 글) |
| `template`, `palette` | 템플릿, 색 |
| `card_path` | 카드 위치 (`data/cards/` 기준) |
| `crop_path` | 잘라낸 사진 위치 (`--save-crops`를 붙였을 때만) |
| `photo_path` | 원본 사진 위치 (`data/ai_photos/` 기준) |
| `box_x1`~`box_y2` | 카드 안에서 사진이 있는 자리 |
| `photo_scale` | 원본 사진을 카드에 넣을 때 키우거나 줄인 비율 |
| `text_over_photo` | 사진 위에 글이 덮였는지 (1 = 덮임) |
| `text` | 카드에 쓴 글 (원래 문항의 글과 같음) |

## 세 가지 입력 만들기

```python
import pandas as pd
from PIL import Image

df = pd.read_csv("data/cards/cards.csv")
r = df.iloc[0]
original = Image.open("data/ai_photos/" + r.photo_path)                    # ① 원본 사진
card = Image.open("data/cards/" + r.card_path)                             # ② 카드 전체
crop = card.convert("RGB").crop((r.box_x1, r.box_y1, r.box_x2, r.box_y2))  # ③ 카드에서 잘라낸 사진
```

잘라낸 사진을 파일로도 저장하려면 `--save-crops`를 붙여요 (장당 약 1MB).

원래 문항 결과와 비교할 때는 `cross_item_id`나 `combined_item_id`로 이어 붙이면 돼요.

## 주의

- 카드에 글자가 들어 있어서, CLIP처럼 글자를 읽는 모델은 글을 보고 판단할 수도 있어요. AI 사진 판정을 비교할 때는 교차 시험 카드 5,000장을 쓰는 게 공정해요. 사진마다 사실 글 카드와 거짓 글 카드가 하나씩 있어서, 글로는 AI 사진을 맞힐 수 없어요. (통합 시험의 글이 안 맞는 카드는 AI 사진에 항상 지어낸 글이 붙어 있어요.)
- BBC·CNN은 진짜 사진이 AI 사진보다 작아서, 카드에 넣을 때 더 많이 키워져요 (중간값: 진짜 1.16~1.35배, AI 0.92~1.05배). NYT(test1)는 둘 다 2.1배로 같아서 가장 공정하게 비교할 수 있어요.
- 카드의 "NEWS CARD"는 가상의 이름이에요.
