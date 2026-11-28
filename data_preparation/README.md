# 데이터 준비

팀에서 쓰는 데이터를 한 번에 받는 곳이에요.
깃에는 섞는 방법이 적힌 표와 스크립트만 있고, 사진은 실행하면 각자 컴퓨터에 받아져요.

## 받기

저장소 맨 위 폴더에서 실행해요.

```bash
pip install huggingface_hub pyarrow pillow
python data_preparation/prepare_all.py
```

- 처음 실행할 때 MiRAGeNews 원본 약 2.1GB를 받아요 (인터넷 속도에 따라 몇 분).
- 테스트용만 필요하면 `--test-only`를 붙여요 (원본 약 1.3GB).
- 다시 실행해도 이미 받은 사진은 다시 받지 않아요.

## 받으면 생기는 것

결과는 저장소 맨 위 `data/` 폴더에 생겨요. 각자 컴퓨터에만 있고 깃에는 올라가지 않아요.

| 데이터 | 위치 | 내용 | 자세히 |
|---|---|---|---|
| 글-사진 짝 | `data/mismatch/` | 진짜 사진 + 맞는 글/안 맞는 글 (easy·hard 세트당 14,152쌍) | [mismatch_pairs/README.md](mismatch_pairs/README.md) |
| AI 사진 | `data/ai_photos/` | 진짜·AI 사진 15,000장과 정답 | [ai_photos/README.md](ai_photos/README.md) |
| 통합 시험 | `data/ai_photos/combined_test.csv` | 진짜/AI 사진 × 맞는/안 맞는 글 2,500문항 | [ai_photos/README.md](ai_photos/README.md) |
| 교차 시험 | `data/ai_photos/cross_test.csv` | 같은 사건의 진짜/AI 사진 × 사실/거짓 글 5,000문항 | [ai_photos/README.md](ai_photos/README.md) |
| 카드 (실험용) | `data/cards/` | 통합·교차 시험 문항을 카드뉴스 이미지로 만든 것 6,250장 (따로 실행) | [cards/README.md](cards/README.md) |

## 결과 올리기

평가 결과는 저장소 맨 위 [`results/`](../results/README.md)의 각자 폴더에 올려요.

## 파일

| 파일·폴더 | 내용 |
|---|---|
| `prepare_all.py` | 데이터를 한 번에 받기 |
| `mismatch_pairs/` | 글-사진 짝 데이터의 표와 스크립트 |
| `ai_photos/` | AI 사진 데이터·통합 시험 세트·교차 시험 세트의 표와 스크립트 |
| `cards/` | 카드뉴스 이미지를 만드는 스크립트 (`python data_preparation/cards/make_cards.py`) |
| `test_prepare_all.py` | 코드 점검용 (`python -m pytest -q data_preparation`) |
