# 평가 결과 올리는 곳

각자 방법으로 짝 데이터(`easy.csv`, `hard.csv`)를 판정한 결과를 여기에 올려요.

## 올리는 방법

1. `_template` 폴더를 복사해서 `<이름>_<방법>`으로 이름을 바꿔요. 예: `results/name_clip/`
2. 아래 파일을 넣어요.

| 파일 | 내용 |
|---|---|
| `metrics.md` | 방법 설명과 지표 (양식 채우기) |
| `predictions_easy.csv` | easy 세트 test 쌍마다 판정 결과 |
| `predictions_hard.csv` | hard 세트 test 쌍마다 판정 결과 |

predictions 칸:

| 칸 | 뜻 |
|---|---|
| `pair_id` | `easy.csv`·`hard.csv`의 쌍 번호 |
| `pred` | 판정: 0 = 맞음, 1 = 안 맞음 |
| `score` | (있으면) 안 맞음일 가능성 0~1 |

## 지킬 것

- 결과는 test 묶음(`test_nyt`, `test_bbc`, `test_cnn`)으로 내요. `train`·`validation`은 학습과 기준값(맞음·안 맞음을 가르는 값) 정하기에만 써요.
- 사진이나 `data/` 폴더는 올리지 않아요. csv와 md만 올려요.
- 정확도는 다 같이 채우고, 다른 지표는 각자 자유롭게 추가해요.
