# 평가 결과

팀원마다 폴더를 하나씩 만들어, 각자 방법으로 평가한 결과를 올려요.

## 올리는 방법

1. `_template` 폴더를 복사해서 자기 이름(영문)으로 바꿔요. 예: `results/hong/`
2. `metrics.md` 양식을 채우고, 판정 결과 파일을 같은 폴더에 넣어요.

```
results/
├── _template/                복사해서 쓰는 양식
└── <이름>/
    ├── metrics.md            방법 설명과 점수
    ├── mismatch_easy.csv     짝 판정 결과 (easy)
    ├── mismatch_hard.csv     짝 판정 결과 (hard)
    ├── ai_photos.csv         (선택) AI 사진 판정 결과
    └── combined.csv          (선택) 통합 시험 결과
```

방법을 여러 개 시도했다면 `<이름>/` 안에 방법별 폴더를 하나 더 만들어도 돼요. 예: `results/hong/clip/`

## 판정 결과 파일 칸

| 파일 | 칸 |
|---|---|
| `mismatch_easy.csv`, `mismatch_hard.csv` | `pair_id`, `pred` (0 맞음 / 1 안 맞음), `score` (있으면, 안 맞음일 가능성 0~1) |
| `ai_photos.csv` | `image_id`, `pred` (0 진짜 / 1 AI), `score` (있으면, AI일 가능성 0~1) |
| `combined.csv` | `item_id`, `ai_pred` (0 진짜 / 1 AI), `mismatch_pred` (0 맞음 / 1 안 맞음) |

번호(`pair_id`, `image_id`, `item_id`)는 `data/` 폴더에 받은 표의 번호를 그대로 써요.

## 지킬 것

- 결과는 테스트 데이터로 내요. `train`·`validation`은 학습과 기준값(판정을 가르는 값) 정하기에만 써요.
- 사진이나 `data/` 폴더는 올리지 않아요. csv와 md만 올려요.
- 정확도는 다 같이 채우고, 다른 지표는 각자 자유롭게 추가해요.
