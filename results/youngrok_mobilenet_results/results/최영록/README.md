# 최영록 | MobileNetV2로 실제 사진과 AI 생성 사진 구분하기

MiRAGeNews의 원본 테스트 사진 2,500장에서 **1,740장(69.60%)**을 맞혔다. 실제 사진은 1,250장 중 1,006장, AI 생성 사진은 1,250장 중 734장을 맞혔다. 같은 사진의 가운데를 잘라 JPEG 형식을 통일한 뒤에는 정확도가 **61.56%**였다. 사진·글 짝 판정과 통합 시험에도 같은 사진 특징을 사용했다. 모든 판정은 CPU에서 실행했고 웹 검색이나 유료 API는 사용하지 않았다.

## 사진 한 장 판정하기

저장소 루트에서 실행한다. 첫 실행 때 공개 MobileNetV2 가중치(약 14 MB)를 받는다. 이 폴더의 학습된 분류기 [image_head.json](image_head.json)을 사용하므로 개별 사진을 검사할 때는 학습 데이터를 받을 필요가 없다.

```bash
python -m pip install numpy scipy pandas scikit-learn pillow onnx onnxruntime huggingface_hub
python "results/최영록/mobilenet_eval.py" --image "my_photo.jpg"
```

출력의 `is_ai`가 `true`면 AI 생성으로, `false`면 실제 사진으로 예측한 것이다. `ai_score`는 분류기의 AI 점수이고 `threshold`는 검증 데이터에서 정한 판정 기준이다.

## 사진 판정 결과

| 테스트 묶음 | 실제 / AI | 원본 정확도 | 중앙 크롭·JPEG 정확도 | 원본 ROC-AUC |
|---|---:|---:|---:|---:|
| NYT · Midjourney | 250 / 250 | 91.60% | 87.40% | 0.976 |
| BBC · DALL-E 3 | 250 / 250 | 65.00% | 57.00% | 0.708 |
| CNN · DALL-E 3 | 250 / 250 | 62.80% | 56.20% | 0.691 |
| BBC · SDXL | 250 / 250 | 62.40% | 53.60% | 0.691 |
| CNN · SDXL | 250 / 250 | 66.20% | 53.60% | 0.750 |
| **전체** | **1,250 / 1,250** | **69.60%** | **61.56%** | **0.774** |

NYT 기사 사진과 Midjourney 생성 사진으로 학습했다. 학습 때와 같은 출처·생성기의 테스트에서 정확도가 가장 높았다. 원본을 중앙 256×256, JPEG 품질 95로 통일했을 때 실제 사진을 AI로 잘못 판정한 수는 244장 → 178장으로 줄었지만, AI 사진을 실제로 잘못 판정한 수는 516장 → 783장으로 늘었다. 이 비교에서는 사진 속에 남는 영역과 JPEG 압축이 함께 바뀐다.

## 사진·글 짝 판정과 통합 시험

| 테스트 묶음 | 문항 수 | easy 정확도 | hard 정확도 |
|---|---:|---:|---:|
| NYT | 세트당 500 | 71.00% | 60.80% |
| BBC | 세트당 582 | 52.58% | 50.69% |
| CNN | 세트당 578 | 56.92% | 54.33% |
| **전체** | **세트당 1,660** | **59.64%** | **55.00%** |

easy는 무작위로 바꾼 글, hard는 비슷한 단어를 쓰는 다른 글을 포함한다. 두 세트의 맞는 짝은 동일하다. 실제 불일치 830건 중 easy에서 559건, hard에서 482건을 잡았다. BBC hard에서는 50.69%여서 균형 잡힌 두 범주를 구별하는 힘이 약했다.

통합 시험은 이미지 종류와 짝 일치 여부를 교차한 2,500문항이다. 짝 판정 정확도는 **58.76%**, 두 정답을 모두 맞힌 비율은 **40.84%(1,021/2,500)**였다. 촬영된 실제 사진에 다른 글이 붙은 경우에는 두 정답을 625건 중 343건 맞혔다. 여기서 짝 판정은 주어진 사진에 글이 어울리는지를 측정한다. 글에 들어 있는 역사적·정치적 주장의 사실 여부는 별도 과제다.

## 학습과 재현

1. **사진 분류**. ImageNet 학습 MobileNetV2의 가중치를 고정하고 마지막 전역 평균 풀링에서 1,280차원 특징을 얻었다. RGB 사진을 비율을 유지해 중앙 기준으로 224×224에 맞추고 ImageNet 평균·표준편차로 정규화했다. 학습 사진 10,000장으로 표준화와 로지스틱 회귀(`C=0.1`)를 적합했다.
2. **사진·글 짝**. 학습 세트의 원래 짝 4,997개에서 영어 캡션을 TF-IDF와 128차원 SVD로 표현했다. Ridge 회귀(`alpha=2`)로 사진 특징에서 캡션 벡터를 예측했다. 각 평가 문항에서는 예측한 벡터와 주어진 글 사이의 코사인 유사도로 불일치 점수 `(1 − cosine) / 2`를 계산했다.
3. **판정 기준**. 클래스별 정답률 평균이 가장 높은 임계값을 검증 세트에서 한 번 골랐다. 사진은 원본 검증 사진 2,500장, 짝은 easy·hard 검증 쌍 4,996개를 사용했다. 사진 기준은 약 0.53610, 짝 기준은 약 0.40907이다. 모든 테스트 묶음에 해당 기준을 그대로 적용했다.

전체 실험을 다시 실행하려면 저장소 루트에서 다음 명령을 사용한다. 재실행용 특징 캐시는 `data/youngrok_mobilenet/`에 저장된다.

```bash
python -m pip install pyarrow numpy scipy pandas scikit-learn pillow onnx onnxruntime huggingface_hub
python data_preparation/prepare_all.py
python "results/최영록/mobilenet_eval.py"
```

| 파일 | 기록한 내용 |
|---|---|
| [metrics.csv](metrics.csv) | 테스트 묶음별 정확도, ROC-AUC, 오탐률, 혼동행렬, 적용한 임계값 |
| [image_scores.csv](image_scores.csv) | 원본·중앙 크롭 사진별 AI 점수와 예측. `label=1`이면 AI |
| [match_scores.csv](match_scores.csv) | 사진·글 문항별 불일치 점수와 예측. `label=1`이면 불일치 |
| [combined_scores.csv](combined_scores.csv) | 통합 문항별 두 정답과 두 모델 출력 |
| [mobilenet_eval.py](mobilenet_eval.py) | 학습, 평가, 단일 사진 판정 코드 |

실행 환경은 Python 3.12, scikit-learn 1.8.0, ONNX Runtime 1.32.0이다. 데이터와 가중치는 팀 스크립트 및 실험 코드에 지정한 버전을 사용했다. [팀 데이터 구성](../../data_preparation/README.md) · [MiRAGeNews](https://huggingface.co/datasets/anson-huang/mirage-news) · [MobileNetV2 가중치](https://huggingface.co/onnxmodelzoo/mobilenetv2-12/tree/23a2a25eaaa6fcd61865414c8a1acf13dda66b7a)
