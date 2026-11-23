# 파이프라인 설계 문서 (초안)

> 카드뉴스 허위 정보 판별 및 AI 생성 이미지 탐지 · v0.1 (2026-11-23)<br/>
> 인터페이스나 형식을 바꿀 때는 이 문서를 먼저 수정하고 팀에 공유

## 1. 개요

| 항목 | 내용 |
|---|---|
| 입력 | 카드 1장 (카드뉴스 이미지) |
| 출력 | 내용 판정 (사실 / 허위 / 판단 불가 + 근거) · 이미지 판정 (AI 생성 확률) |
| 탐지 모델 | UnivFD (메인), ResNet-50 (베이스라인), NPR (비교) |
| 데이터 | MiRAGeNews 뉴스 사진 + 직접 생성한 카드 |
| 범위 밖 | 맥락 왜곡 (실제 사진 + 다른 사건 설명), 얼굴 합성 딥페이크 |

> **연구 질문** 사진이 카드뉴스로 편집될 때(축소·압축·텍스트 배치) AI 생성 이미지 탐지 성능이 얼마나 떨어지고, 사진 영역 크롭과 카드 데이터 추가 학습으로 얼마나 회복되는가?

| 용어 | 뜻 |
|---|---|
| 카드 | 카드뉴스 이미지 1장 (사진 + 제목·본문 텍스트·도형) |
| 카드 생성기 | 정답(실제/AI)을 아는 사진으로 실험용 카드를 자동 생성하는 프로그램 |
| GT 크롭 / YOLO 크롭 | 카드에서 사진만 잘라낸 것 (정답 좌표 기준 / YOLO 검출 기준) |
| AI 생성 확률 | 탐지 모델 출력값 (0~1), 클수록 AI 생성 가능성이 높음 |

## 2. 시스템 구조

색상: 🟦 ① 데이터·카드 생성 · 🟩 ② 사진 위치 검출·비교 모델 · 🟧 ③ 메인 탐지 모델 · 🟪 ④ 내용 검증·통합

### 2.1 실험 파이프라인 (모델 학습·평가)

```mermaid
flowchart LR
    A1["원본 사진<br/>실제 / AI"] --> A2["① 카드 생성기<br/>카드 + 메타데이터"]
    A2 --> A3["입력 구성<br/>카드 전체<br/>GT 크롭<br/>② YOLO 크롭"]
    A3 --> A4["③ UnivFD<br/>ResNet-50<br/>학습 · 평가"]
    A3 --> A5["② NPR<br/>학습 · 평가 (비교)"]
    A4 --> A6["결과<br/>표 · 그래프"]
    A5 --> A6

    classDef base fill:#F3F4F6,stroke:#6B7280,color:#111827
    classDef r1 fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A
    classDef r2 fill:#DCFCE7,stroke:#16A34A,color:#14532D
    classDef r3 fill:#FFEDD5,stroke:#EA580C,color:#7C2D12
    class A1,A3,A6 base
    class A2 r1
    class A5 r2
    class A4 r3
```

### 2.2 시연 파이프라인 (카드 1장 판정)

```mermaid
flowchart LR
    B1["카드 업로드"] --> B2["② 사진 위치 검출<br/>YOLO"]
    B2 --> B3["③ 이미지 판정<br/>UnivFD"]
    B1 --> B4["④ 텍스트 추출<br/>OCR · 주장 추출"]
    B4 --> B5["④ 내용 판정<br/>근거 검색 · 판정"]
    B3 --> B6["④ 결과 화면<br/>두 판정 함께 표시"]
    B5 --> B6

    classDef base fill:#F3F4F6,stroke:#6B7280,color:#111827
    classDef r2 fill:#DCFCE7,stroke:#16A34A,color:#14532D
    classDef r3 fill:#FFEDD5,stroke:#EA580C,color:#7C2D12
    classDef r4 fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95
    class B1 base
    class B2 r2
    class B3 r3
    class B4,B5,B6 r4
```

- 실험에서 학습한 모델을 시연에서 사용
- 실험용 카드는 정답이 필요해 직접 생성하고, 시연에는 실제 유통 카드를 사용
- 두 판정은 각각 수행해 함께 표시. 이미지 판정은 확률로 표시 (AI 생성 이미지 ≠ 허위 정보)

### 2.3 내용 판정 방식

| 구분 | 방식 |
|---|---|
| 트랙 1 | 근거 검색 후 LLM이 판정과 이유 생성 |
| 트랙 2 | 주장-근거 문장 쌍을 NLI 모델(KLUE-NLI 학습)로 분류 |
| 근거 출처 | KTV 카드뉴스 API, 정책브리핑 (정책·생활정보 카드뉴스 대상) |

> [!WARNING]
> 네이버 검색 API는 개정 약관상 AI 답변 생성 활용이 제한되어 사용하지 않음

## 3. 역할 및 인터페이스

| 구분 | 역할 | 주요 업무 | 출력 | 담당 |
|---|---|---|---|---|
| 🟦 ① | 데이터·카드 생성 | 데이터 준비·분할, 카드 생성기 | 카드, `meta.csv` | |
| 🟩 ② | 사진 위치 검출·비교 모델 | YOLO, NPR, 편집 강도 실험 | `detect_photos()` → 사진 위치 | |
| 🟧 ③ | 메인 탐지 모델 | UnivFD, ResNet-50, 공통 평가 코드 | `predict()` → AI 생성 확률 | |
| 🟪 ④ | 내용 검증·통합 | OCR·근거 검색·판정, 데모 앱 | `check_claims()` → 내용 판정 | |

- 번호 순서는 사진 판정 흐름과 같음 (① 카드 생성 → ② 사진 위치 검출 → ③ AI 판정). ④는 텍스트 판정과 결과 화면
- 공통 규격: 이미지는 RGB, 좌표는 픽셀 `(x1, y1, x2, y2)`, 확률은 0~1
- 발표 담당 1명은 별도이며, 발표 관련 사항은 추후 결정

## 4. 데이터

| 데이터 | 용도 | 규모 |
|---|---|---|
| MiRAGeNews | 원본 사진 (학습·검증·테스트) | 학습 10,000 · 검증 2,500 · 테스트 5 × 500 |
| B-Free 추가 데이터 | 미학습 생성기 테스트 | 실제 1,000 · FLUX 1,000 · SD3.5 1,000 |

- 실제/AI 사진의 형식·크기를 맞춘 뒤 카드 생성 (모델이 JPEG 여부·크기를 대신 학습하는 문제 방지)
- 원본 사진 기준으로 분할하고, 테스트용 템플릿은 학습에 사용하지 않음
- 카드마다 `meta.csv`에 정답·사진 위치·편집 강도 기록

> [!WARNING]
> 이미지 라이선스가 명시되어 있지 않아 데이터는 저장소에 올리지 않음

## 5. 실험 계획

| 실험 | 비교 내용 | 결과 |
|---|---|---|
| 1. 입력 방식 | 입력 (카드 전체·GT 크롭·YOLO 크롭) × 학습 조건 (공개 가중치·원본·원본+카드) | 표 |
| 2. 편집 강도 | JPEG 품질, 축소 비율, 텍스트 가림 비율을 하나씩 변경 | 그래프 |
| 3. 일반화 | 미학습 생성기 (DALL-E 3, SDXL, FLUX, SD3.5)·미학습 디자인 | 표 |

- 지표: AUC, 정확도, FPR (실제 사진을 AI로 잘못 판정한 비율)
- 추가 실험 (선택): GenD 방식 미세조정, GPT·Gemini 비교, 내용 판정 트랙 1·2 비교

> [!IMPORTANT]
> 임계값은 편집하지 않은 검증셋에서 한 번 정하고 모든 조건에 동일하게 적용

## 6. 진행 순서

```mermaid
flowchart LR
    S1["1. 규격 확정<br/>전원"] --> S2["2. 병렬 착수<br/>① 카드 일부 공유<br/>②③ 기준선 측정<br/>④ 검증 프로토타입"]
    S2 --> S3["3. 본 실험<br/>실험 1 → 2 → 3"]
    S3 --> S4["4. 통합<br/>데모 연동<br/>추가 실험"]

    classDef base fill:#F3F4F6,stroke:#6B7280,color:#111827
    class S1,S2,S3,S4 base
```

**결정 필요 사항**

- [ ] 담당자 배정
- [ ] 카드 문구(한글) 생성 방식, 템플릿 수, 카드 크기
- [ ] OCR 도구, 트랙 2 진행 여부
- [ ] GPU 환경, 데모 도구 (Streamlit / Gradio)

## 7. 참고 자료

- 탐지 모델: [UnivFD](https://github.com/WisconsinAIVision/UniversalFakeDetect) · [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) · [GenD](https://arxiv.org/abs/2508.06248)
- 데이터·실험 설계: [MiRAGeNews](https://huggingface.co/datasets/anson-huang/mirage-news) · [B-Free](https://grip-unina.github.io/B-Free) · [Fake or JPEG?](https://arxiv.org/abs/2403.17608)
- 기타: [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) · [KLUE](https://arxiv.org/abs/2105.09680) · [KTV 카드뉴스 API](https://www.data.go.kr/data/15150410/openapi.do)
