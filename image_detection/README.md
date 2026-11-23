# 🖼️ 사진 AI 생성 징후 탐지 (담당: 유민)

카드뉴스 속 사진이 AI로 생성됐을 확률을 판정하는 파트. 내용 사실 여부(RAG)와는 **따로** 보여준다. "AI 생성 = 거짓"이 아니다.

## 폴더 구조

```
image_detection/
├── data/
│   ├── raw/
│   │   ├── real/      진짜 사진 (Wikimedia Commons, 원본 그대로)
│   │   └── ai/        AI 생성 이미지 (ChatGPT, Gemini, 원본 포맷 그대로)
│   └── meta.csv       이미지별 정보 (라벨, 카테고리, 출처, 라이선스, split)
├── prompts/
│   └── ai_prompts.md  AI 이미지 생성 프롬프트와 규칙
└── scripts/
    └── build_pilot_dataset.py  진짜 사진 다운로드 + meta.csv 생성
```

## 현재 상태 (파일럿)

| | crowd | street | speech | vote | city | 합계 |
|---|---|---|---|---|---|---|
| real (Wikimedia) | 10 | 10 | 10 | 10 | 10 | 50 |
| ai / gemini | 5 | 5 | 5 | 5 | 5 | 25 |
| ai / chatgpt | 5 | – | – | – | – | 5 (진행 중) |

카테고리: crowd 유세·집회 군중 / street 거리·현수막 / speech 연설·행사장 / vote 투표소 / city 일반 도시 풍경

## meta.csv 컬럼

| 컬럼 | 설명 |
|---|---|
| `img_id` | 파일 이름(확장자 제외) |
| `source_id` | 원본 ID. 카드 편집본은 원본과 같은 값을 갖고, split은 이 값 기준으로 나눈다 |
| `label` | `real` / `ai` |
| `generator` | AI 생성기 (`chatgpt`, `gemini`), 진짜 사진은 빈칸 |
| `category` | 장면 카테고리 |
| `file` | `image_detection/` 기준 상대 경로 |
| `orig_format`, `width`, `height` | 원본 포맷과 크기 |
| `face_closeup` | 실존 인물 얼굴이 크게 나오는지 (`yes`/`no`) |
| `origin_url`, `license`, `author` | 출처 페이지, 라이선스, 작가 |
| `note` | 참고 사항 (예: Gemini 워터마크) |
| `split` | train/val/test. Step 2에서 채운다 |

## 주의 사항

- **원본은 포맷이 제각각이다** (진짜 JPEG / ChatGPT PNG / Gemini WEBP). 학습 전에 반드시 Step 2 전처리(짧은 변 512px → JPEG q95 → EXIF 제거)를 거친다. 안 그러면 모델이 AI 흔적 대신 파일 포맷을 외운다 (Fake or JPEG?, arXiv 2403.17608).
- **Gemini 이미지는 오른쪽 아래에 반짝이 워터마크가 있다.** 전처리에서 잘라낸다.
- **진짜 사진 라이선스**: 대부분 CC BY / CC BY-SA, 일부 KOGL 1유형, Public domain. 발표 자료 등에 쓸 때는 `author`와 `origin_url`로 출처를 표시한다.
- **AI 이미지에는 실존 정치인, 실제 정당 이름·로고를 넣지 않는다.** 모든 인물과 정당은 가상이다. 이 데이터는 연구용이며, AI 이미지를 단독으로 캡처해서 퍼뜨리지 않는다.
- 진짜 사진은 실제 선거·집회 장면이라 실제 정치인 얼굴이나 정당 현수막이 나올 수 있다 (`face_closeup` 참고).

## 진짜 사진 다시 받기

```bash
python scripts/build_pilot_dataset.py <all.json> <ai_images 폴더> .
```
`all.json`(= `scripts/wikimedia_candidates.json`)은 Wikimedia Commons 후보 목록(카테고리 검색 결과)이다. 고른 사진 번호는 스크립트의 `PICKS`에 있다.
