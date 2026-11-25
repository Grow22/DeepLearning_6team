#!/usr/bin/env python3
"""데이터(글-사진 짝, AI 사진, 통합 시험 세트, 교차 시험 세트)를 한 번에 받는다.

실행 (저장소 맨 위 폴더에서):
    pip install huggingface_hub pyarrow pillow
    python data_preparation/prepare_all.py              # 전부
    python data_preparation/prepare_all.py --test-only  # 테스트용만

결과는 저장소 맨 위 data/ 폴더에 생긴다 (깃에 올라가지 않음).
    data/mismatch/    글-사진 짝 데이터
    data/ai_photos/   AI 사진 데이터 + 통합·교차 시험 세트
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mismatch_pairs"))
sys.path.insert(0, str(HERE / "ai_photos"))
import prepare_ai_photos  # noqa: E402
import prepare_mismatch_data  # noqa: E402

TEST_GROUPS = ["test_nyt", "test_bbc", "test_cnn"]
TEST_SPLITS = [s for s in prepare_ai_photos.SPLITS if s.startswith("test")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="데이터를 한 번에 받기")
    parser.add_argument("--test-only", action="store_true", help="테스트용만 받기 (원본 약 1.3GB)")
    args = parser.parse_args(argv)

    mismatch_args = ["--groups", *TEST_GROUPS] if args.test_only else []
    ai_args = ["--normalize"] + (["--splits", *TEST_SPLITS] if args.test_only else [])

    print("[1/2] 글-사진 짝 데이터", flush=True)
    prepare_mismatch_data.main(mismatch_args)
    print("\n[2/2] AI 사진 데이터 + 통합·교차 시험 세트", flush=True)
    prepare_ai_photos.main(ai_args)
    print("\n모두 완료: data/mismatch/, data/ai_photos/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
