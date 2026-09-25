"""워크플로의 모델 실행 검증. 가중치 적재, 1장 임베딩, Qdrant 연결

이 환경에서 가중치 파일, torch, torchvision, transformers, Qdrant가 맞물리는지 확인.
서버와 같은 build_engine으로 실제 엔진을 만들고 반환 계약까지 확인.
인증과 S3가 필요 없는 독립 점검이며 FAKE_PIPELINE 설정과 무관하게 실제 모델을 사용.

환경 변수
    MODEL_PATH   기본 models/siglip2. Docker 이미지 안에서는 /opt/models/siglip2
    EMBED_DEVICE 기본 cpu
    QDRANT_URL   없으면 Qdrant 확인 생략

사용
    uv run scripts/runtime_check.py
    uv run scripts/runtime_check.py --skip-qdrant
"""

import argparse
import os
import sys
import time
from pathlib import Path

DIM = 1152


def check_model(model_path: str) -> None:
    import numpy as np
    from PIL import Image

    from app.core.config import Settings
    from app.pipeline.bootstrap import build_engine

    settings = Settings(
        _env_file=None,
        API_KEY="runtime-check",
        IMAGE_SOURCE="local",
        FAKE_PIPELINE=False,
        MODEL_PATH=Path(model_path),
        EMBED_DEVICE=os.environ.get("EMBED_DEVICE", "cpu"),
    )
    started = time.perf_counter()
    engine = build_engine(settings)
    print(f"적재 {time.perf_counter() - started:.1f}s, {type(engine).__name__}")

    image = Image.new("RGB", (1024, 768), (120, 160, 200))
    started = time.perf_counter()
    vector = engine.encode_images([image])
    if vector.shape != (1, DIM):
        raise RuntimeError(f"벡터 모양 {vector.shape}, 기대 (1, {DIM})")
    if vector.dtype != np.float32:
        raise RuntimeError(f"벡터 dtype {vector.dtype}, 기대 float32")
    if not np.isfinite(vector).all():
        raise RuntimeError("벡터에 유한하지 않은 값이 있음")
    norm = float(np.linalg.norm(vector[0]))
    if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"벡터 norm {norm}, 기대 1.0")
    print(f"임베딩 1장 {time.perf_counter() - started:.2f}s, shape {vector.shape}, norm {norm:.3f}")


def check_qdrant(url: str) -> None:
    import httpx

    response = httpx.get(f"{url}/readyz", timeout=5.0)
    if response.status_code != 200:
        raise RuntimeError(f"Qdrant /readyz {response.status_code}")
    print(f"Qdrant {url} ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-qdrant", action="store_true", help="Qdrant 확인 생략")
    args = parser.parse_args(argv)

    model_path = os.environ.get("MODEL_PATH", "models/siglip2")
    qdrant_url = os.environ.get("QDRANT_URL")

    try:
        check_model(model_path)
        if qdrant_url and not args.skip_qdrant:
            check_qdrant(qdrant_url)
        elif not args.skip_qdrant:
            print("QDRANT_URL 없음, Qdrant 확인 생략")
    except Exception as e:  # noqa: BLE001, 워크플로가 보는 건 exit code와 한 줄 메시지
        print(f"실패: {e}", file=sys.stderr)
        return 1
    print("통과")
    return 0


if __name__ == "__main__":
    # 파일 경로로 실행해도 저장소의 app 패키지를 찾을 수 있게 한다.
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
