"""워크플로의 모델 실행 검증. 가중치 적재, 1장 임베딩, Qdrant 연결

이 환경에서 가중치 파일, torch, torchvision, transformers, Qdrant가 맞물리는지 확인.
적재 방식은 #14에서 확정한 Siglip2VisionModel + AutoImageProcessor bf16.
파이프라인 담당의 엔진 클래스가 머지되면 check_model을 build_engine(settings)로 교체

환경 변수
    MODEL_PATH   기본 models/siglip2. Docker 이미지 안에서는 /opt/models/siglip2
    QDRANT_URL   없으면 Qdrant 확인 생략

사용
    uv run scripts/runtime_check.py
    uv run scripts/runtime_check.py --skip-qdrant
"""

import argparse
import os
import sys
import time

DIM = 1152


def check_model(model_path: str) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, Siglip2VisionModel

    started = time.time()
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = Siglip2VisionModel.from_pretrained(model_path, dtype=torch.bfloat16).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"적재 {time.time() - started:.1f}s, {type(processor).__name__}, bf16 {params:.0f}M")

    image = Image.new("RGB", (1024, 768), (120, 160, 200))
    started = time.time()
    with torch.no_grad():
        inputs = processor(images=[image], return_tensors="pt")
        output = model(**inputs).pooler_output
    vector = output.float().numpy()
    if vector.shape != (1, DIM):
        raise RuntimeError(f"벡터 모양 {vector.shape}, 기대 (1, {DIM})")
    norm = float(np.linalg.norm(vector[0]))
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError(f"벡터 norm {norm}")
    print(f"임베딩 1장 {time.time() - started:.2f}s, shape {vector.shape}, norm {norm:.3f}")


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
    sys.exit(main())
