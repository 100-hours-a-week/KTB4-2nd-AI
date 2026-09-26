"""서버 없이 실제 파이프라인 실행.

uv run python -m scripts.run_pipeline request.json --image-dir data/photos
"""

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from app.core.config import Settings
    from app.pipeline.bootstrap import build_context, build_engine
    from app.pipeline.run import run
    from app.schemas.process import ProcessRequest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="ProcessRequest 형식 JSON")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("models/siglip2"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--qdrant-url", default=":memory:", help="기본값은 프로세스 내 임시 저장소")
    parser.add_argument("--thresholds", type=Path, default=Path("app/pipeline/thresholds.yaml"))
    parser.add_argument("--output", type=Path, help="생략하면 JSON을 표준 출력에 기록")
    args = parser.parse_args(argv)
    try:
        request = ProcessRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
        settings = Settings(
            _env_file=None,
            API_KEY="local-pipeline",
            IMAGE_SOURCE="local",
            IMAGE_DIR=args.image_dir,
            MODEL_PATH=args.model_path,
            EMBED_DEVICE=args.device,
            QDRANT_URL=args.qdrant_url,
            THRESHOLDS_PATH=args.thresholds,
            FAKE_PIPELINE=False,
        )
        engine = build_engine(settings)
        ctx = build_context(settings, engine)
        result = run(
            request,
            ctx,
            lambda step, done, total: print(f"{step.value} {done}/{total}", file=sys.stderr),
            lambda: False,
        )
        output = result.model_dump_json(indent=2)
        if args.output is not None:
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
    except Exception as exc:
        print(f"실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
