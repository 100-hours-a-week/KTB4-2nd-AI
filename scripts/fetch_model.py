"""임베딩 모델 가중치 준비. Docker 빌드 전과 로컬 개발에서 한 번 실행

HF 저장소를 커밋 해시로 고정해 models/.raw/에 원본을 받고, bf16으로 변환한 결과를
models/siglip2에 씀.
가중치는 git에 없고(.gitignore의 models/) Dockerfile이 COPY models/siglip2로 이미지에 넣음.
워크플로는 --cache-key 출력을 캐시 키로 써서 같은 revision과 변환 방식이면 다시 만들지 않음

bf16인 이유: 파이프라인 담당의 실측(200장 373초)과 임계값이 bf16 기준. 이미지에 든 가중치가
잰 것과 같아야 함. fp32 4.5GB가 2.3GB로 줄어 워커 RAM도 그만큼 줄어듦

사용
    uv run scripts/fetch_model.py                    models/siglip2에 bf16 전체 모델
    uv run scripts/fetch_model.py --vision-only      text 타워 제외, 0.9GB
                                                     엔진이 Siglip2VisionModel로 적재할 때만
    uv run scripts/fetch_model.py --dtype fp32       변환 없이 원본 그대로
    uv run scripts/fetch_model.py --cache-key        캐시 키만 출력하고 종료
    uv run scripts/fetch_model.py --force            결과가 있어도 다시 만듦
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Settings.EMBED_MODEL 값 → HF 저장소와 커밋 해시
# revision을 바꾸면 캐시 키가 바뀌어 워크플로가 다시 받음
MODELS: dict[str, dict[str, str]] = {
    "siglip2-so400m-naflex": {
        "repo_id": "google/siglip2-so400m-patch16-naflex",
        "revision": "cc24074f717b612951c2dead130904ab9b65a81e",  # 2026-09-21 main, fp32 4.5GB
        "dest": "models/siglip2",
    },
}

RAW_ROOT = Path("models/.raw")  # HF 원본, Dockerfile COPY 대상이 아님

# 이미지 임베딩에 필요한 파일만
# 토크나이저는 v1이 텍스트를 안 쓰지만 AutoProcessor 적재에 필요해 포함
ALLOW_PATTERNS = ["*.json", "*.safetensors", "tokenizer.model"]
WEIGHTS = "model.safetensors"

# 결과 폴더의 완료 표시. 내용이 variant()와 같으면 아무것도 안 함
MARKER = ".revision"

DTYPES = {"fp32": "float32", "bf16": "bfloat16"}


def variant(name: str, dtype: str, vision_only: bool) -> str:
    """가중치 파일이 무엇으로 만들어졌는지 한 줄. revision, dtype, 타워 범위"""
    scope = "vision" if vision_only else "full"
    return f"{MODELS[name]['revision']} {dtype} {scope}"


def cache_key(name: str, dtype: str, vision_only: bool) -> str:
    """워크플로 캐시 키. 모델 이름, revision 앞 12자, dtype, vision-only면 -vision"""
    key = f"{name}-{MODELS[name]['revision'][:12]}-{dtype}"
    return key + "-vision" if vision_only else key


def download_raw(name: str) -> Path:
    """HF 원본을 models/.raw/<이름>에. 이미 받은 파일은 huggingface_hub가 건너뜀"""
    spec = MODELS[name]
    raw = RAW_ROOT / name
    print(f"원본 {spec['repo_id']} @{spec['revision'][:12]} → {raw}")
    snapshot_download(
        repo_id=spec["repo_id"],
        revision=spec["revision"],
        local_dir=raw,
        allow_patterns=ALLOW_PATTERNS,
    )
    return raw


def convert_weights(src: Path, dst: Path, dtype: str, vision_only: bool) -> int:
    """safetensors를 텐서 하나씩 읽어 dtype으로 바꿔 저장

    전체를 한 번에 올리지 않아 피크 메모리가 결과 크기 정도.

    vision_only면 vision_model.* 텐서만 남김. 이 파일은 Siglip2VisionModel로만 적재 가능
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    target = getattr(torch, DTYPES[dtype])
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(src), framework="pt") as f:
        for key in f.keys():
            if vision_only and not key.startswith("vision_model."):
                continue
            tensor = f.get_tensor(key)
            tensors[key] = tensor.to(target) if tensor.is_floating_point() else tensor
    save_file(tensors, str(dst), metadata={"format": "pt"})
    return len(tensors)


def build(name: str, dest: Path, dtype: str, vision_only: bool, force: bool = False) -> Path:
    marker = dest / MARKER
    want = variant(name, dtype, vision_only)
    if not force and marker.exists() and marker.read_text().strip() == want:
        print(f"이미 있음 {dest} ({want})")
        return dest

    raw = download_raw(name)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for path in raw.iterdir():
        if path.name != WEIGHTS and not path.name.startswith("."):
            shutil.copy2(path, dest / path.name)

    if dtype == "fp32" and not vision_only:
        shutil.copy2(raw / WEIGHTS, dest / WEIGHTS)
    else:
        print(f"변환 {dtype}, {'vision 타워만' if vision_only else '전체'}")
        count = convert_weights(raw / WEIGHTS, dest / WEIGHTS, dtype, vision_only)
        print(f"텐서 {count}개")

    marker.write_text(want + "\n")
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"완료 {dest} {size / 1e9:.2f}GB ({want})")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EMBED_MODEL", "siglip2-so400m-naflex"),
        choices=sorted(MODELS),
        help="Settings.EMBED_MODEL 값, 기본은 환경 변수 EMBED_MODEL 또는 siglip2-so400m-naflex",
    )
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPES), help="기본 bf16")
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="text 타워 제외. 엔진이 vision 클래스로 적재할 때만",
    )
    parser.add_argument("--dest", type=Path, help="결과 경로, 기본은 모델별 models/ 아래")
    parser.add_argument("--force", action="store_true", help="결과가 있어도 다시 만듦")
    parser.add_argument("--cache-key", action="store_true", help="캐시 키만 출력하고 종료")
    args = parser.parse_args(argv)

    if args.cache_key:
        print(cache_key(args.model, args.dtype, args.vision_only))
        return 0

    dest = args.dest or Path(MODELS[args.model]["dest"])
    build(args.model, dest, args.dtype, args.vision_only, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
