"""평가 데이터셋 원본에서 자동 평가용 기준 지문을 만든다. 원본 사진과 라벨은 읽기만 한다.

HEIC 원본을 읽으려면 pillow-heif가 필요하다. 서비스 의존성에는 넣지 않는다.

    uv run --with pillow-heif==1.1.0 python -m scripts.build_eval_fingerprints \\
        --dataset eval-travel-photos --photos ../images/eval \\
        --photo-map ../outputs/.../eval-travel-photos/photo-map.csv \\
        --output app/observability/eval_sets/eval-travel-photos.json

photo-map.csv에는 source_relative_path, expected_place_id, expected_unclassified 열이 있어야 한다.
결과에는 파일 경로를 넣지 않고 지문과 정답 장소만 둔다.
"""

import argparse
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

from app.observability.datasets import FINGERPRINT, fingerprint

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None


def resolve(root: Path, relative: str) -> Path:
    # macOS 파일 이름은 NFD로 저장될 수 있다.
    for form in ("NFC", "NFD"):
        path = root / unicodedata.normalize(form, relative)
        if path.exists():
            return path
    raise FileNotFoundError(relative)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--photos", required=True, type=Path)
    parser.add_argument("--photo-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    items = []
    with args.photo_map.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        value = fingerprint(resolve(args.photos, row["source_relative_path"]).read_bytes())
        if value is None:
            raise SystemExit(f"사진을 읽지 못했습니다: {row['source_relative_path']}")
        unclassified = row["expected_unclassified"] == "True"
        items.append(
            {"phash": f"{value:016x}", "label": None if unclassified else row["expected_place_id"]}
        )
    output = {
        "dataset": args.dataset,
        "fingerprint": FINGERPRINT,
        "photos": len(items),
        "photo_map_sha256": hashlib.sha256(args.photo_map.read_bytes()).hexdigest(),
        "items": items,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
    labels = {item["label"] for item in items}
    print(f"{args.dataset}: 사진 {len(items)}장, 정답 장소 {len(labels - {None})}개")


if __name__ == "__main__":
    main()
