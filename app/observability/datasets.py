"""알려진 평가 데이터셋을 사진 지문으로 알아보고 정리 결과를 자동으로 채점한다

BE 사본은 긴 변 1024px JPEG로 다시 인코딩되므로 원본 해시로는 대조할 수 없고,
촬영 시각과 GPS가 없는 사진도 있어 EXIF로도 대조할 수 없다. 그래서 방향을 바로 세운
회색조 이미지의 pHash(64비트)를 지문으로 쓴다. 기준 지문은 원본에서 같은 함수로 만든다.

채점 규칙(2026-10-17 사용자 결정)
- 흐림이나 중복으로 미분류된 사진은 결과의 place_id, 즉 원래 장소로 채점한다.
- 정답이 미분류인 사진은 사진마다 단독 묶음으로 채점한다.
- 미분류 예측은 singleton과 exclude 두 정책으로 모두 계산한다.

채점은 관측 기능이다. 실패해도 정리 결과를 바꾸지 않는다.
"""

import io
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image, ImageOps

from app.observability.quality import EvaluationInput, evaluate

EVAL_SETS = Path(__file__).parent / "eval_sets"
FINGERPRINT = "phash8-upright-gray"
# 1024px JPEG로 흉내 낸 사본과 원본의 차이는 최대 2비트였다(398장, 2026-10-17). 여유를 둔 상한
MAX_DISTANCE = 8
# 여행 사진 가운데 이 비율 이상이 한 데이터셋과 맞을 때만 평가 여행으로 본다.
MIN_MATCH_RATIO = 0.9
MIN_MATCHED = 10
# 대시보드 지표에는 데이터셋 사진의 이 비율 이상이 대조된 여행만 반영한다(2026-10-17 사용자 결정).
# 일부만 올린 여행의 점수가 전체 데이터셋의 최근 지표를 덮어쓰지 않게 한다. 로그에는 모두 남긴다.
MIN_DATASET_COVERAGE = 0.9
POLICIES = ("singleton", "exclude")
# 위치는 판단했지만 품질 때문에 미분류로 간 사진. 원래 장소로 채점한다.
QUALITY_ISSUES = {"BLURRY", "DUPLICATED"}


def _phash(image) -> int:
    return int(str(imagehash.phash(image.convert("L"), hash_size=8, highfreq_factor=4)), 16)


def fingerprint(data: bytes) -> int | None:
    """기준 지문. 원본의 방향 태그대로 바로 세운 이미지의 pHash. 실패하면 None"""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.draft("RGB", (256, 256))
            return _phash(ImageOps.exif_transpose(image))
    except Exception:
        # 손상된 EXIF는 OSError가 아닌 예외(SyntaxError 등)도 낸다.
        # 관측 기능이 작업을 멈추면 안 된다.
        return None


def candidates(data: bytes) -> tuple[int, ...] | None:
    """사본 쪽 지문 후보. 픽셀 그대로와 방향 태그대로 바로 세운 것

    BE가 픽셀을 회전하고 태그를 지웠는지, 픽셀은 두고 태그를 남겼는지, 픽셀을 회전하고
    태그도 남겼는지에 따라 바로 선 이미지가 둘 중 하나가 된다. 둘 다 기준과 대조한다.
    JPEG는 draft로 축소 디코딩해 사진 한 장에 수 ms 안에 계산한다.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.draft("RGB", (256, 256))
            raw = _phash(image)
            try:
                upright = _phash(ImageOps.exif_transpose(image))
            except Exception:
                upright = raw
    except Exception:
        return None
    return (raw,) if upright == raw else (raw, upright)


@dataclass(frozen=True)
class ReferenceSet:
    dataset: str
    photos: int
    hashes: tuple[int, ...]
    # 정답 장소. None은 정답이 미분류인 사진
    labels: tuple[str | None, ...]

    def nearest(self, values: int | tuple[int, ...]) -> tuple[int, int]:
        """(거리, 기준 사진 위치). 같은 장소의 거의 같은 사진이면 어느 쪽이든 정답은 같다."""
        values = values if isinstance(values, tuple) else (values,)
        return min(
            ((value ^ ref).bit_count(), i) for value in values for i, ref in enumerate(self.hashes)
        )


@lru_cache(maxsize=1)
def reference_sets() -> tuple[ReferenceSet, ...]:
    sets = []
    for path in sorted(EVAL_SETS.glob("*.json")):
        data = json.loads(path.read_text())
        if data["fingerprint"] != FINGERPRINT:
            raise ValueError(f"{path.name}의 지문 방식이 {FINGERPRINT}와 다릅니다.")
        sets.append(
            ReferenceSet(
                dataset=data["dataset"],
                photos=data["photos"],
                hashes=tuple(int(item["phash"], 16) for item in data["items"]),
                labels=tuple(item["label"] for item in data["items"]),
            )
        )
    return tuple(sets)


@dataclass(frozen=True)
class Match:
    dataset: str
    photos: int
    # trip_attachment_id → 정답 장소, 정답이 미분류면 None
    truth: dict[int, str | None]
    unmatched: int
    max_distance: int
    # 평가 여행으로 받아들일 기준을 넘었는지. 넘지 못해도 가장 가까운 데이터셋은 로그로 남긴다.
    accepted: bool = True

    @property
    def ratio(self) -> float:
        return len(self.truth) / (len(self.truth) + self.unmatched)


def closest(hashes: dict[int, tuple[int, ...] | int | None], sets=None) -> Match | None:
    """사진이 가장 많이 맞은 평가 데이터셋. 기준을 넘었는지는 accepted에 담는다."""
    if not hashes:
        return None
    best = None
    for reference in sets if sets is not None else reference_sets():
        truth, distances = {}, []
        for photo_id, value in hashes.items():
            if value is None:
                continue
            distance, index = reference.nearest(value)
            if distance <= MAX_DISTANCE:
                truth[photo_id] = reference.labels[index]
                distances.append(distance)
        if best is None or len(truth) > len(best.truth):
            best = Match(
                reference.dataset,
                reference.photos,
                truth,
                len(hashes) - len(truth),
                max(distances, default=0),
                len(truth) >= MIN_MATCHED and len(truth) / len(hashes) >= MIN_MATCH_RATIO,
            )
    return best


def detect(hashes: dict[int, tuple[int, ...] | int | None], sets=None) -> Match | None:
    """여행 사진 지문이 한 평가 데이터셋과 충분히 맞으면 그 데이터셋의 정답을 돌려준다."""
    match = closest(hashes, sets)
    return match if match is not None and match.accepted else None


def predicted_places(result: dict[str, Any]) -> dict[int, str | None]:
    """정리 결과의 사진별 예측 장소. 품질 미분류는 원래 장소, 나머지 미분류는 None"""
    predicted: dict[int, str | None] = {}
    for place in result["places"]:
        for photo in place["attachments"]:
            predicted[photo["trip_attachment_id"]] = place["place_id"]
    for photo in result["unclassified"]:
        place = photo.get("place_id") if photo.get("issue") in QUALITY_ISSUES else None
        predicted[photo["trip_attachment_id"]] = place
    return predicted


def score(match: Match, result: dict[str, Any], *, model: str, release: str) -> list[dict]:
    """두 미분류 정책으로 채점한 ai_evaluation 보고서 목록"""
    predicted = predicted_places(result)
    items = [
        {
            # 정답이 미분류인 사진은 사진마다 단독 묶음
            "expected_place": label if label is not None else f"unclassified:{photo_id}",
            "predicted_place": predicted.get(photo_id),
        }
        for photo_id, label in sorted(match.truth.items())
        if photo_id in predicted
    ]
    reports = []
    for policy in POLICIES:
        if policy == "exclude" and all(item["predicted_place"] is None for item in items):
            # 모든 사진이 미분류면 제외 정책으로 채점할 사진이 없다. 다른 오류는 호출자가 기록한다.
            continue
        report = evaluate(
            EvaluationInput(
                feature="process",
                dataset=match.dataset,
                model=model,
                items=items,
                unclassified_policy=policy,
            )
        )
        coverage = len(match.truth) / match.photos if match.photos else 0.0
        report.update(
            release=release,
            source="auto",
            dataset_photos=match.photos,
            matched_photos=len(match.truth),
            unmatched_photos=match.unmatched,
            max_fingerprint_distance=match.max_distance,
            dataset_coverage=round(coverage, 3),
            dashboard=coverage >= MIN_DATASET_COVERAGE,
        )
        reports.append(report)
    return reports
