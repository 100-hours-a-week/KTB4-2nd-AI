"""API 명세의 예시 JSON이 스키마를 통과하는지, 명세 규칙을 검증기가 지키는지"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.schemas.health import HealthResponse
from app.schemas.process import (
    PlaceAttachment,
    ProcessAttachment,
    ProcessRequest,
    ProcessResult,
    UnclassifiedAttachment,
)
from app.schemas.task import TaskStatusResponse

# 명세 "사진 정리" 요청 예시
REQUEST_EXAMPLE = {
    "trip_name": "제주도 가을",
    "period": {"start_date": "2026-10-12", "end_date": "2026-10-14"},
    "regions": [{"latitude": 33.4996, "longitude": 126.5312}],
    "attachments": [
        {
            "trip_attachment_id": 101,
            "analyze_storage_key": "trips/77/analyze/a1b2c3.jpg",
            "taken_at": "2026-10-12T06:14:00+09:00",
            "latitude": 33.4580,
            "longitude": 126.9423,
            "device_model": "Apple iPhone 15",
        },
        {
            "trip_attachment_id": 102,
            "analyze_storage_key": "trips/77/analyze/d4e5f6.jpg",
            "taken_at": "2026-10-09T06:31:00+09:00",
            "latitude": None,
            "longitude": None,
            "device_model": "Canon EOS M50",
        },
    ],
    "existing_places": [],
}

# 명세 "정리 result" 예시
RESULT_EXAMPLE = {
    "places": [
        {
            "place_id": "p1",
            "matched_trip_place_id": None,
            "latitude": 33.4580,
            "longitude": 126.9423,
            "first_taken_at": "2026-10-12T06:14:00+09:00",
            "last_taken_at": "2026-10-12T07:30:00+09:00",
            "representative_attachment_id": 102,
            "attachments": [
                {
                    "trip_attachment_id": 101,
                    "taken_at": "2026-10-12T06:14:00+09:00",
                    "latitude": 33.4580,
                    "longitude": 126.9423,
                    "region_origin": "EXIF",
                    "evaluation": 74,
                },
                {
                    "trip_attachment_id": 102,
                    "taken_at": "2026-10-12T06:31:00+09:00",
                    "latitude": 33.4582,
                    "longitude": 126.9420,
                    "region_origin": "INFERRED",
                    "evaluation": 91,
                },
            ],
        }
    ],
    "unclassified": [
        {
            "trip_attachment_id": 211,
            "issue": "BLURRY",
            "place_id": "p1",
            "duplicate_of_attachment_id": None,
            "taken_at": "2026-10-12T06:20:00+09:00",
            "latitude": 33.4580,
            "longitude": 126.9423,
            "region_origin": "EXIF",
            "evaluation": 31,
        }
    ],
    "failed": [],
    "clock_offsets": [{"device_model": "Canon EOS M50", "days": 3, "matched_pairs": 5}],
}


def _attachment(i: int) -> dict:
    return {
        "trip_attachment_id": i,
        "taken_at": "2026-10-12T06:14:00+09:00",
        "latitude": 1.0,
        "longitude": 1.0,
        "region_origin": "EXIF",
        "evaluation": 50,
    }


def _place(place_id: str, ids: list[int]) -> dict:
    return {
        "place_id": place_id,
        "latitude": 1.0,
        "longitude": 1.0,
        "representative_attachment_id": ids[0],
        "attachments": [_attachment(i) for i in ids],
    }


# ---------- 요청 ----------


def test_request_example_passes():
    req = ProcessRequest.model_validate(REQUEST_EXAMPLE)
    assert len(req.attachments) == 2
    assert req.attachments[1].latitude is None


def test_request_ignores_unknown_fields():
    req = ProcessRequest.model_validate({**REQUEST_EXAMPLE, "added_by_backend": 1})
    assert not hasattr(req, "added_by_backend")


def test_request_rejects_naive_taken_at():
    with pytest.raises(ValidationError, match="timezone"):
        ProcessAttachment(
            trip_attachment_id=1, analyze_storage_key="k", taken_at="2026-10-12T06:14:00"
        )


def test_request_rejects_over_200_attachments():
    many = [{"trip_attachment_id": i, "analyze_storage_key": f"k{i}"} for i in range(201)]
    with pytest.raises(ValidationError):
        ProcessRequest.model_validate({**REQUEST_EXAMPLE, "attachments": many})


# ---------- 결과 ----------


def test_result_example_passes():
    res = ProcessResult.model_validate(RESULT_EXAMPLE)
    assert res.places[0].place_id == "p1"
    assert res.unclassified[0].issue == "BLURRY"
    assert res.clock_offsets[0].days == 3


def test_result_serializes_time_as_kst():
    att = PlaceAttachment(
        trip_attachment_id=1,
        taken_at=datetime(2026, 10, 11, 21, 14, tzinfo=UTC),
        latitude=1.0,
        longitude=1.0,
        region_origin="EXIF",
        evaluation=50,
    )
    assert '"taken_at":"2026-10-12T06:14:00+09:00"' in att.model_dump_json()


def test_place_attachment_rejects_unknown_origin():
    with pytest.raises(ValidationError, match="UNKNOWN"):
        PlaceAttachment.model_validate({**_attachment(1), "region_origin": "UNKNOWN"})


def test_place_rejects_representative_outside_attachments():
    bad = {**_place("p1", [1, 2]), "representative_attachment_id": 99}
    with pytest.raises(ValidationError, match="representative"):
        ProcessResult.model_validate({"places": [bad], "unclassified": []})


@pytest.mark.parametrize(
    "issue, extra, message",
    [
        ("BLURRY", {}, "place_id"),
        ("DUPLICATED", {"place_id": "p1"}, "duplicate_of_attachment_id"),
    ],
)
def test_unclassified_issue_rules(issue, extra, message):
    with pytest.raises(ValidationError, match=message):
        UnclassifiedAttachment.model_validate(
            {"trip_attachment_id": 1, "issue": issue, "region_origin": "EXIF", **extra}
        )


def test_unclassified_unclear_location_needs_nothing():
    u = UnclassifiedAttachment(
        trip_attachment_id=1, issue="UNCLEAR_LOCATION", region_origin="UNKNOWN"
    )
    assert u.place_id is None


@pytest.mark.parametrize(
    "places, unclassified",
    [
        ([_place("p1", [1, 2]), _place("p2", [2])], []),
        (
            [_place("p1", [1])],
            [{"trip_attachment_id": 1, "issue": "UNCLEAR_LOCATION", "region_origin": "UNKNOWN"}],
        ),
    ],
)
def test_result_rejects_duplicate_attachment_ids(places, unclassified):
    with pytest.raises(ValidationError, match="두 곳 이상"):
        ProcessResult.model_validate({"places": places, "unclassified": unclassified})


# ---------- 상태 응답과 헬스 ----------


def test_status_response_requires_result_only_when_completed():
    progress = {"done": 1, "total": 1}
    with pytest.raises(ValidationError, match="result"):
        TaskStatusResponse(trip_id=1, status="COMPLETED", progress=progress)
    with pytest.raises(ValidationError, match="error"):
        TaskStatusResponse(trip_id=1, status="FAILED", progress=progress)
    ok = TaskStatusResponse(
        trip_id=1, status="PROCESSING", progress=progress, current_step="EMBEDDING"
    )
    assert ok.result is None and ok.error is None


def test_status_response_failed_example():
    res = TaskStatusResponse.model_validate(
        {
            "trip_id": 77,
            "status": "FAILED",
            "progress": {"done": 12, "total": 128},
            "current_step": None,
            "result": None,
            "error": {
                "code": "DOWNLOAD_FAILED",
                "message": "사본 다운로드에 실패했습니다.",
                "detail": {"keys": ["trips/77/analyze/d4e5f6.jpg"]},
            },
        }
    )
    assert res.error is not None and res.error.code == "DOWNLOAD_FAILED"


def test_health_example():
    h = HealthResponse.model_validate(
        {
            "status": "ok",
            "model_loaded": True,
            "model_name": "siglip2-so400m-naflex",
            "active_tasks": 0,
            "queued": 0,
            "idle_seconds": 640,
        }
    )
    assert h.status == "ok"
