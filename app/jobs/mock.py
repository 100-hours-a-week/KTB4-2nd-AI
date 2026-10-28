"""인프라 연결 검증용 결과. 이미지 다운로드나 모델 추론을 수행하지 않는다."""

import asyncio

from app.jobs.execution import JobReporter
from app.jobs.models import PhotoMetricsRecord


async def run(request: dict, reporter: JobReporter) -> dict:
    if request["type"] == "process":
        total = len(request["attachments"])
        for step in (
            "DOWNLOADING",
            "EMBEDDING",
            "CLOCK",
            "LOCATING",
            "CLUSTERING",
            "FILTERING",
            "FINALIZING",
        ):
            await reporter.report(step, total, total)
            await asyncio.sleep(0.02)
        return {
            "places": [],
            "unclassified": [
                {
                    "trip_attachment_id": item["trip_attachment_id"],
                    "issue": "UNCLEAR_LOCATION",
                    "region_origin": "UNKNOWN",
                }
                for item in request["attachments"]
            ],
            "failed": [],
            "clock_offsets": [],
        }
    for index, step in enumerate(("SELECTING", "CAPTIONING", "WRITING", "VERIFYING"), 1):
        await reporter.report(step, index, 4)
        await asyncio.sleep(0.02)
    blocks = [
        {
            "order_number": i,
            "trip_place_id": place["trip_place_id"],
            "trip_attachment_id": place["attachments"][0]["trip_attachment_id"],
            "day_label": "테스트",
            "detail_summary": "연동 확인용 사진",
            "memo": "모델이 생성한 문장이 아닌 연동 확인용 결과입니다.",
        }
        for i, place in enumerate(request["places"], 1)
    ]
    return {
        "story_summary": "연동 확인용 스토리",
        "blocks": blocks,
        "verification": {"passed": False, "retry_count": 0},
    }


def embed_photos(photos: dict[int, str]) -> dict[int, PhotoMetricsRecord]:
    """연동 확인용 사진 묶음 처리. 다운로드와 임베딩을 하지 않으므로 저장할 사진 지표도 없음

    가짜 지표를 저장하면 같은 환경의 실제 정리가 그 값을 재사용하게 됨
    """
    return {}
