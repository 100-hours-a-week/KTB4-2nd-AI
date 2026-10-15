"""한 번의 VLM 집필을 수행하고 BE에 전달할 스토리 결과를 검증한다."""

from dataclasses import dataclass
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from app.core.errors import ErrorCode
from app.infra.vlm import VLMUnavailable
from app.jobs.models import StoryBlock, StoryMessage, StoryResult, Verification
from app.jobs.sync_bridge import run_sync
from app.search.context import ModelUnavailable
from app.story.prompts import StoryDraft, messages, response_format
from app.story.selection import failure, filter_candidates, select


@dataclass(frozen=True)
class StoryContext:
    images: object
    vectors: object
    urls: object
    vlm: object


@dataclass(frozen=True)
class StoryOptions:
    image_budget: int = 10
    blur_min: float = 80.0
    phash_max: int = 4
    max_tokens: int = 4096


def validate_request(request: dict) -> StoryMessage:
    try:
        message = StoryMessage.model_validate(request)
        places = [place.trip_place_id for place in message.places]
        ids = [p.trip_attachment_id for place in message.places for p in place.attachments]
        if len(set(places)) != len(places) or len(set(ids)) != len(ids):
            raise ValueError("장소 또는 사진 ID 중복")
        if message.period.start_date > message.period.end_date:
            raise ValueError("여행 기간 역전")
        return message
    except (ValidationError, ValueError):
        raise failure(ErrorCode.INVALID_REQUEST, "스토리 요청 형식이 올바르지 않습니다.") from None


def finalize(raw: str, request: StoryMessage, selected) -> dict:
    try:
        draft = StoryDraft.model_validate_json(raw)
        photos = {photo.trip_attachment_id: (place_id, photo) for place_id, photo in selected}
        ids = [block.trip_attachment_id for block in draft.blocks]
        if len(set(ids)) != len(ids) or any(photo_id not in photos for photo_id in ids):
            raise ValueError("선별되지 않은 사진 또는 중복 사진")
        if {photos[photo_id][0] for photo_id in ids} != {p.trip_place_id for p in request.places}:
            raise ValueError("선택 장소 누락")
        places = {place.trip_place_id: place.place_name for place in request.places}
        ordered = sorted(
            draft.blocks,
            key=lambda block: (
                photos[block.trip_attachment_id][1].taken_at,
                block.trip_attachment_id,
            ),
        )
        blocks = []
        for order, block in enumerate(ordered, 1):
            place_id, photo = photos[block.trip_attachment_id]
            day = (
                photo.taken_at.astimezone(ZoneInfo("Asia/Seoul")).date() - request.period.start_date
            ).days + 1
            if day < 1:
                label = photo.taken_at.astimezone(ZoneInfo("Asia/Seoul")).date().isoformat()
            else:
                label = f"{day}째 날"
            blocks.append(
                StoryBlock(
                    order_number=order,
                    trip_place_id=place_id,
                    day_label=f"{label}, {places[place_id]}"[:40],
                    **block.model_dump(),
                )
            )
        return StoryResult(
            story_summary=draft.story_summary,
            blocks=blocks,
            verification=Verification(passed=True, retry_count=0),
        ).model_dump(mode="json")
    except (ValidationError, ValueError):
        raise failure(
            ErrorCode.INTERNAL_ERROR,
            "생성된 스토리가 출력 규칙을 만족하지 못했습니다.",
            reason="INVALID_STORY_OUTPUT",
        ) from None


async def run(request: dict, reporter, ctx: StoryContext, options: StoryOptions) -> dict:
    message = validate_request(request)
    if len(message.places) > options.image_budget:
        raise failure(ErrorCode.INVALID_REQUEST, "사진 입력 한도가 선택한 장소 수보다 작습니다.")
    await reporter.check_cancel()
    await reporter.report("SELECTING", 0, 4)

    def filter_images(_, cancel):
        # S3 자격 증명 초기화와 사진 처리는 같은 스레드에서 수행하고 종료한다.
        with ctx.images() as images:
            return filter_candidates(
                message,
                images,
                blur_min=options.blur_min,
                phash_max=options.phash_max,
                cancelled=cancel,
            )

    groups = await run_sync(filter_images, reporter)
    await reporter.report("SELECTING", 1, 4)
    try:
        ids = [p.trip_attachment_id for group in groups.values() for p in group]
        await reporter.check_cancel()
        vectors = await ctx.vectors.fetch(ids)
        selected = select(groups, vectors, budget=options.image_budget)
        await reporter.check_cancel()
        urls = []
        for _, photo in selected:
            await reporter.check_cancel()
            urls.append(await ctx.urls.resolve(photo.analyze_storage_key))
        await reporter.report("WRITING", 2, 4)
        await reporter.check_cancel()
        raw = await ctx.vlm.chat(
            messages(message, selected), urls, options.max_tokens, response_format=response_format()
        )
        await reporter.check_cancel()
        await reporter.report("VERIFYING", 3, 4)
        result = finalize(raw, message, selected)
        await reporter.report("VERIFYING", 4, 4)
        await reporter.check_cancel()
        return result
    except TimeoutError:
        raise failure(ErrorCode.TIMEOUT, "스토리 외부 호출 시간이 초과되었습니다.") from None
    except ModelUnavailable:
        raise failure(ErrorCode.MODEL_NOT_READY, "사진 벡터를 조회할 수 없습니다.") from None
    except VLMUnavailable:
        raise failure(
            ErrorCode.VLM_UNAVAILABLE, "스토리 생성 서버 또는 사진 URL을 사용할 수 없습니다."
        ) from None
