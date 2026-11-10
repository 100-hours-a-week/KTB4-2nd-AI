"""사진 확인과 집필, 두 번의 VLM 호출로 스토리를 쓰고 BE에 전달할 결과를 검증한다."""

from dataclasses import dataclass
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from app.core.errors import ErrorCode
from app.core.logging import get_logger, validation_fields
from app.core.tracing import stage
from app.infra.vlm import VLMUnavailable
from app.jobs.models import StoryBlock, StoryMessage, StoryResult, Verification
from app.jobs.sync_bridge import run_sync
from app.search.context import ModelUnavailable
from app.story.prompts import (
    PhotoNotes,
    StoryDraft,
    messages,
    perceive_format,
    perceive_messages,
    response_format,
)
from app.story.selection import (
    duplicate_threshold,
    failure,
    filter_candidates,
    select,
    similarities,
    unit_vectors,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class StoryContext:
    images: object
    vectors: object
    urls: object
    vlm: object
    # 장소별 경험 메모 제공자(app.story.notes). 없으면 메모 없이 쓴다.
    notes: object | None = None


PERCEIVE_TOKENS_PER_PHOTO = 150
WRITE_TOKENS_PER_PHOTO = 200


@dataclass(frozen=True)
class StoryOptions:
    image_budget: int = 10
    blur_min: float = 80.0
    phash_max: int = 4
    dedupe_z: float = 1.0
    max_tokens: int = 2048
    perceive_max_tokens: int = 1024


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


def invalid_output(rule: str, **fields) -> Exception:
    """검증에서 어긴 규칙을 남기고 기존 실패를 만든다. 생성된 글 본문은 남기지 않는다."""
    log.warning(
        "생성된 스토리가 출력 규칙을 만족하지 못했습니다.",
        extra={"event": "ai_story_output_invalid", "result": "failed", "rule": rule, **fields},
    )
    return failure(
        ErrorCode.INTERNAL_ERROR,
        "생성된 스토리가 출력 규칙을 만족하지 못했습니다.",
        reason="INVALID_STORY_OUTPUT",
    )


async def perceive(vlm, message: StoryMessage, selected, urls, max_tokens: int) -> dict | None:
    """사진마다 실제로 찍힌 것을 적게 한다. 형식이 틀리면 메모 없이 쓰도록 None을 돌려준다.

    생성 서버 오류와 시간 초과는 집필 호출과 같이 호출자에게 올린다.
    """
    ids = [photo.trip_attachment_id for _, photo in selected]
    turn = await vlm.chat(
        perceive_messages(message, selected), urls, max_tokens, response_format=perceive_format(ids)
    )
    try:
        notes = PhotoNotes.model_validate_json(turn.content).notes
        shows = {note.trip_attachment_id: note.shows for note in notes}
        if len(shows) != len(notes) or set(shows) != set(ids):
            raise ValueError("사진 확인 메모의 사진 ID가 요청과 다르다")
    except (ValidationError, ValueError):
        log.warning(
            "사진 확인 메모가 출력 규칙을 만족하지 못해 메모 없이 씁니다.",
            extra={"event": "ai_story_perceive_invalid", "result": "skipped", "photos": len(ids)},
        )
        return None
    return shows


async def place_notes(provider, message: StoryMessage) -> dict:
    """장소별 경험 메모를 가져온다. 메모는 보조 자료이므로 실패해도 스토리 생성을 막지 않는다."""
    if provider is None:
        return {}
    try:
        found = await provider.notes_for(
            [(place.trip_place_id, place.place_name) for place in message.places]
        )
    except Exception as exc:
        log.warning(
            "장소 경험 메모를 가져오지 못해 메모 없이 씁니다.",
            extra={
                "event": "ai_story_place_notes_failed",
                "result": "skipped",
                "error_type": type(exc).__name__,
            },
        )
        return {}
    return {place_id: notes for place_id, notes in found.items() if notes}


def day_label(taken_at, start_date, place_name: str) -> str:
    """N일차는 한국 시각의 날짜로 센다. 여행 시작 전 사진은 날짜를 그대로 쓴다."""
    local = taken_at.astimezone(ZoneInfo("Asia/Seoul")).date()
    day = (local - start_date).days + 1
    label = local.isoformat() if day < 1 else f"{day}일차"
    return f"{label}, {place_name}"[:40]


def finalize(raw: str, request: StoryMessage, selected) -> dict:
    try:
        draft = StoryDraft.model_validate_json(raw)
    except ValidationError as exc:
        raise invalid_output(
            "draft_schema",
            validation_errors=validation_fields(exc.errors()),
            output_chars=len(raw),
        ) from None
    photos = {photo.trip_attachment_id: (place_id, photo) for place_id, photo in selected}
    ids = [block.trip_attachment_id for block in draft.blocks]
    if len(set(ids)) != len(ids):
        raise invalid_output("duplicate_photo", blocks=len(ids), unique_photos=len(set(ids)))
    if set(ids) != set(photos):
        raise invalid_output(
            "photo_set_mismatch",
            missing_photos=len(set(photos) - set(ids)),
            unexpected_photos=len(set(ids) - set(photos)),
        )
    if {photos[photo_id][0] for photo_id in ids} != {p.trip_place_id for p in request.places}:
        raise invalid_output("place_missing")
    try:
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
            blocks.append(
                StoryBlock(
                    order_number=order,
                    trip_place_id=place_id,
                    day_label=day_label(
                        photo.taken_at, request.period.start_date, places[place_id]
                    ),
                    **block.model_dump(exclude={"about"}),
                )
            )
        return StoryResult(
            story_summary=draft.story_summary,
            blocks=blocks,
            verification=Verification(passed=True, retry_count=0),
        ).model_dump(mode="json")
    except ValidationError as exc:
        raise invalid_output(
            "result_schema", validation_errors=validation_fields(exc.errors())
        ) from None
    except ValueError:
        raise invalid_output("result_value") from None


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

    candidates = sum(len(place.attachments) for place in message.places)
    with stage("filter", parent="selecting", places=len(message.places)) as fields:
        fields["candidates"] = candidates
        groups = await run_sync(filter_images, reporter)
        fields["retained"] = sum(len(group) for group in groups.values())
        fields["excluded"] = candidates - fields["retained"]
    await reporter.report("SELECTING", 1, 4)
    try:
        ids = [p.trip_attachment_id for group in groups.values() for p in group]
        await reporter.check_cancel()
        with stage("vectors", parent="selecting", requested=len(ids)) as fields:
            vectors = await ctx.vectors.fetch(ids)
            fields["missing"] = len(set(ids) - set(vectors))
        with stage("select", parent="selecting", budget=options.image_budget) as fields:
            units = unit_vectors(groups, vectors)
            sims = similarities(units)
            threshold = duplicate_threshold(sims, options.dedupe_z)
            selected = select(groups, units, sims, budget=options.image_budget, threshold=threshold)
            fields["selected"] = len(selected)
            fields["unfilled"] = min(options.image_budget, len(ids)) - len(selected)
            fields["duplicate_threshold"] = None if threshold is None else round(threshold, 4)
        await reporter.check_cancel()
        urls = []
        with stage("urls", parent="selecting", count=len(selected)):
            for _, photo in selected:
                await reporter.check_cancel()
                urls.append(await ctx.urls.resolve(photo.analyze_storage_key))
        await reporter.report("WRITING", 2, 4)
        await reporter.check_cancel()
        with stage("place_notes", parent="writing", places=len(message.places)) as fields:
            notes = await place_notes(ctx.notes, message)
            fields["places_with_notes"] = len(notes)
        await reporter.check_cancel()
        # 출력이 상한에서 잘리면 스토리가 실패하므로, 사진이 많으면 상한을 사진 수에 맞춰 늘린다.
        # 사진 한 장에 사진 확인은 150토큰, 글은 200토큰까지 잡는다(11/10 측정: 평균 50, 100).
        perceive_tokens = max(options.perceive_max_tokens, PERCEIVE_TOKENS_PER_PHOTO * len(urls))
        write_tokens = max(options.max_tokens, WRITE_TOKENS_PER_PHOTO * len(urls))
        with stage("perceive", parent="writing", images=len(urls)) as fields:
            shows = await perceive(ctx.vlm, message, selected, urls, perceive_tokens)
            fields["noted"] = 0 if shows is None else len(shows)
        await reporter.check_cancel()
        with stage("generate", parent="writing", images=len(urls)):
            raw = await ctx.vlm.chat(
                messages(message, selected, shows, notes),
                urls,
                write_tokens,
                response_format=response_format(
                    [photo.trip_attachment_id for _, photo in selected]
                ),
            )
        await reporter.check_cancel()
        await reporter.report("VERIFYING", 3, 4)
        with stage("validate", parent="verifying"):
            result = finalize(raw.content, message, selected)
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
