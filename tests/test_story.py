"""사진 선별과 실제 스토리 핸들러의 정상 및 오류 경계를 검증한다."""

import io
import json
import logging
from contextlib import nullcontext
from datetime import datetime
from unittest.mock import AsyncMock

import numpy as np
import pytest
from PIL import Image

from app.core.errors import ErrorCode, PipelineCancelled, PipelineError
from app.core.tracing import end_trace, start_trace
from app.infra.vlm import ChatTurn, VLMUnavailable
from app.story.pipeline import StoryContext, StoryOptions, finalize, run, validate_request
from app.story.prompts import StoryDraft, response_format
from app.story.selection import (
    allocate,
    duplicate_threshold,
    select,
    similarities,
    unit_vectors,
)
from tests.observability_helpers import events
from tests.test_v2_jobs import scenario, story_request


class Images:
    def get_many(self, keys):
        result = {}
        for key in keys:
            buf = io.BytesIO()
            rng = np.random.default_rng(sum(key.encode()))
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
                buf, format="JPEG"
            )
            result[key] = buf.getvalue()
        return result


def request_two_places():
    request = story_request()
    request["places"].append(
        {
            "trip_place_id": 5002,
            "place_name": "공원",
            "attachments": [
                {
                    "trip_attachment_id": 102,
                    "analyze_storage_key": "b.jpg",
                    "taken_at": "2026-10-13T12:00:00+09:00",
                }
            ],
        }
    )
    return request


def draft(ids=(101, 102)):
    return json.dumps(
        {
            "story_summary": "해변과 공원에서 보낸 이틀",
            "blocks": [
                {
                    "trip_attachment_id": i,
                    "detail_summary": "사진 속 풍경",
                    "memo": "풍경을 사진에 남겼다.",
                }
                for i in ids
            ],
        }
    )


def context():
    vectors = AsyncMock()
    vectors.fetch.return_value = {101: np.array([1.0, 0]), 102: np.array([0.0, 1.0])}
    vlm = AsyncMock()
    vlm.chat.return_value = ChatTurn(draft())
    urls = AsyncMock()
    urls.resolve.side_effect = lambda key: "https://images.test/" + key
    return StoryContext(lambda: nullcontext(Images()), vectors, urls, vlm)


@pytest.mark.parametrize(
    "counts,budget,expected",
    [([10, 1], 5, [4, 1]), ([5, 5], 5, [3, 2]), ([1, 1], 30, [1, 1]), ([3, 1, 2], 6, [3, 1, 2])],
)
def test_allocation_preserves_places_and_budget(counts, budget, expected):
    assert allocate(counts, budget) == expected


def test_allocation_invariants():
    for n in range(1, 11):
        counts = list(range(1, n + 1))
        for budget in range(n, 31):
            result = allocate(counts, budget)
            assert sum(result) == min(sum(counts), budget)
            assert all(1 <= v <= c for v, c in zip(result, counts, strict=True))


def photos(*ids):
    p = validate_request(story_request()).places[0].attachments[0]
    return [p.model_copy(update={"trip_attachment_id": i}) for i in ids]


def picks(groups, vectors, budget, threshold):
    units = unit_vectors(groups, vectors)
    sims = similarities(units)
    return select(groups, units, sims, budget=budget, threshold=threshold)


def picked(groups, vectors, budget, threshold):
    return [p.trip_attachment_id for _, p in picks(groups, vectors, budget, threshold)]


def threshold_of(groups, vectors, z):
    return duplicate_threshold(similarities(unit_vectors(groups, vectors)), z)


def test_mean_embedding_selection():
    groups = {5001: photos(1, 2, 3)}
    vectors = {1: np.array([1.0, 0]), 2: np.array([0.9, 0.1]), 3: np.array([0, 1.0])}
    assert picked(groups, vectors, 1, None) == [2]


def test_duplicate_threshold_uses_only_same_place_pairs():
    groups = {5001: photos(1, 2, 3), 5002: photos(4, 5), 5003: photos(6)}
    raw = {1: [1.0, 0], 2: [1.0, 0.1], 3: [0, 1.0], 4: [1.0, 0], 5: [1.0, 1.0], 6: [1.0, 0]}
    vectors = {k: np.array(v) for k, v in raw.items()}
    unit = {k: v / np.linalg.norm(v) for k, v in vectors.items()}
    # 장소가 다른 쌍(예: 1과 4, 유사도 1.0)은 넣지 않는다.
    pairs = np.array([unit[a] @ unit[b] for a, b in ((1, 2), (1, 3), (2, 3), (4, 5))])
    assert threshold_of(groups, vectors, 1.0) == pytest.approx(pairs.mean() + pairs.std())
    assert threshold_of(groups, vectors, 0.0) == pytest.approx(pairs.mean())


def test_duplicate_threshold_without_same_place_pairs_is_none():
    groups = {5001: photos(1), 5002: photos(2)}
    vectors = {1: np.array([1.0, 0]), 2: np.array([1.0, 0])}
    assert threshold_of(groups, vectors, 1.0) is None


def test_selection_skips_photo_too_similar_to_chosen():
    groups = {5001: photos(1, 2, 3)}
    # 1과 2는 거의 같은 사진(유사도 약 0.99), 평균에 가까운 순서는 2, 1, 3
    vectors = {1: np.array([1.0, 0]), 2: np.array([0.99, 0.14]), 3: np.array([0.6, 0.8])}
    assert sorted(picked(groups, vectors, 2, None)) == [1, 2]
    assert sorted(picked(groups, vectors, 2, 0.95)) == [2, 3]


def test_selection_leaves_slot_empty_when_only_duplicates_remain():
    groups = {5001: photos(1, 2)}
    vectors = {1: np.array([1.0, 0]), 2: np.array([0.99, 0.14])}
    # 후보가 2장이면 평균 근접도가 수학적으로 같아 어느 쪽이 먼저인지는 반올림으로 갈린다.
    assert len(picked(groups, vectors, 2, 0.95)) == 1


def test_similarity_equal_to_threshold_counts_as_duplicate():
    groups = {5001: photos(1, 2)}
    vectors = {1: np.array([1.0, 0]), 2: np.array([0, 1.0])}
    assert len(picked(groups, vectors, 2, 0.0)) == 1
    assert len(picked(groups, vectors, 2, 1e-9)) == 2


def test_threshold_and_selection_compare_the_same_similarity_values():
    # 쌍이 하나면 기준이 그 쌍의 유사도와 같다. 기준 계산과 선택이 다른 연산으로 유사도를 구하면
    # 마지막 자리 반올림 때문에 판정이 뒤바뀌므로, 고차원 float32 벡터로 같은 값을 쓰는지 본다.
    rng = np.random.default_rng(7)
    groups = {5001: photos(1, 2)}
    for _ in range(200):
        a, b = rng.standard_normal((2, 2048)).astype(np.float32)
        vectors = {1: a, 2: b}
        assert len(picked(groups, vectors, 2, threshold_of(groups, vectors, 1.0))) == 1


def test_selection_compares_with_every_chosen_photo():
    # 평균에 가까운 순서는 3, 1, 2, 4다.
    # 2는 처음 고른 3과 닮았고(0.986) 직전에 고른 1과는 다르다(0.684).
    groups = {5001: photos(1, 2, 3, 4)}
    raw = {1: [0.5, 1.0, 0.1], 2: [0.9, 0.3, 0.4], 3: [0.8, 0.4, 0.5], 4: [0.0, 0.8, 0.5]}
    vectors = {k: np.array(v) for k, v in raw.items()}
    assert sorted(picked(groups, vectors, 4, 0.9)) == [1, 3, 4]


def test_selection_ignores_skipped_photos():
    # 평균에 가까운 순서는 2, 1, 3, 4다. 1은 2와 닮아 건너뛴다(0.931). 3은 건너뛴 1과 닮았지만(0.95)
    # 고른 2와는 다르므로(0.878) 고른다. 4는 2와 닮아 건너뛴다(0.971).
    groups = {5001: photos(1, 2, 3, 4)}
    raw = {1: [0.5, 0.6, 0.9], 2: [0.4, 0.9, 0.6], 3: [0.8, 0.5, 0.7], 4: [0.3, 0.5, 0.2]}
    vectors = {k: np.array(v) for k, v in raw.items()}
    assert sorted(picked(groups, vectors, 4, 0.9)) == [2, 3]


def test_unfilled_slot_is_not_moved_to_another_place():
    # 예산 4장은 2장, 2장으로 배분된다. 5001의 두 장은 거의 같아 한 장만 고르고,
    # 남은 자리를 5002로 넘기지 않는다.
    groups = {5001: photos(1, 2), 5002: photos(3, 4, 5)}
    raw = {1: [1.0, 0, 0], 2: [0.99, 0.14, 0], 3: [0, 1.0, 0], 4: [0, 0, 1.0], 5: [0, 0.7, 0.7]}
    vectors = {k: np.array(v) for k, v in raw.items()}
    assert allocate([2, 3], 4) == [2, 2]
    counts = {}
    for place_id, _ in picks(groups, vectors, 4, 0.9):
        counts[place_id] = counts.get(place_id, 0) + 1
    assert counts == {5001: 1, 5002: 2}


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_duplicate_setting_rejects_invalid_values(value):
    from pydantic import ValidationError

    from app.jobs.config import JobsSettings

    with pytest.raises(ValidationError):
        JobsSettings(_env_file=None, API_KEY="test", STORY_DEDUPE_Z=value)


@scenario
async def test_one_vlm_call_and_code_derived_metadata():
    ctx = context()
    reporter = AsyncMock()
    result = await run(request_two_places(), reporter, ctx, StoryOptions())
    assert ctx.vlm.chat.await_count == 1
    args = ctx.vlm.chat.call_args
    assert args.args[1] == ["https://images.test/a.jpg", "https://images.test/b.jpg"]
    assert args.kwargs["response_format"]["type"] == "json_schema"
    assert [b["trip_place_id"] for b in result["blocks"]] == [5001, 5002]
    assert result["blocks"][1]["day_label"] == "2일차, 공원"
    assert result["verification"] == {"passed": True, "retry_count": 0}
    assert reporter.report.call_args.args == ("VERIFYING", 4, 4)


def request_near_duplicates():
    request = request_two_places()
    request["places"][0]["attachments"] += [
        {"trip_attachment_id": i, "analyze_storage_key": key, "taken_at": taken}
        for i, key, taken in (
            (103, "c.jpg", "2026-10-12T06:00:01+09:00"),
            (104, "d.jpg", "2026-10-12T07:00:00+09:00"),
        )
    ]
    return request


@pytest.mark.parametrize(
    "dedupe_z,expected", [(1.0, [103, 104, 102]), (100.0, [101, 103, 104, 102])]
)
@scenario
async def test_story_skips_near_duplicate_photo_by_trip_threshold(dedupe_z, expected, caplog):
    ctx = context()
    # 101과 103은 거의 같은 사진이다. 같은 장소 쌍 유사도의 평균 + 1.0 × 표준편차는 약 0.92다.
    ctx.vectors.fetch.return_value = {
        101: np.array([1.0, 0]),
        102: np.array([0, 1.0]),
        103: np.array([0.999, 0.04]),
        104: np.array([0.6, 0.8]),
    }
    ctx.vlm.chat.return_value = ChatTurn(draft(expected))
    caplog.set_level(logging.INFO)
    _, token = start_trace("story", emit_stage_logs=True)
    try:
        options = StoryOptions(dedupe_z=dedupe_z)
        result = await run(request_near_duplicates(), AsyncMock(), ctx, options)
    finally:
        end_trace(token)
    schema = response_format(expected)["json_schema"]["schema"]
    assert ctx.vlm.chat.call_args.kwargs["response_format"]["json_schema"]["schema"] == schema
    assert [b["trip_attachment_id"] for b in result["blocks"]] == expected
    selecting = next(r for r in events(caplog, "ai_stage") if r.stage == "selecting.select")
    # 예산 10장 가운데 후보 4장이 배정되고, 건너뛴 한 장만큼 비운다.
    assert selecting.selected == len(expected)
    assert selecting.unfilled == 4 - len(expected)
    if dedupe_z == 1.0:
        assert selecting.duplicate_threshold == pytest.approx(0.9248, abs=1e-4)


@scenario
async def test_each_image_is_labeled_and_precedes_photo_text():
    ctx = context()
    await run(request_two_places(), AsyncMock(), ctx, StoryOptions())
    content = ctx.vlm.chat.call_args.args[0][-1]["content"]
    assert content[:4] == [
        {"type": "text", "text": "image_number 1, trip_attachment_id 101:"},
        {"type": "image"},
        {"type": "text", "text": "image_number 2, trip_attachment_id 102:"},
        {"type": "image"},
    ]
    photos = json.loads(content[4]["text"])["photos"]
    assert [(p["image_number"], p["trip_attachment_id"]) for p in photos] == [(1, 101), (2, 102)]
    assert len(content) == 5


@pytest.mark.parametrize(
    "raw",
    ["not json", draft((101,)), draft((101, 999)), draft((101, 101, 102)), draft((True, 102))],
)
def test_model_output_rejects_missing_or_unselected_or_duplicate_photos(raw):
    message = validate_request(request_two_places())
    selected = [(p.trip_place_id, p.attachments[0]) for p in message.places]
    with pytest.raises(PipelineError) as error:
        finalize(raw, message, selected)
    assert error.value.code == ErrorCode.INTERNAL_ERROR
    assert error.value.detail["reason"] == "INVALID_STORY_OUTPUT"


def test_output_rejects_missing_photo_even_when_every_place_is_present():
    message = validate_request(request_two_places())
    selected = [(p.trip_place_id, p.attachments[0]) for p in message.places]
    selected.append((5001, selected[0][1].model_copy(update={"trip_attachment_id": 103})))
    with pytest.raises(PipelineError) as error:
        finalize(draft((101, 102)), message, selected)
    assert error.value.detail["reason"] == "INVALID_STORY_OUTPUT"
    assert len(finalize(draft((103, 102, 101)), message, selected)["blocks"]) == 3


def test_day_label_counts_days_from_trip_start_in_kst():
    # 여행 기간은 10/12~10/14다. 일차와 시작일 앞 사진의 날짜는 서울 시간 기준 날짜로 정하고,
    # 종료일 뒤의 사진은 기간을 넘는 일차가 된다(Q9에서 바꾸지 않은 동작).
    message = validate_request(request_two_places())
    first = message.places[0].attachments[0]
    taken = {
        201: "2026-10-10T16:00:00+00:00",  # 서울 시간 10/11 01:00
        202: "2026-10-11T15:10:00+00:00",  # 서울 시간 10/12 00:10
        203: "2026-10-14T23:50:00+09:00",
        204: "2026-10-15T09:00:00+09:00",
    }
    selected = [
        (
            5001,
            first.model_copy(
                update={"trip_attachment_id": i, "taken_at": datetime.fromisoformat(t)}
            ),
        )
        for i, t in taken.items()
    ] + [(5002, message.places[1].attachments[0])]
    result = finalize(draft((201, 202, 203, 204, 102)), message, selected)
    labels = {b["trip_attachment_id"]: b["day_label"] for b in result["blocks"]}
    assert labels == {
        201: "2026-10-11, 해변",
        202: "1일차, 해변",
        102: "2일차, 공원",
        203: "3일차, 해변",
        204: "4일차, 해변",
    }


def test_day_label_is_cut_at_40_characters():
    request = request_two_places()
    request["places"][0]["place_name"] = "가" * 50
    message = validate_request(request)
    selected = [(p.trip_place_id, p.attachments[0]) for p in message.places]
    label = finalize(draft((101, 102)), message, selected)["blocks"][0]["day_label"]
    assert label == ("1일차, " + "가" * 50)[:40]
    assert len(label) == 40


@pytest.mark.parametrize("count", [1, 2, 10])
def test_response_schema_requires_actual_selected_photos(count):
    ids = list(range(101, 101 + count))
    schema = response_format(ids)["json_schema"]["schema"]
    blocks = schema["properties"]["blocks"]
    assert blocks["minItems"] == blocks["maxItems"] == count
    block = schema["$defs"]["DraftBlock"]["properties"]
    assert block["trip_attachment_id"]["enum"] == ids
    assert schema["properties"]["story_summary"]["maxLength"] == 40
    assert block["detail_summary"]["maxLength"] == 40
    assert block["memo"]["maxLength"] == 125


def test_response_schemas_do_not_share_mutable_state_between_requests():
    ids = [101, 102]
    first = response_format(ids)["json_schema"]["schema"]
    ids.append(103)
    second = response_format([201])["json_schema"]["schema"]
    first["properties"]["blocks"]["maxItems"] = 99
    first["$defs"]["DraftBlock"]["properties"]["trip_attachment_id"]["enum"].append(999)
    assert second["properties"]["blocks"]["maxItems"] == 1
    assert second["$defs"]["DraftBlock"]["properties"]["trip_attachment_id"]["enum"] == [201]
    assert StoryDraft.model_json_schema()["properties"]["blocks"]["maxItems"] == 30
    assert response_format([101, 102])["json_schema"]["schema"] != first


@pytest.mark.parametrize(
    "fault,code",
    [
        (VLMUnavailable("secret endpoint"), ErrorCode.VLM_UNAVAILABLE),
        (TimeoutError("secret endpoint"), ErrorCode.TIMEOUT),
    ],
)
@scenario
async def test_generation_error_mapping(fault, code):
    ctx = context()
    ctx.vlm.chat.side_effect = fault
    with pytest.raises(PipelineError) as error:
        await run(request_two_places(), AsyncMock(), ctx, StoryOptions())
    assert error.value.code == code and "secret" not in str(error.value)


@scenario
async def test_missing_vectors_does_not_silently_skip_place():
    ctx = context()
    ctx.vectors.fetch.return_value = {101: np.array([1.0, 0.0])}
    with pytest.raises(PipelineError) as error:
        await run(request_two_places(), AsyncMock(), ctx, StoryOptions())
    assert error.value.code == ErrorCode.MODEL_NOT_READY
    assert error.value.detail["trip_attachment_ids"] == [102]
    ctx.vlm.chat.assert_not_called()


@scenario
async def test_empty_place_after_filtering_does_not_generate():
    ctx = context()
    with pytest.raises(PipelineError) as error:
        await run(request_two_places(), AsyncMock(), ctx, StoryOptions(blur_min=1e20))
    assert error.value.detail["reason"] == "NO_USABLE_PHOTOS"
    ctx.vlm.chat.assert_not_called()


@scenario
async def test_cancel_after_generation_cannot_return_completed():
    ctx = context()
    reporter = AsyncMock()

    async def generate(*args, **kwargs):
        reporter.check_cancel.side_effect = PipelineCancelled()
        return ChatTurn(draft())

    ctx.vlm.chat.side_effect = generate
    with pytest.raises(PipelineCancelled):
        await run(request_two_places(), reporter, ctx, StoryOptions())


@scenario
async def test_invalid_request_prevents_all_external_work():
    request = request_two_places()
    request["places"][1]["attachments"][0]["trip_attachment_id"] = 101
    ctx = context()
    with pytest.raises(PipelineError) as error:
        await run(request, AsyncMock(), ctx, StoryOptions())
    assert error.value.code == ErrorCode.INVALID_REQUEST
    ctx.vectors.fetch.assert_not_called()
    ctx.vlm.chat.assert_not_called()


@scenario
async def test_handler_cleans_clients_after_initialization_failure(monkeypatch):
    from app.jobs.config import JobsSettings
    from app.story import handler

    cfg = JobsSettings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local")
    monkeypatch.setattr(handler, "JobsSettings", lambda: cfg)
    vlm = AsyncMock()
    vectors = AsyncMock()
    monkeypatch.setattr(handler.VllmClient, "from_settings", lambda _: vlm)
    monkeypatch.setattr(handler, "AsyncQdrantVectors", lambda *args: vectors)
    monkeypatch.setattr(
        handler.S3ImageResolver, "create", AsyncMock(side_effect=RuntimeError("S3 unavailable"))
    )
    with pytest.raises(RuntimeError):
        await handler.run(story_request(), AsyncMock())
    vlm.__aexit__.assert_awaited_once()
    vectors.aclose.assert_awaited_once()


@scenario
async def test_handler_passes_duplicate_setting(monkeypatch):
    from app.jobs.config import JobsSettings
    from app.story import handler

    cfg = JobsSettings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local", STORY_DEDUPE_Z=1.5)
    monkeypatch.setattr(handler, "JobsSettings", lambda: cfg)
    monkeypatch.setattr(handler.VllmClient, "from_settings", lambda _: AsyncMock())
    monkeypatch.setattr(handler, "AsyncQdrantVectors", lambda *args: AsyncMock())
    monkeypatch.setattr(handler.S3ImageResolver, "create", AsyncMock(return_value=AsyncMock()))
    pipeline = AsyncMock(return_value={})
    monkeypatch.setattr(handler, "run_pipeline", pipeline)
    await handler.run(story_request(), AsyncMock())
    assert pipeline.call_args.args[3].dedupe_z == 1.5


@scenario
async def test_actual_vlm_transport_and_job_result_outbox():
    import httpx

    from app.infra.vllm_client import VllmClient
    from app.jobs.execution import execute
    from app.jobs.kafka import RequestReceiver
    from tests.test_v2_jobs import environment, message

    requests = []

    def complete(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["response_format"]["json_schema"]["name"] == "travel_story"
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["blocks"]["minItems"] == 2
        assert schema["properties"]["blocks"]["maxItems"] == 2
        assert schema["$defs"]["DraftBlock"]["properties"]["trip_attachment_id"]["enum"] == [
            101,
            102,
        ]
        assert body["max_tokens"] == 2048
        assert "40자 이내의 짧은 한 구절" in body["messages"][0]["content"]
        content = body["messages"][-1]["content"]
        assert [part["type"] for part in content] == ["text", "image_url"] * 2 + ["text"]
        assert [part["image_url"]["url"] for part in content if part["type"] == "image_url"] == [
            "https://images.test/a.jpg",
            "https://images.test/b.jpg",
        ]
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": draft()}}]}
        )

    async with VllmClient(
        "http://vlm.test/v1", "test", transport=httpx.MockTransport(complete)
    ) as vlm:
        original = context()
        ctx = StoryContext(original.images, original.vectors, original.urls, vlm)
        async with environment() as (cfg, store, _, producer):
            request = request_two_places()
            await RequestReceiver(store, producer).accept(message(cfg, request, "story-create"))
            record = await store.claim("story", "story-owner")

            async def handler(request, reporter):
                return await run(request, reporter, ctx, StoryOptions())

            await execute(store, record.identity, record.owner, handler)
            done = await store.get(record.identity)
            assert done.status == "COMPLETED"
            assert done.event.type == "story_result"
            assert done.event.result["blocks"][1]["trip_place_id"] == 5002
            assert not done.published
            assert len(requests) == 1
