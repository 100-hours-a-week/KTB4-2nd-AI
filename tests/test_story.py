"""사진 선별과 실제 스토리 핸들러의 정상 및 오류 경계를 검증한다."""

import io
import json
from contextlib import nullcontext
from unittest.mock import AsyncMock

import numpy as np
import pytest
from PIL import Image

from app.core.errors import ErrorCode, PipelineCancelled, PipelineError
from app.infra.vlm import VLMUnavailable
from app.story.pipeline import StoryContext, StoryOptions, finalize, run, validate_request
from app.story.prompts import StoryDraft, response_format
from app.story.selection import allocate, select
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
    vlm.chat.return_value = draft()
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


def test_mean_embedding_selection():
    message = validate_request(story_request())
    p = message.places[0].attachments[0]
    groups = {5001: [p.model_copy(update={"trip_attachment_id": i}) for i in range(1, 4)]}
    vectors = {1: np.array([1.0, 0]), 2: np.array([0.9, 0.1]), 3: np.array([0, 1.0])}
    assert select(groups, vectors, budget=1)[0][1].trip_attachment_id == 2


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
    assert result["blocks"][1]["day_label"] == "2째 날, 공원"
    assert result["verification"] == {"passed": True, "retry_count": 0}
    assert reporter.report.call_args.args == ("VERIFYING", 4, 4)


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
        return draft()

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
