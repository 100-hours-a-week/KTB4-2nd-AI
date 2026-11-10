"""분위기 변경 미리보기의 요청 검증, 출력 검증, 잠금과 오류 경계를 실제 GPU 없이 확인한다."""

import asyncio
import json
from copy import deepcopy

import fakeredis
import httpx
import pytest
from pydantic import ValidationError

from app.core.errors import AppError, ErrorCode
from app.infra.vlm import ChatTurn, VLMUnavailable
from app.schemas.story_rewrite import RewriteRequest
from app.story.prompts import EXAMPLES, MOOD_VOICE, SPEECH, STYLE, voice_of
from app.story.prompts import system_prompt as generation_system_prompt
from app.story.rewrite import InvalidRewrite, RewriteService, finalize
from app.story.rewrite_prompts import (
    CONTENT,
    HEAD,
    LENGTH,
    SOURCES,
    TAIL,
    messages,
    mood_voice,
    response_format,
    sentence_count,
    system_prompt,
)
from app.v2_main import create_app
from tests.test_v2_jobs import environment, scenario

BODY = {
    "trip_name": "제주도 가을",
    "period": {"start_date": "2026-10-12", "end_date": "2026-10-14"},
    "mood": "HUMOROUS",
    "blocks": [
        {
            "order_number": 1,
            "trip_attachment_id": 102,
            "place_name": "성산일출봉",
            "taken_at": "2026-10-12T06:31:00+09:00",
            "analyze_storage_key": "trips/77/analyze/a2.jpg",
            "day_label": "1일차, 성산일출봉",
            "detail_summary": "능선 위에서 본 일출",
            "memo": "새벽 어스름을 헤치고 오른 능선 위에서 하루가 시작됐다.",
        },
        {
            "order_number": 2,
            "trip_attachment_id": 135,
            "place_name": "협재해수욕장",
            "taken_at": "2026-10-13T15:20:00+09:00",
            "analyze_storage_key": "trips/77/analyze/b5.jpg",
            "day_label": None,
            "detail_summary": None,
            "memo": None,
        },
    ],
}


def rewrite_request(**changes):
    body = deepcopy(BODY)
    body.update(changes)
    return RewriteRequest.model_validate(body)


def draft(summary="일출 보고 바다까지", blocks=None):
    if blocks is None:
        blocks = [
            {"trip_attachment_id": 135, "detail_summary": "물빛이 맑은 협재 바다", "memo": None},
            {
                "trip_attachment_id": 102,
                "detail_summary": "능선 위로 떠오른 첫 해",
                "memo": "새벽에 능선까지 올라가서 해 뜨는 걸 봤어요.",
            },
        ]
    return json.dumps({"story_summary": summary, "blocks": blocks}, ensure_ascii=False)


def test_finalize_keeps_request_order_labels_and_null_memo():
    result = finalize(draft(), rewrite_request())
    assert [b.trip_attachment_id for b in result.blocks] == [102, 135]
    assert [b.order_number for b in result.blocks] == [1, 2]
    # 요청 값은 그대로, 새 사진은 생성과 같은 규칙으로 계산한다.
    assert result.blocks[0].day_label == "1일차, 성산일출봉"
    assert result.blocks[1].day_label == "2일차, 협재해수욕장"
    assert result.blocks[1].memo is None


def test_finalize_drops_memo_written_for_empty_block():
    blocks = json.loads(draft())["blocks"]
    blocks[0]["memo"] = "비어 있던 자리에 쓴 문장"
    assert finalize(draft(blocks=blocks), rewrite_request()).blocks[1].memo is None


@pytest.mark.parametrize(
    ("change", "rule"),
    [
        (lambda b: b[1].update(memo=None), "memo_missing"),
        (lambda b: b[0].update(trip_attachment_id=102), "duplicate_photo"),
        (lambda b: b.pop(), "photo_set_mismatch"),
        (lambda b: b[0].update(trip_attachment_id=999), "photo_set_mismatch"),
        (lambda b: b[0].update(extra="x"), "draft_schema"),
    ],
)
def test_finalize_rejects_broken_output(change, rule):
    blocks = json.loads(draft())["blocks"]
    change(blocks)
    with pytest.raises(InvalidRewrite) as exc:
        finalize(draft(blocks=blocks), rewrite_request())
    assert exc.value.rule == rule


def test_summary_over_contract_limit_is_rejected():
    with pytest.raises(InvalidRewrite) as exc:
        finalize(draft(summary="가" * 21), rewrite_request())
    assert exc.value.rule == "summary_too_long"
    # 스키마 상한(40자)을 넘는 초안은 형식 오류다.
    with pytest.raises(InvalidRewrite) as exc:
        finalize(draft(summary="가" * 41), rewrite_request())
    assert exc.value.rule == "draft_schema"


@pytest.mark.parametrize(
    "change",
    [
        {"mood": "EMOTIONAL"},
        {"blocks": list(reversed(BODY["blocks"]))},
        {"blocks": [BODY["blocks"][0], {**BODY["blocks"][1], "trip_attachment_id": 102}]},
        {
            "blocks": [
                {**BODY["blocks"][0], "order_number": i + 1, "trip_attachment_id": i + 1}
                for i in range(31)
            ]
        },
        {"trip_name": "가" * 11},
        {"blocks": [BODY["blocks"][0], {**BODY["blocks"][1], "order_number": 1}]},
        {"blocks": []},
        {"period": {"start_date": "2026-10-14", "end_date": "2026-10-12"}},
    ],
)
def test_invalid_requests_are_rejected(change):
    with pytest.raises(ValidationError):
        rewrite_request(**change)


def test_messages_mark_empty_memo_and_label_images_in_block_order():
    system, user = messages(rewrite_request(), 77)
    assert system["content"] == system_prompt("HUMOROUS", voice_of(77))
    assert MOOD_VOICE["HUMOROUS"] in system["content"]
    assert MOOD_VOICE["PLAIN"] not in system["content"]
    assert "story_summary는 10자 내외의 짧은 한 구절" in system["content"]
    labels = [p["text"] for p in user["content"] if p["type"] == "text"][:-1]
    assert labels == [
        "order_number 1, trip_attachment_id 102:",
        "order_number 2, trip_attachment_id 135:",
    ]
    assert sum(p["type"] == "image" for p in user["content"]) == 2
    data = json.loads(user["content"][-1]["text"])
    assert [b["memo_needed"] for b in data["blocks"]] == [True, False]
    # 서술 문장이 있는 블록에만 지금 문구의 문장 수를 준다.
    assert data["blocks"][0]["sentences"] == 1 and "sentences" not in data["blocks"][1]
    assert data["blocks"][1]["current_detail_summary"] is None
    assert "analyze_storage_key" not in data["blocks"][0]
    assert "day_label" not in data["blocks"][0]


def test_prompt_uses_the_same_style_speech_and_mood_as_generation():
    # 분위기를 바꿔도 생성과 같은 글투, 그 여행의 말씨, 분위기 문단, 예시를 쓴다(결정 40·50·120번).
    diary = "그날의 일기를 쓰듯 하루 순서대로 쓰고"
    for mood in ("PLAIN", "HUMOROUS", "LITERARY"):
        for voice in ("da", "yo"):
            system = system_prompt(mood, voice)
            generation = generation_system_prompt(mood, voice)
            for shared in (STYLE, SPEECH[voice], EXAMPLES[(mood, voice)]):
                assert shared in system and shared in generation
            assert MOOD_VOICE[mood] in generation and mood_voice(mood) in system
            assert system.startswith(HEAD + CONTENT + STYLE + LENGTH) and system.endswith(TAIL)
            assert diary not in system and "어조 규칙" not in system
    # 말씨는 분위기가 아니라 여행으로 정한다.
    assert SPEECH["da"] in messages(rewrite_request(), 33)[0]["content"]
    assert SPEECH["yo"] in messages(rewrite_request(), 40)[0]["content"]
    # 지금 문구의 사실은 담고 말끝은 따르지 않으며, 장소 메모나 지어내는 규칙은 없다.
    assert "새 문구에도 담으세요" in CONTENT and "말끝, 문장 구조" in CONTENT
    # 이 호출이 받지 않는 자료(사진 확인 메모, 장소 메모)의 이름은 어느 분위기에도 남지 않는다.
    for mood in ("PLAIN", "HUMOROUS", "LITERARY"):
        for voice in ("da", "yo"):
            system = system_prompt(mood, voice)
            for missing in ("place_notes", "shows", "사진 확인 메모", "장소 메모"):
                assert missing not in system
    assert mood_voice("PLAIN") == MOOD_VOICE["PLAIN"]
    assert mood_voice("HUMOROUS") == MOOD_VOICE["HUMOROUS"]
    # 생성 문단의 문구가 바뀌어 바꿀 구절이 사라지면 알 수 있게, 구절마다 한 번씩 있는지 본다.
    for generation, rewrite in SOURCES:
        assert MOOD_VOICE["LITERARY"].count(generation) == 1
        assert rewrite in mood_voice("LITERARY") and generation not in mood_voice("LITERARY")
    # 소리와 냄새는 [내용]이 허용한 "장면에서 짐작되는 감각"과 부딪치지 않게 막는다.
    assert "장면에서 짐작되는 감정, 감각은 글쓴이가 겪은 것으로 써도 됩니다" in CONTENT
    literary = mood_voice("LITERARY")
    assert "(사진이나 지금 문구에 있는 것)을 하나 짚어" in literary
    # 지금 문구에 적힌 소리나 냄새(사용자가 직접 적었을 수 있다)는 빼지 않게 두 자료를 모두 적는다.
    assert "지금 문구에 없고 사진에서도 짐작할 수 없는 소리나 냄새를 지어내지 말고" in literary
    # 생성이 사진을 찍었다는 말을 두 번까지 쓰므로, 그 말을 금지하면 지금 문구의 일을 담으라는
    # 지시와 부딪친다.
    assert "사진을 찍었다" not in CONTENT and "'사진 속에는 ~'" in CONTENT
    assert "memo를 장소 이름으로 시작하지 말고" in HEAD


@pytest.mark.parametrize(
    "memo,count",
    [
        ("택시를 타고 도착해서 전나무숲길을 걸었다.", 1),
        ("아쉬웠다ㅠㅠ 그래서 인형 뽑기에 매달렸음ㅋㅋ", 2),
        ("요금 15,000원 내고 탔는데 길더라. 정상에서 2.5km 걸었다.", 2),
        ("갔다. 먹었다. 걸었다. 잤다.", 4),
        ("하나 했다. 둘 했다. 셋 했다. 넷 했다. 다섯 했다. 여섯 했다.", 5),
        ("2023.5.1. 방문했다.", 1),
        ("아. 좋았다.", 1),
        ("ㅋㅋ", 1),
    ],
)
def test_sentence_count_follows_current_memo(memo, count):
    assert sentence_count(memo) == count


def test_response_format_limits_ids_count_and_summary():
    schema = response_format([102, 135])["json_schema"]["schema"]
    # 20자는 코드가 검사하고, 스키마는 낱말 중간에서 끊기지 않게 넉넉히 둔다.
    assert schema["properties"]["story_summary"]["maxLength"] == 40
    assert schema["properties"]["blocks"]["minItems"] == 2
    assert schema["properties"]["blocks"]["maxItems"] == 2
    # 위치마다 그 자리의 사진 ID만 받는다(결정 64번).
    pinned = schema["properties"]["blocks"]["prefixItems"]
    assert schema["properties"]["blocks"]["items"] is False
    assert [b["properties"]["trip_attachment_id"]["const"] for b in pinned] == [102, 135]
    block = pinned[1]
    assert {"type": "null"} in block["properties"]["memo"]["anyOf"]
    assert set(block["required"]) == {"trip_attachment_id", "detail_summary", "memo"}


class FakeVlm:
    def __init__(self, reply=None, error=None, delay=0.0, replies=None):
        self.replies = list(replies or [reply or draft()])
        self.error, self.delay = error, delay
        self.calls = []

    async def chat(self, messages, image_urls, max_tokens, response_format=None, **options):
        self.calls.append(
            {"urls": image_urls, "max_tokens": max_tokens, "messages": messages, **options}
        )
        self.formats = [*getattr(self, "formats", []), response_format]
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return ChatTurn(content=self.replies.pop(0))


class FakeUrls:
    async def resolve(self, key):
        return "https://s3.example/" + key

    async def aclose(self):
        pass


def service(vlm, redis=None, timeout_sec=5.0):
    redis = redis or fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RewriteService(vlm, FakeUrls(), redis, "t:", timeout_sec=timeout_sec, max_tokens=6144)


@scenario
async def test_service_streams_one_call_and_releases_lock():
    vlm = FakeVlm()
    svc = service(vlm)
    result = await svc.rewrite(7, rewrite_request())
    assert result.story_summary == "일출 보고 바다까지"
    assert [{k: v for k, v in call.items() if k != "messages"} for call in vlm.calls] == [
        {
            "urls": [
                "https://s3.example/trips/77/analyze/a2.jpg",
                "https://s3.example/trips/77/analyze/b5.jpg",
            ],
            "max_tokens": 6144,
            "stream": True,
        }
    ]
    assert await svc.redis.get("t:story-rewrite:7") is None


@pytest.mark.parametrize("trip_id,voice", [(33, "da"), (40, "yo")])
@scenario
async def test_service_writes_in_the_speech_of_the_trip(trip_id, voice):
    # 말씨는 생성과 같이 여행 ID로 정한다. 같은 여행은 분위기를 바꿔도 말씨가 같다.
    vlm = FakeVlm()
    await service(vlm).rewrite(trip_id, rewrite_request())
    system = vlm.calls[0]["messages"][0]["content"]
    assert system == system_prompt("HUMOROUS", voice) and SPEECH[voice] in system


@scenario
async def test_pinned_ids_follow_request_block_order_and_image_order():
    # 위치마다 고정한 사진 ID가 요청 블록·이미지 순서와 같아야 다른 사진의 글이 붙지 않는다
    # (결정 64번).
    vlm = FakeVlm()
    await service(vlm).rewrite(7, rewrite_request())
    pinned = vlm.formats[0]["json_schema"]["schema"]["properties"]["blocks"]["prefixItems"]
    assert [b["properties"]["trip_attachment_id"]["const"] for b in pinned] == [102, 135]
    assert vlm.calls[0]["urls"] == [
        "https://s3.example/trips/77/analyze/a2.jpg",
        "https://s3.example/trips/77/analyze/b5.jpg",
    ]


@scenario
async def test_same_trip_in_progress_is_duplicate_but_other_trip_is_not():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    slow = service(FakeVlm(delay=0.2), redis)
    first = asyncio.create_task(slow.rewrite(7, rewrite_request()))
    await asyncio.sleep(0.05)
    with pytest.raises(AppError) as exc:
        await service(FakeVlm(), redis).rewrite(7, rewrite_request())
    assert exc.value.code is ErrorCode.DUPLICATE_TASK and exc.value.http_status == 409
    await service(FakeVlm(), redis).rewrite(8, rewrite_request())
    await first


@scenario
async def test_caller_disconnect_does_not_stop_processing_or_leave_lock():
    svc = service(FakeVlm(delay=0.2))
    caller = asyncio.create_task(svc.rewrite(7, rewrite_request()))
    await asyncio.sleep(0.05)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert await svc.redis.get("t:story-rewrite:7") is not None
    await svc.aclose()
    assert await svc.redis.get("t:story-rewrite:7") is None


@pytest.mark.parametrize(
    ("vlm", "code", "status"),
    [
        (FakeVlm(delay=1.0), ErrorCode.TIMEOUT, 504),
        (FakeVlm(error=TimeoutError()), ErrorCode.TIMEOUT, 504),
        (FakeVlm(error=VLMUnavailable()), ErrorCode.VLM_UNAVAILABLE, 503),
        (FakeVlm(reply="{}"), ErrorCode.INTERNAL_ERROR, 500),
        (FakeVlm(error=RuntimeError("x")), ErrorCode.INTERNAL_ERROR, 500),
    ],
)
@scenario
async def test_failures_map_to_contract_codes_and_release_lock(vlm, code, status):
    svc = service(vlm, timeout_sec=0.1)
    with pytest.raises(AppError) as exc:
        await svc.rewrite(7, rewrite_request())
    assert (exc.value.code, exc.value.http_status) == (code, status)
    assert await svc.redis.get("t:story-rewrite:7") is None


@scenario
async def test_http_endpoint_contract():
    async with environment() as (cfg, a, _, _):
        svc = service(FakeVlm(), a.redis)
        app = create_app(cfg, a, start_kafka=False, rewrite=svc)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://api.test"
            ) as client,
        ):
            path = "/trips/7/story/rewrite"
            assert (await client.post(path, json=BODY)).status_code == 401
            auth = {"Authorization": "Bearer test"}
            response = await client.post(path, json=BODY, headers=auth)
            invalid = await client.post(path, json={**BODY, "mood": "CALM"}, headers=auth)
    assert response.status_code == 200
    assert response.json() == {
        "story_summary": "일출 보고 바다까지",
        "blocks": [
            {
                "order_number": 1,
                "trip_attachment_id": 102,
                "day_label": "1일차, 성산일출봉",
                "detail_summary": "능선 위로 떠오른 첫 해",
                "memo": "새벽에 능선까지 올라가서 해 뜨는 걸 봤어요.",
            },
            {
                "order_number": 2,
                "trip_attachment_id": 135,
                "day_label": "2일차, 협재해수욕장",
                "detail_summary": "물빛이 맑은 협재 바다",
                "memo": None,
            },
        ],
    }
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "INVALID_REQUEST"


@scenario
async def test_endpoint_without_rewrite_dependencies_is_unavailable():
    async with environment() as (cfg, a, _, _):
        app = create_app(cfg, a, start_kafka=False)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://api.test"
            ) as client,
        ):
            response = await client.post(
                "/trips/7/story/rewrite", json=BODY, headers={"Authorization": "Bearer test"}
            )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "VLM_UNAVAILABLE"


def test_thirty_unique_blocks_are_accepted():
    blocks = [
        {**BODY["blocks"][0], "order_number": i + 1, "trip_attachment_id": i + 1} for i in range(30)
    ]
    assert len(rewrite_request(blocks=blocks).blocks) == 30


def test_empty_day_label_is_returned_as_given():
    blocks = deepcopy(BODY["blocks"])
    blocks[0]["day_label"] = ""
    result = finalize(draft(), rewrite_request(blocks=blocks))
    assert result.blocks[0].day_label == ""


@scenario
async def test_shutdown_cancels_slow_work_and_releases_lock():
    svc = service(FakeVlm(delay=5.0), timeout_sec=10.0)
    caller = asyncio.create_task(svc.rewrite(7, rewrite_request()))
    await asyncio.sleep(0.05)
    await svc.aclose(wait_sec=0.05)
    assert await svc.redis.get("t:story-rewrite:7") is None
    with pytest.raises(asyncio.CancelledError):
        await caller


@scenario
async def test_rewrite_service_builds_gated_client_with_shorter_call_limit(monkeypatch):
    from app.api import search_bootstrap
    from app.infra.image_resolver import S3ImageResolver

    async def fake_create(cls, settings):
        return FakeUrls()

    monkeypatch.setattr(S3ImageResolver, "create", classmethod(fake_create))
    async with environment(
        VLM_BASE_URL="https://vlm.example/v1", VLM_MODEL="gemma-4", STORY_REWRITE_ENABLED=True
    ) as (cfg, a, _, _):
        async with search_bootstrap.rewrite_service(cfg, a.redis) as svc:
            assert (svc.timeout_sec, svc.max_tokens) == (300, 6144)
            assert svc.vlm.timeout_sec == 290
            assert svc.vlm.client._timeout_sec == 290


@scenario
async def test_zero_trip_id_is_rejected():
    async with environment() as (cfg, a, _, _):
        app = create_app(cfg, a, start_kafka=False, rewrite=service(FakeVlm(), a.redis))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://api.test"
            ) as client,
        ):
            response = await client.post(
                "/trips/0/story/rewrite", json=BODY, headers={"Authorization": "Bearer test"}
            )
    assert response.status_code == 422


@scenario
async def test_long_summary_fails_without_second_call_and_releases_lock(caplog):
    import logging

    caplog.set_level(logging.INFO)
    vlm = FakeVlm(replies=[draft(summary="판교와 잠실을 오간 카카오 부스트업데이 이틀")])
    svc = service(vlm)
    with pytest.raises(AppError) as exc:
        await svc.rewrite(7, rewrite_request())
    assert (exc.value.code, exc.value.http_status) == (ErrorCode.INTERNAL_ERROR, 500)
    assert len(vlm.calls) == 1
    assert await svc.redis.get("t:story-rewrite:7") is None
    invalid = [
        r for r in caplog.records if getattr(r, "event", None) == "ai_story_rewrite_output_invalid"
    ]
    assert [(r.rule, r.summary_chars) for r in invalid] == [("summary_too_long", 24)]
    # 요약 본문은 로그에 남기지 않는다.
    text = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert "판교와 잠실" not in text


@scenario
async def test_broken_blocks_are_reported_before_long_summary(caplog):
    import logging

    caplog.set_level(logging.INFO)
    blocks = json.loads(draft())["blocks"]
    blocks[1]["memo"] = None
    with pytest.raises(AppError):
        await service(FakeVlm(replies=[draft(summary="가" * 25, blocks=blocks)])).rewrite(
            7, rewrite_request()
        )
    rules = [r.rule for r in caplog.records if getattr(r, "event", None) == "ai_story_rewrite"]
    assert rules == ["memo_missing"]


@pytest.mark.parametrize(
    ("summary", "ok"),
    [
        ("가" * 20, True),
        ("가" * 21, False),
        ("🌅" * 20, True),
        ("\U0001f468\u200d\U0001f469\u200d\U0001f467" * 4, True),
        ("\U0001f468\u200d\U0001f469\u200d\U0001f467" * 5, False),
    ],
)
@scenario
async def test_summary_limit_counts_code_points(summary, ok):
    vlm = FakeVlm(replies=[draft(summary=summary)])
    if ok:
        assert (await service(vlm).rewrite(7, rewrite_request())).story_summary == summary
    else:
        with pytest.raises(AppError):
            await service(vlm).rewrite(7, rewrite_request())
    assert len(vlm.calls) == 1
