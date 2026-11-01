"""정리 에이전트(#153): 가짜 VLM으로 그래프 흐름, 실패 처리, 결과 형식을 확인한다."""

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import ErrorCode, PipelineCancelled, PipelineError
from app.infra.kakao_local import KakaoPlace, KakaoRegion
from app.infra.qdrant import QdrantStore
from app.infra.vllm_client import _wire_messages
from app.infra.vlm import ChatTurn, ToolCall, VLMTransientError, VLMUnavailable
from app.pipeline.agent import (
    MAX_ROUNDS,
    AgentResources,
    AgentTimeout,
    AgentUnavailable,
    model,
    prompts,
    run_agent,
)
from app.pipeline.agent.trip import TripView, region_key
from app.pipeline.bootstrap import build_context
from app.pipeline.run import run
from app.pipeline.state import TimeStatus, build_photo_states
from app.schemas.process import Issue, Period, ProcessAttachment, ProcessRequest, RegionOrigin
from app.schemas.task import ProcessStep
from tests.test_process_handler import Engine, reporter
from tests.test_v2_jobs import process_request, scenario

BASE = datetime(2026, 10, 2, 10, 0, tzinfo=UTC)
HOME = (37.38, 128.66)
TRIP_REGION = KakaoRegion("강원특별자치도", "정선군")


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def LOW(messages):
    return call("decide", decision="existing", place_id="p1", confidence="low", reason="r")


def call(name, **arguments):
    tool = ToolCall(id=f"call-{name}", name=name, arguments=json.dumps(arguments))
    return ChatTurn(content="", tool_calls=[tool], finish_reason="tool_calls")


def target_of(messages):
    text = "".join(p["text"] for p in messages[1]["content"] if p["type"] == "text")
    return re.search(r"대상 사진 (S\d+)", text).group(1)


def high_ids(messages):
    text = messages[1]["content"][0]["text"]
    block = text.split("확신도 high 제안:\n", 1)[1].split("\n\n", 1)[0]
    return [json.loads(line)["photo_id"] for line in block.splitlines()]


class ScriptedVlm:
    """지시문으로 단계를 가려 정해진 답을 돌려준다. 이미지 자리와 URL 수가 맞는지도 확인한다."""

    def __init__(self, estimate, aggregate=None, merge=None):
        self.estimate = estimate
        self.aggregate = aggregate or (lambda messages: call("finalize", accept=high_ids(messages)))
        self.merge = merge or (lambda messages: call("merge", groups=[]))
        self.calls = []

    async def chat(self, messages, image_urls, max_tokens, response_format=None, **options):
        _wire_messages(messages, image_urls)
        assert len(image_urls) <= prompts.MAX_IMAGES
        assert options["temperature"] == 0 and options["tools"]
        stage = {
            prompts.ESTIMATE_SYSTEM: "estimate",
            prompts.AGGREGATE_SYSTEM: "aggregate",
            prompts.MERGE_SYSTEM: "merge",
        }[messages[0]["content"]]
        self.calls.append((stage, messages, list(image_urls)))
        await asyncio.sleep(0)
        return getattr(self, stage)(messages)


class FakeKakao:
    def __init__(self, regions=None, found=()):
        self.regions = regions or {}
        self.found = list(found)
        self.region_calls = 0

    async def region(self, latitude, longitude, budget):
        if not budget.take():
            return None
        self.region_calls += 1
        return self.regions.get((round(latitude, 2), round(longitude, 2)), TRIP_REGION)

    async def keyword(self, query, latitude, longitude, radius_m, budget, *, size=10):
        if not budget.take():
            return None
        return self.found


class FakeUrls:
    async def resolve(self, storage_key):
        return f"https://images.test/{storage_key}"


class FakePlaces:
    def search(self, query, latitude, longitude, radius_m, limit=10):
        return []


@pytest.fixture
def loop():
    """작업의 이벤트 루프. 파이프라인 스레드가 run_coroutine_threadsafe로 코루틴을 넘긴다."""
    running = asyncio.new_event_loop()
    thread = threading.Thread(target=running.run_forever, daemon=True)
    thread.start()
    yield running
    running.call_soon_threadsafe(running.stop)
    thread.join(5)
    running.close()


def resources(loop, vlm, kakao=None, deadline=None, **fields):
    return AgentResources(
        vlm=vlm,
        urls=FakeUrls(),
        places=FakePlaces(),
        kakao=kakao or FakeKakao(),
        submit=lambda coroutine: asyncio.run_coroutine_threadsafe(coroutine, loop),
        deadline=time.monotonic() + 60 if deadline is None else deadline,
        **fields,
    )


def scene(rows):
    """CLUSTERING 규칙 단계를 마친 상태. rows는 (ID, GPS, 장소, 분, 벡터)"""
    attachments = [
        ProcessAttachment(
            trip_attachment_id=photo_id,
            analyze_storage_key=f"trip/{photo_id}.jpg",
            taken_at=BASE + timedelta(minutes=minute) if minute is not None else None,
            latitude=gps[0] if gps else None,
            longitude=gps[1] if gps else None,
        )
        for photo_id, gps, _, minute, _ in rows
    ]
    request = ProcessRequest(
        trip_name="정선 여행",
        period=Period(start_date=date(2026, 10, 2), end_date=date(2026, 10, 4)),
        regions=[{"latitude": HOME[0], "longitude": HOME[1]}],
        attachments=attachments,
    )
    photos = build_photo_states(request)
    for photo, (_, _, place, _, vector) in zip(photos, rows, strict=True):
        photo.taken_at = photo.source.taken_at
        photo.time_status = TimeStatus.ORIGINAL if photo.taken_at else TimeStatus.UNKNOWN
        photo.embedding = unit(*vector)
        if place:
            photo.place_id = place
        else:
            photo.issue = Issue.UNCLEAR_LOCATION
    return request, photos


def context(agent, photos):
    qdrant = QdrantStore(":memory:", "agent-test", len(photos[0].embedding))
    ids = [p.source.trip_attachment_id for p in photos]
    qdrant.upsert(ids, np.stack([p.embedding for p in photos]))
    return SimpleNamespace(
        agent=agent,
        qdrant=qdrant,
        thresholds={"agent_same_photo_min_sim": 0.95},
        log=logging.getLogger("test.agent"),
    )


def by_id(photos):
    return {p.source.trip_attachment_id: p for p in photos}


def run_scene(loop, vlm, rows, kakao=None, report=None, check_cancel=lambda: None, **fields):
    request, photos = scene(rows)
    ctx = context(resources(loop, vlm, kakao, **fields), photos)
    summary = run_agent(photos, request, ctx, report or (lambda done: None), check_cancel)
    return photos, summary


TWO_PLACES = [
    (1, (37.380, 128.660), "p1", 0, (1, 0, 0, 0)),
    (2, (37.381, 128.661), "p1", 5, (0.9, 0.1, 0, 0)),
    (3, (37.300, 128.500), "p2", 60, (0, 1, 0, 0)),
    (4, None, None, None, (0.95, 0.05, 0, 0)),  # p1과 닮은, 시각 없는 사진
    (5, None, None, None, (0, 0, 1, 0)),
]


def test_high_existing_gets_folder_coordinates_and_medium_stays(loop):
    def estimate(messages):
        target = target_of(messages)
        if target == "S4":
            return call(
                "decide", decision="existing", place_id="p1", confidence="high", reason="간판"
            )
        return call(
            "decide", decision="existing", place_id="p2", confidence="medium", reason="비슷함"
        )

    vlm = ScriptedVlm(estimate)
    done = []
    photos, summary = run_scene(loop, vlm, TWO_PLACES, report=done.append)
    photo = by_id(photos)
    assert photo[4].place_id == "p1" and photo[4].issue is None
    assert photo[4].region_origin is RegionOrigin.INFERRED
    # 그 폴더에서 가장 닮은 GPS 사진(1번)의 좌표를 받는다(결정 150).
    assert (photo[4].latitude, photo[4].longitude) == (37.380, 128.660)
    assert photo[5].place_id is None and photo[5].issue is Issue.UNCLEAR_LOCATION
    assert summary["assigned_existing"] == 1 and summary["remaining_targets"] == 1
    stages = [stage for stage, _, _ in vlm.calls]
    # 1차례(두 장) → 종합 → 2차례(한 장, high 없음, 종합 생략) → 묶기 → 마지막 추정
    assert stages == ["estimate"] * 2 + ["aggregate", "estimate", "merge", "estimate"]
    # 호출이 끝날 때마다 1씩 오른다. total 직전 상한은 run()이 건다.
    assert done == list(range(1, 8))


def test_loop_stops_after_five_rounds_then_merge_and_final(loop):
    rows = [(1, HOME, "p1", 0, (1, 0, 0))] + [
        (i, None, None, None, (1, 0.01 * i, 0)) for i in range(10, 17)
    ]

    def estimate(messages):
        # 남은 사진 가운데 가장 작은 ID 하나만 high로 낸다. 차례마다 바뀌는 사진이 생긴다.
        target = target_of(messages)
        first = min(t for t in remaining if t)
        confidence = "high" if target == first else "low"
        return call("decide", decision="existing", place_id="p1", confidence=confidence, reason="r")

    remaining = {f"S{i}" for i in range(10, 17)}

    def aggregate(messages):
        accepted = high_ids(messages)
        remaining.difference_update(accepted)
        return call("finalize", accept=accepted)

    vlm = ScriptedVlm(lambda m: estimate(m), aggregate)
    photos, summary = run_scene(loop, vlm, rows)
    stages = [stage for stage, _, _ in vlm.calls]
    assert stages.count("aggregate") == MAX_ROUNDS + 1
    assert stages.count("merge") == 0  # 폴더가 하나라 묶기 호출은 없다
    assert summary["rounds"] == MAX_ROUNDS + 1
    assert summary["assigned_existing"] == MAX_ROUNDS + 1 and summary["remaining_targets"] == 1


def test_merge_applies_only_high_groups_and_renumbers(loop):
    rows = [
        (1, HOME, "p1", 0, (1, 0, 0)),
        (2, (37.39, 128.67), "p2", 10, (0.9, 0.1, 0)),
        (3, (37.20, 128.40), "p3", 120, (0, 1, 0)),
        (4, (37.21, 128.41), "p4", 130, (0, 0.9, 0.1)),
    ]
    merges = [
        {"place_ids": ["p1", "p2"], "confidence": "high", "reason": "같은 건물"},
        {"place_ids": ["p3", "p4"], "confidence": "medium", "reason": "가까움"},
    ]
    vlm = ScriptedVlm(lambda m: None, merge=lambda m: call("merge", groups=merges))
    photos, summary = run_scene(loop, vlm, rows)
    places = {p.source.trip_attachment_id: p.place_id for p in photos}
    assert places == {1: "p1", 2: "p1", 3: "p2", 4: "p3"}
    assert summary["merged_place_count"] == 1
    merge_input = vlm.calls[0][1][1]["content"][0]["text"]
    assert '"center_distance_m"' in merge_input and '"time_gap_min"' in merge_input


def test_new_place_in_region_is_created_and_outside_is_rejected(loop):
    inside = KakaoPlace("kakao:1", "병방치 스카이워크", "관광", "", 37.40, 128.62, 900)
    outside = KakaoPlace("kakao:2", "다른 곳", "관광", "", 35.10, 129.00, 9000)
    kakao = FakeKakao(
        regions={(35.10, 129.00): KakaoRegion("부산광역시", "해운대구")}, found=[inside, outside]
    )
    rows = [
        (1, HOME, "p1", 0, (1, 0, 0)),
        (7, None, None, None, (0, 1, 0)),
        (8, None, None, None, (0, 0, 1)),
    ]

    def estimate(messages):
        searched = any(m["role"] == "tool" for m in messages)
        if not searched:
            return call("kakao_keyword_search", query="스카이워크", near="trip")
        candidate = "kakao:1" if target_of(messages) == "S7" else "kakao:2"
        return call(
            "decide",
            decision="new",
            candidate_id=candidate,
            new_place_name="새",
            confidence="high",
            reason="간판",
        )

    photos, summary = run_scene(loop, ScriptedVlm(estimate), rows, kakao)
    photo = by_id(photos)
    assert photo[7].place_id == "p2" and photo[7].region_origin is RegionOrigin.INFERRED
    assert (photo[7].latitude, photo[7].longitude) == (37.40, 128.62)
    assert photo[8].place_id is None
    assert summary["assigned_new"] == 1 and summary["new_place_count"] == 1
    assert summary["regions"] == ["강원특별자치도 정선군"]


def test_estimate_gets_trip_info_and_actual_neighbors(loop):
    rows = [
        (1, HOME, "p1", 0, (1, 0, 0)),
        (2, None, None, 10, (0, 1, 0)),  # 장소 없는 바로 앞 사진
        (3, None, None, 20, (0, 0, 1)),
        (4, HOME, "p1", 40, (1, 0.1, 0)),
    ]
    vlm = ScriptedVlm(LOW)
    run_scene(loop, vlm, rows)
    messages, urls = next(
        (m, u) for s, m, u in vlm.calls if s == "estimate" and target_of(m) == "S3"
    )
    text = "".join(p["text"] for p in messages[1]["content"] if p["type"] == "text")
    assert '"name": "정선 여행"' in text and "2026-10-02 ~ 2026-10-04" in text
    assert "바로 앞인 사진 S2: 장소 미분류" in text and "바로 뒤인 사진 S4: 장소 p1" in text
    assert urls[:3] == [f"https://images.test/trip/{i}.jpg" for i in (3, 2, 4)]


def test_vlm_failure_fails_the_agent(loop):
    async def broken(*args, **kwargs):
        raise VLMUnavailable("down")

    vlm = SimpleNamespace(chat=broken)
    with pytest.raises(AgentUnavailable) as error:
        run_scene(loop, vlm, TWO_PLACES)
    assert error.value.code is ErrorCode.MODEL_NOT_READY


def test_vlm_timeout_and_budget_fail_with_timeout(loop):
    async def slow(*args, **kwargs):
        raise TimeoutError()

    # 호출 시간 초과의 재시도는 아래 재시도 절에서 확인한다.
    with pytest.raises(AgentTimeout) as error:
        run_scene(loop, SimpleNamespace(chat=slow), TWO_PLACES, vlm_retries=0)
    assert error.value.code is ErrorCode.TIMEOUT
    vlm = ScriptedVlm(LOW)
    with pytest.raises(AgentTimeout):
        run_scene(loop, vlm, TWO_PLACES, deadline=time.monotonic() - 1)
    assert vlm.calls == []


def test_cancel_stops_between_calls(loop):
    def cancel():
        raise PipelineCancelled()

    vlm = ScriptedVlm(LOW)
    with pytest.raises(PipelineCancelled):
        run_scene(loop, vlm, TWO_PLACES, check_cancel=cancel)
    assert vlm.calls == []


def test_tools_respect_image_limit_and_missing_final(loop):
    def estimate(messages):
        # 계속 사진만 읽고 최종 답을 내지 않는다.
        return call("read_photo", photo_ids=["S1", "S2", "S3", "S1"])

    vlm = ScriptedVlm(estimate)
    photos, summary = run_scene(loop, vlm, TWO_PLACES)
    estimates = [(m, u) for s, m, u in vlm.calls if s == "estimate"]
    assert max(len(u) for _, u in estimates) <= prompts.MAX_IMAGES
    assert summary["no_final_count"] >= 2 and summary["assigned_existing"] == 0
    first = [len(u) for m, u in estimates if target_of(m) == "S4"][: prompts.MAX_CALLS]
    assert len(first) == prompts.MAX_CALLS


def test_skips_without_targets_and_with_one_folder(loop):
    vlm = ScriptedVlm(lambda m: None)
    photos, summary = run_scene(loop, vlm, [(1, HOME, "p1", 0, (1, 0))])
    assert summary["result"] == "skipped" and vlm.calls == []


def test_region_key_compares_city_not_district():
    assert region_key("경기도", "성남시 분당구") == region_key("경기도", "성남시 중원구")
    assert region_key("경기도", "성남시") != region_key("경기도", "용인시")


def test_trip_view_ignores_photos_outside_period():
    request, photos = scene(TWO_PLACES + [(9, None, None, 60 * 24 * 5, (1, 0, 0, 0))])
    view = TripView(photos, request, regions=[])
    assert [p.source.trip_attachment_id for p in view.targets()] == [4, 5]


def test_qdrant_search_is_limited_to_candidates():
    store = QdrantStore(":memory:", "search-test", 2)
    store.upsert([1, 2, 3], np.stack([unit(1, 0), unit(1, 0.1), unit(1, 0.05)]))
    hits = store.search(unit(1, 0), [2, 3], 5)
    assert [i for i, _ in hits] == [3, 2]
    assert store.search(unit(1, 0), [], 5) == []


# ---------- run()과 결과 형식
def test_run_assigns_with_agent_and_result_passes_contract(tmp_path, loop):
    rng = np.random.default_rng(7)
    attachments = []
    for i in range(4):
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            tmp_path / f"{i}.png"
        )
        gps = i < 2
        attachments.append(
            dict(
                trip_attachment_id=200 + i,
                analyze_storage_key=f"{i}.png",
                taken_at=None if i == 3 else (BASE + timedelta(minutes=i * 90)).isoformat(),
                latitude=HOME[0] + i * 0.05 if gps else None,
                longitude=HOME[1] if gps else None,
            )
        )
    request = ProcessRequest.model_validate(
        dict(
            trip_name="정선 여행",
            period={"start_date": "2026-10-02", "end_date": "2026-10-04"},
            regions=[{"latitude": HOME[0], "longitude": HOME[1]}],
            attachments=attachments,
        )
    )
    settings = Settings(
        _env_file=None,
        API_KEY="t",
        IMAGE_SOURCE="local",
        IMAGE_DIR=tmp_path,
        QDRANT_URL=":memory:",
        FAKE_PIPELINE=False,
    )

    class Engine:
        model_version, dim = "siglip2-so400m-naflex", 3
        vectors = [unit(1, 0, 0), unit(0, 1, 0), unit(0.1, 0, 1), unit(0, 0.2, 1)]

        def encode_images(self, images):
            return np.stack(self.vectors[: len(images)])

    def estimate(messages):
        target = target_of(messages)
        if target == "S202":
            return call("decide", decision="existing", place_id="p2", confidence="high", reason="r")
        return call("decide", decision="existing", place_id="p1", confidence="medium", reason="r")

    vlm = ScriptedVlm(estimate)
    ctx = replace(build_context(settings, Engine()), agent=resources(loop, vlm))
    ctx.thresholds.update(blur_var_min=0, phash_max_dist=0, inherit_min_sim=0.99)
    steps = []
    result = run(request, ctx, lambda step, done, total: steps.append((step, done)), lambda: False)
    placed = {a.trip_attachment_id: (p.place_id, a) for p in result.places for a in p.attachments}
    assert placed[202][0] == placed[201][0]
    assert placed[202][1].region_origin is RegionOrigin.INFERRED
    assert [u.trip_attachment_id for u in result.unclassified] == [203]
    clustering = [done for step, done in steps if step is ProcessStep.CLUSTERING]
    assert clustering[0] == 0 and clustering[-1] == 4 and 0 < max(clustering[1:-1]) < 4


# ---------- 정리 실행기 연결
@pytest.fixture
def configured(tmp_path, monkeypatch):
    """tests/test_process_handler.py와 같은 정리 실행기 설정. 에이전트는 기본으로 끈다."""
    from app.embedding.client import MODEL_VERSION
    from app.jobs import process_handler
    from app.jobs.config import JobsSettings

    photo = tmp_path / "trips/77/a.jpg"
    photo.parent.mkdir(parents=True)
    noise = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(noise).save(photo)
    cfg = JobsSettings(
        _env_file=None,
        API_KEY="test",
        IMAGE_SOURCE="local",
        IMAGE_DIR=tmp_path,
        QDRANT_URL=":memory:",
        EMBED_MODEL=MODEL_VERSION,
        PROCESS_AGENT_ENABLED=False,
    )
    engine = Engine()
    monkeypatch.setattr(process_handler, "JobsSettings", lambda: cfg)
    monkeypatch.setattr(process_handler, "build_engine", lambda _: engine)
    return cfg, engine


class _Opened:
    """async with로 여는 클라이언트 자리. 열면 미리 만든 가짜를 돌려준다."""

    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc_info):
        return None


def agent_settings(cfg, monkeypatch, vlm, kakao):
    from pydantic import SecretStr

    from app.jobs import process_handler

    for name, value in {
        "PROCESS_AGENT_ENABLED": True,
        "VLM_BASE_URL": "http://vlm.test/v1",
        "VLM_MODEL": "gemma-4",
        "PLACE_INDEX_PATH": "/tmp/places.sqlite",
        "KAKAO_REST_API_KEY": SecretStr("kakao-key"),
    }.items():
        setattr(cfg, name, value)
    places = FakePlaces()
    places.close = lambda: None
    monkeypatch.setattr(process_handler, "PlaceIndex", lambda path: places)
    monkeypatch.setattr(process_handler, "VllmClient", _Opened(vlm))
    monkeypatch.setattr(process_handler, "KakaoLocalClient", _Opened(kakao))
    monkeypatch.setattr(process_handler, "RequestGate", lambda *args: None)
    monkeypatch.setattr(process_handler, "GatedVlm", lambda client, gate, timeout: client)


@scenario
async def test_process_handler_runs_agent_and_creates_new_place(configured, monkeypatch):
    from app.jobs import process_handler

    cfg, _ = configured
    found = KakaoPlace("kakao:9", "협재 해수욕장", "관광", "", 33.39, 126.24, 100)

    def estimate(messages):
        if not any(m["role"] == "tool" for m in messages):
            return call("kakao_keyword_search", query="해수욕장", near="trip")
        return call(
            "decide",
            decision="new",
            candidate_id="kakao:9",
            new_place_name="협재 해수욕장",
            confidence="high",
            reason="표지판",
        )

    vlm = ScriptedVlm(estimate)
    agent_settings(cfg, monkeypatch, vlm, FakeKakao(found=[found]))
    r = reporter()
    result = await process_handler.run(process_request(), r)
    (place,) = result["places"]
    (photo,) = place["attachments"]
    assert photo["trip_attachment_id"] == 101 and photo["region_origin"] == "INFERRED"
    assert (photo["latitude"], photo["longitude"]) == (33.39, 126.24)
    assert result["unclassified"] == []
    # 로컬 사진 폴더는 data URL로 넘긴다.
    assert vlm.calls[0][2][0].startswith("data:image/jpeg;base64,")
    steps = [c.args[0] for c in r.report.call_args_list]
    assert "CLUSTERING" in steps and steps[-1] == "FINALIZING"


@scenario
async def test_process_handler_fails_when_agent_settings_are_missing(configured):
    from app.jobs import process_handler

    cfg, engine = configured
    cfg.PROCESS_AGENT_ENABLED = True
    with pytest.raises(PipelineError) as error:
        await process_handler.run(process_request(), reporter())
    assert error.value.code == ErrorCode.MODEL_NOT_READY and not engine.batches


@scenario
async def test_process_handler_without_agent_finishes_with_rule_result(configured, monkeypatch):
    from app.jobs import process_handler

    cfg, _ = configured
    seen = []
    build = process_handler.build_context

    def capture(*args, **kwargs):
        ctx = build(*args, **kwargs)
        seen.append(ctx.agent)
        return ctx

    monkeypatch.setattr(process_handler, "build_context", capture)
    result = await process_handler.run(process_request(), reporter())
    assert seen == [None] and result["unclassified"][0]["trip_attachment_id"] == 101


@pytest.mark.parametrize(
    "fields",
    [
        {"PROCESS_AGENT_BUDGET_SEC": 3000, "JOBS_PROCESS_TIMEOUT_SEC": 3000},
        {"PROCESS_AGENT_CONCURRENCY": 9, "GPU_GENERATION_INFLIGHT": 8},
    ],
)
def test_agent_settings_validation(fields):
    from pydantic import ValidationError

    from app.jobs.config import JobsSettings

    with pytest.raises(ValidationError):
        JobsSettings(_env_file=None, API_KEY="t", IMAGE_SOURCE="local", **fields)


def test_job_timeout_is_separate_for_process():
    from app.jobs.config import JobsSettings

    cfg = JobsSettings(_env_file=None, API_KEY="t", IMAGE_SOURCE="local")
    assert cfg.job_timeout("process") == cfg.JOBS_PROCESS_TIMEOUT_SEC == 3000
    assert cfg.job_timeout("story") == cfg.JOBS_TIMEOUT_SEC == 300


# ---------- 독립 검토 지적 재현
class SlotFullVlm:
    """GatedVlm이 슬롯을 얻지 못한 경우처럼 원인 사슬에 TimeoutError를 남기고 실패한다."""

    async def chat(self, *args, **kwargs):
        try:
            raise TimeoutError
        except TimeoutError:
            raise VLMUnavailable("생성 서버의 호출 대기 한도를 초과했습니다.") from None


@scenario
async def test_process_handler_keeps_agent_error_code_over_slot_timeout(configured, monkeypatch):
    from app.jobs import process_handler

    cfg, _ = configured
    agent_settings(cfg, monkeypatch, SlotFullVlm(), FakeKakao())
    with pytest.raises(PipelineError) as error:
        await process_handler.run(process_request(), reporter())
    assert error.value.code == ErrorCode.MODEL_NOT_READY


@scenario
async def test_process_handler_rejects_malformed_agent_settings(configured, monkeypatch):
    from app.jobs import process_handler

    cfg, _ = configured
    agent_settings(cfg, monkeypatch, ScriptedVlm(LOW), FakeKakao())

    def bad(*args, **kwargs):
        raise ValueError("bad url")

    monkeypatch.setattr(process_handler, "VllmClient", bad)
    with pytest.raises(PipelineError) as error:
        await process_handler.run(process_request(), reporter())
    assert error.value.code == ErrorCode.MODEL_NOT_READY


def test_search_anchor_ignores_non_string_near():
    request, photos = scene(TWO_PLACES)
    view = TripView(photos, request, regions=[])
    assert view.anchor(5770) == view.anchor(["S1"]) == view.trip_center()
    assert view.anchor("S1") == (37.380, 128.660)


def test_neighbors_with_same_time_are_distinct():
    rows = [
        (1, HOME, "p1", 0, (1, 0)),
        (2, None, None, 10, (0, 1)),
        (3, None, None, 10, (1, 1)),
        (4, HOME, "p1", 20, (1, 0)),
    ]
    request, photos = scene(rows)
    view = TripView(photos, request, regions=[])
    before, after = view.neighbors(by_id(photos)[2])
    assert (before.source.trip_attachment_id, after.source.trip_attachment_id) == (1, 3)
    before, after = view.neighbors(by_id(photos)[3])
    assert (before.source.trip_attachment_id, after.source.trip_attachment_id) == (2, 4)


def test_same_candidate_reuses_new_place_across_rounds():
    request, photos = scene(TWO_PLACES)
    view = TripView(photos, request, regions=[])
    first = view.add_new_place("협재", 33.39, 126.24, "kakao:9")
    view.assign(by_id(photos)[4], first)
    assert view.add_new_place("협재 해변", 33.39, 126.24, "kakao:9") == first
    assert len(view.new_places) == 1


def test_finish_keeps_cluster_numbers_and_appends_new_places():
    rows = [
        (1, None, "p2", 0, (1, 0)),  # 규칙 단계가 p2에 넣은, ID가 가장 작은 사진
        (5, HOME, "p1", 5, (0, 1)),
        (6, (37.39, 128.67), "p2", 10, (1, 0)),
        (7, None, None, None, (1, 1)),
    ]
    request, photos = scene(rows)
    view = TripView(photos, request, regions=[])
    view.finish()
    assert {p.source.trip_attachment_id: p.place_id for p in photos if p.place_id} == {
        1: "p2",
        5: "p1",
        6: "p2",
    }
    view.assign(by_id(photos)[7], view.add_new_place("새 곳", 37.4, 128.7, "kakao:1"))
    view.finish()
    assert by_id(photos)[7].place_id == "p3"


def test_disabled_agent_accepts_previous_timeout_and_slots():
    from app.jobs.config import JobsSettings

    cfg = JobsSettings(
        _env_file=None,
        API_KEY="t",
        IMAGE_SOURCE="local",
        PROCESS_AGENT_ENABLED=False,
        JOBS_PROCESS_TIMEOUT_SEC=300,
        GPU_GENERATION_INFLIGHT=2,
    )
    assert cfg.job_timeout("process") == 300


def test_process_handler_import_does_not_load_langgraph():
    import subprocess
    import sys
    from pathlib import Path

    code = "import sys, app.jobs.process_handler; print('langgraph' in sys.modules)"
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=root
    ).stdout
    assert out.strip() == "False"


# ---------- 생성 호출 재시도(결정 159)
ONE_TARGET = TWO_PLACES[:4]  # 대상이 한 장이라 호출이 차례로만 나간다


class FlakyVlm(ScriptedVlm):
    """앞의 호출은 faults를 차례로 내며 실패하고, 그 뒤에는 ScriptedVlm처럼 답한다."""

    def __init__(self, fault, failures, estimate=LOW):
        super().__init__(estimate)
        self.faults = [fault] * failures if not isinstance(fault, list) else fault
        self.attempts = 0
        self.sent = []  # 시도마다 보낸 (메시지 수, 사진 URL)

    async def chat(self, messages, image_urls, max_tokens, response_format=None, **options):
        self.attempts += 1
        self.sent.append((len(messages), list(image_urls)))
        if self.attempts <= len(self.faults):
            raise self.faults[self.attempts - 1]
        return await super().chat(messages, image_urls, max_tokens, response_format, **options)


class CountingUrls(FakeUrls):
    def __init__(self):
        self.resolved = 0

    async def resolve(self, storage_key):
        self.resolved += 1
        return await super().resolve(storage_key)


@pytest.fixture
def waits(monkeypatch):
    """재시도 대기 시간을 기록하고 실제로는 기다리지 않는다."""
    recorded, real_sleep = [], asyncio.sleep

    async def sleep(seconds, *args, **kwargs):
        if seconds:
            recorded.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return recorded


@pytest.mark.parametrize("fault", [VLMTransientError("503"), TimeoutError()])
def test_transient_vlm_errors_are_retried_with_growing_waits(loop, waits, caplog, fault):
    vlm = FlakyVlm(fault, failures=3)
    with caplog.at_level(logging.INFO, logger="test.agent"):
        _, summary = run_scene(loop, vlm, ONE_TARGET)
    assert waits == [5.0, 10.0, 15.0]
    assert summary["result"] == "completed" and summary["model_retries"] == 3
    stages = [r for r in caplog.records if getattr(r, "event", None) == "ai_agent_stage"]
    assert sum(r.model_retries for r in stages) == 3
    # 실패한 시도도 호출 수에 들어간다.
    assert summary["model_calls"] == len(vlm.calls) + 3
    # 다시 보낼 때도 같은 대화와 사진을 보낸다.
    assert vlm.sent[:4] == [vlm.sent[3]] * 4 and vlm.sent[3][1] == vlm.calls[0][2]
    assert vlm.calls[0][2][0] == "https://images.test/trip/4.jpg"


@pytest.mark.parametrize(
    "fault, failure, code",
    [
        (VLMTransientError("503"), AgentUnavailable, ErrorCode.MODEL_NOT_READY),
        (TimeoutError(), AgentTimeout, ErrorCode.TIMEOUT),
    ],
)
@pytest.mark.parametrize("retries", [0, 2])
def test_retryable_vlm_errors_fail_after_retries(loop, waits, retries, fault, failure, code):
    vlm = FlakyVlm(fault, failures=99)
    with pytest.raises(failure) as error:
        run_scene(loop, vlm, ONE_TARGET, vlm_retries=retries)
    assert error.value.code is code
    assert vlm.attempts == retries + 1 and waits == [5.0 * n for n in range(1, retries + 1)]


@pytest.mark.parametrize(
    "fault",
    [
        VLMUnavailable("완성된 응답이 아님"),
        VLMUnavailable("생성 서버의 호출 대기 한도를 초과했습니다."),
    ],
)
def test_incomplete_output_and_slot_wait_are_not_retried(loop, waits, fault):
    vlm = FlakyVlm(fault, failures=99)
    with pytest.raises(AgentUnavailable):
        run_scene(loop, vlm, ONE_TARGET)
    assert vlm.attempts == 1 and waits == []


def test_mixed_retryable_errors_report_the_last_error(loop, waits):
    vlm = FlakyVlm([TimeoutError(), TimeoutError(), TimeoutError(), VLMTransientError("503")], 0)
    with pytest.raises(AgentUnavailable) as error:
        run_scene(loop, vlm, ONE_TARGET)
    assert error.value.code is ErrorCode.MODEL_NOT_READY and vlm.attempts == 4


def test_retry_stops_at_first_non_retryable_error(loop, waits):
    vlm = FlakyVlm([VLMTransientError("503"), VLMUnavailable("완성된 응답이 아님")], failures=0)
    with pytest.raises(AgentUnavailable):
        run_scene(loop, vlm, ONE_TARGET)
    assert vlm.attempts == 2 and waits == [5.0]


def test_retry_resolves_photo_urls_again(loop, waits):
    urls = CountingUrls()
    request, photos = scene(ONE_TARGET)
    agent = replace(resources(loop, FlakyVlm(TimeoutError(), failures=1)), urls=urls)
    ctx = context(agent, photos)
    run_agent(photos, request, ctx, lambda done: None, lambda: None)
    vlm = agent.vlm
    # 실패한 시도의 사진 수만큼 한 번 더 받는다.
    assert urls.resolved == sum(len(sent) for _, sent in vlm.sent)
    assert len(vlm.sent[0][1]) > 0 and vlm.sent[0] == vlm.sent[1]


@pytest.mark.parametrize("retries, succeeds", [(1, True), (0, False)])
@scenario
async def test_process_handler_passes_retry_setting(configured, monkeypatch, retries, succeeds):
    from app.jobs import process_handler

    cfg, _ = configured
    vlm = FlakyVlm(VLMTransientError("503"), failures=1)
    agent_settings(cfg, monkeypatch, vlm, FakeKakao())
    cfg.PROCESS_VLM_RETRIES = retries
    monkeypatch.setattr(model, "RETRY_WAIT_SEC", 0)
    if succeeds:
        await process_handler.run(process_request(), reporter())
        assert vlm.attempts == len(vlm.calls) + 1
    else:
        with pytest.raises(PipelineError) as error:
            await process_handler.run(process_request(), reporter())
        assert error.value.code == ErrorCode.MODEL_NOT_READY and vlm.attempts == 1
