"""워크플로우 정리: 가짜 모델과 가짜 카카오로 단계 규칙, 흐름, 실패와 결과 계약을 확인한다."""

import asyncio
import hashlib
import io
import json
import re
import threading
import time
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import httpx
import numpy as np
import pytest
from PIL import Image
from pydantic import SecretStr

from app.core.config import Settings
from app.core.errors import ErrorCode, PipelineCancelled, PipelineError
from app.infra.kakao_local import (
    KakaoBudget,
    KakaoLocalClient,
    KakaoPlace,
    KakaoRegion,
    KakaoUnavailable,
)
from app.infra.vllm_client import VllmClient, _wire_messages
from app.infra.vlm import ChatTurn, VLMTransientError, VLMUnavailable
from app.pipeline.bootstrap import build_context
from app.pipeline.run import run
from app.pipeline.state import build_photo_states
from app.pipeline.workflow import (
    WorkflowResources,
    WorkflowTimeout,
    WorkflowUnavailable,
    organize,
    prompts,
    steps,
)
from app.pipeline.workflow.client import Places, Pod
from app.pipeline.workflow.trip import Trip, clock, load_font, stage2
from app.schemas.process import Issue, ProcessRequest, RegionOrigin
from app.schemas.task import ProcessStep
from tests.test_process_handler import configured, reporter  # noqa: F401
from tests.test_v2_jobs import process_request, scenario

BASE = datetime(2026, 10, 2, 1, 0, tzinfo=UTC)
HOME = (37.38, 128.66)
CAFE = KakaoPlace(
    "kakao:77", "정선 찻집", "음식점 > 카페", "강원 정선군 정선읍 1", 37.39, 128.67, None,
    address_name="강원 정선군 정선읍 1",
)  # fmt: skip
OTHER = KakaoPlace(
    "kakao:78", "서울 찻집", "음식점 > 카페", "서울 강남구 1", 37.5, 127.0, None,
    address_name="서울 강남구 1",
)  # fmt: skip


def unit(*values):
    vector = np.asarray(values, dtype=np.float64)
    return vector / np.linalg.norm(vector)


# ---------- 가짜 자원
class ScriptedVlm:
    """지시문 첫머리로 단계를 가려 답한다. 이미지 자리와 URL 수, 호출 옵션도 확인한다."""

    def __init__(self, answers=None, fail=None):
        self.answers = answers or {}
        self.fail = fail
        self.calls = []

    async def chat(self, messages, image_urls, max_tokens, response_format=None, **options):
        _wire_messages(messages, image_urls)
        assert options["temperature"] == 0 and options["stream"] is True
        text = messages[0]["content"][0]["text"]
        stage = stage_of(text)
        self.calls.append((stage, options.get("extra_body"), max_tokens, len(image_urls)))
        await asyncio.sleep(0)
        if self.fail:
            raise self.fail
        answer = self.answers.get(stage, DEFAULTS.get(stage, {}))
        return ChatTurn(content=json.dumps(answer(text) if callable(answer) else answer))


def stage_of(text):
    for prefix, stage in (
        ("여행 사진 정리를 돕는다. 아래 사진", "describe"),
        ("여행 사진을 장소별로 정리한다", "propose"),
        ("여행 사진 정리를 검토한다. 아래 격자 사진들은 한 장소로", "audit"),
        ("여행 사진 정리에서 이 장소 묶음의 장소 이름을 찾으려", "query"),
        ("여행 사진 정리에서 이 장소 묶음의 장소를 정한다", "choose"),
        ("여행 사진 정리에서 GPS 좌표로 나뉜", "facility"),
        ("아래는 여행 지역 안의 관광 정보 목록이다", "shortlist"),
        ("여행 사진 정리에서 이 장소 묶음이 어느 관광 장소인지 정한다", "imagepick"),
        ("여행 사진 정리를 검토한다. 아래 격자들은 이미 이름을 정한 장소 카드다", "vote"),
        ('여행 사진 정리를 검토한다. 아래 격자 사진들은 "', "purge"),
        ("여행 사진 정리를 검토한다. 첫 격자는 장소 묶음 X", "second"),
    ):
        if text.startswith(prefix):
            return stage
    raise AssertionError(f"모르는 지시문: {text[:40]}")


DEFAULTS = {
    "describe": {"place_type": "카페", "setting": "실내", "distinctive": "", "text": []},
    "propose": {"places": []},
    "query": {"queries": ["찻집"], "visible_name": "", "category": "카페"},
    "choose": {"choice": "none", "retry_queries": []},
    "facility": {"same_facility": False},
    "shortlist": {"candidates": []},
    "imagepick": {"choice": "none"},
    "vote": {"choice": "none"},
    "purge": {"not_here": []},
    "second": {"choice": "none"},
}


SEOUL = (37.5126, 127.1025)  # 지역 밖 예시(잠실)
PYEONGCHANG = (37.3708, 128.3902)
REGIONS = {
    (37.51, 127.1): KakaoRegion("서울특별시", "송파구"),
    (37.37, 128.39): KakaoRegion("강원특별자치도", "평창군"),
    (37.24, 127.18): KakaoRegion("경기도", "용인시 처인구"),
    (37.5, 127.0): KakaoRegion("서울특별시", "강남구"),
}


class FakeKakao:
    """좌표→행정구역은 REGIONS(소수 둘째 자리)에서 찾고 없으면 정선군. failing은 카카오 실패"""

    def __init__(self, found=(), failing=()):
        self.found = list(found)
        self.failing = set(failing)
        self.keyword_calls = []
        self.region_calls = []

    async def search_keyword(self, query, budget, *, center=None, radius_m=None, size=15, sort=""):
        if not budget.take():
            return None
        self.keyword_calls.append((query, center, radius_m, sort))
        return list(self.found)

    async def search_category(self, code, lat, lon, radius_m, budget, **options):
        if not budget.take():
            return None
        return [], True

    async def region(self, latitude, longitude, budget):
        if not budget.take():
            return None
        key = (round(latitude, 2), round(longitude, 2))
        self.region_calls.append(key)
        if key in self.failing:
            raise KakaoUnavailable("x")
        return REGIONS.get(key, KakaoRegion("강원특별자치도", "정선군"))


class FakePlaces:
    def search(self, query, latitude, longitude, radius_m, limit=10):
        return []


class FakeRemote:
    async def get(self, url):
        return None


@pytest.fixture
def loop():
    running = asyncio.new_event_loop()
    thread = threading.Thread(target=running.run_forever, daemon=True)
    thread.start()
    yield running
    running.call_soon_threadsafe(running.stop)
    thread.join(5)
    running.close()


def resources(loop, vlm, kakao=None, deadline=None, **fields):
    return WorkflowResources(
        vlm=vlm,
        kakao=kakao or FakeKakao(),
        places=FakePlaces(),
        tour=[],
        remote_images=FakeRemote(),
        submit=lambda coroutine: asyncio.run_coroutine_threadsafe(coroutine, loop),
        deadline=time.monotonic() + 60 if deadline is None else deadline,
        **fields,
    )


def NAME_ONLY(text):
    """GPS 장소 레코드 고르기(4단계)는 none, GPS 없는 묶음 이름 고르기는 첫 후보"""
    return {"choice": "none" if "GPS 좌표가 있다" in text else "1", "reason": "간판"}


# 사진: (ID, GPS, 분, 벡터, 흐림)
SCENE = [
    (1, HOME, 0, (1, 0, 0, 0, 0), False),
    (2, (HOME[0] + 0.0001, HOME[1]), 20, (1, 0.05, 0, 0, 0), False),
    (3, None, 10, (0, 1, 0, 0, 0), False),  # 앞뒤가 GPS 장소 사진이라 2단계에서 끼워 넣음
    (4, None, None, (0, 0, 1, 0, 0), False),  # 4·5는 한 그룹, 남은 묶음 이름 단계에서 카카오 레코드
    (5, None, None, (0, 0, 1, 0.05, 0), False),
    (6, None, None, (0, 0, 0, 0, 1), False),  # 한 장짜리라 이름 단계를 건너뛰어 미분류
]


def pipeline(tmp_path, rows, period=(date(2026, 10, 2), date(2026, 10, 4)), regions=(HOME,)):
    attachments = []
    for photo_id, gps, minute, _, blurry in rows:
        image = np.random.default_rng(photo_id).integers(0, 256, (48, 48, 3), dtype=np.uint8)
        if blurry:
            image = np.full((48, 48, 3), 128, dtype=np.uint8)
        Image.fromarray(image).save(tmp_path / f"{photo_id}.png")
        attachments.append(
            dict(
                trip_attachment_id=photo_id,
                analyze_storage_key=f"{photo_id}.png",
                taken_at=(BASE + timedelta(minutes=minute)).isoformat()
                if minute is not None
                else None,
                latitude=gps[0] if gps else None,
                longitude=gps[1] if gps else None,
                device_model="Phone",
            )
        )
    request = ProcessRequest.model_validate(
        dict(
            trip_name="정선 여행",
            period={"start_date": str(period[0]), "end_date": str(period[1])},
            regions=[{"latitude": a, "longitude": b} for a, b in regions],
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
        model_version, dim = "siglip2-so400m-naflex", 5
        vectors = [unit(*row[3]).astype(np.float32) for row in rows]
        done = 0

        def encode_images(self, images):
            out = self.vectors[self.done : self.done + len(images)]
            self.done += len(images)
            return np.stack(out)

    return request, settings, Engine()


def run_scene(
    tmp_path, loop, vlm, rows=SCENE, kakao=None, cancelled=lambda: False, regions=(HOME,), **fields
):
    request, settings, engine = pipeline(tmp_path, rows, regions=regions)
    ctx = replace(build_context(settings, engine), workflow=resources(loop, vlm, kakao, **fields))
    ctx.thresholds.update(blur_var_min=10, phash_max_dist=0)
    steps_ = []
    result = run(request, ctx, lambda step, done, total: steps_.append((step, done)), cancelled)
    return result, steps_


# ---------- 흐름
def test_workflow_places_gps_inserted_and_named_photos_and_result_passes_contract(tmp_path, loop):
    vlm = ScriptedVlm({"choose": NAME_ONLY})
    kakao = FakeKakao(found=[CAFE, OTHER])
    result, progress = run_scene(tmp_path, loop, vlm, kakao=kakao)
    placed = {a.trip_attachment_id: (p, a) for p in result.places for a in p.attachments}
    assert sorted(placed) == [1, 2, 3, 4, 5]
    assert placed[1][0].place_id == placed[2][0].place_id == placed[3][0].place_id
    assert placed[1][1].region_origin is RegionOrigin.EXIF
    # 끼워 넣은 사진은 폴더 GPS 사진 가운데 가장 닮은 사진의 좌표(결정 150)
    assert placed[3][1].region_origin is RegionOrigin.INFERRED
    # 사진 3의 벡터는 사진 2와만 내적이 0보다 크다
    assert (placed[3][1].latitude, placed[3][1].longitude) == (HOME[0] + 0.0001, HOME[1])
    # 레코드 폴더는 시·군 이름으로 거른 카카오 후보의 좌표
    record = placed[4][0]
    assert record.place_id == placed[5][0].place_id != placed[1][0].place_id
    assert (placed[4][1].latitude, placed[4][1].longitude) == (CAFE.latitude, CAFE.longitude)
    assert placed[4][1].region_origin is RegionOrigin.INFERRED
    # 장소 이름: 레코드 폴더는 고른 레코드 이름, 4단계에서 이름을 못 찾은 GPS 폴더는 null
    # (자리표시 이름 "GPS 장소 1"을 넘기지 않음)
    assert record.place_name == "정선 찻집"
    assert placed[1][0].place_name is None
    (left,) = result.unclassified
    assert left.trip_attachment_id == 6 and left.issue is Issue.UNCLEAR_LOCATION
    assert left.place_id is None and left.region_origin is RegionOrigin.UNKNOWN
    # 단계별 이미지 예산과 출력 상한(실험 값)
    budgets = {
        stage: ((extra or {}).get("mm_processor_kwargs") or {}).get("max_soft_tokens")
        for stage, extra, _, _ in vlm.calls
    }
    assert budgets["describe"] == 280 and budgets["propose"] == 1120
    assert budgets["query"] == 1120 and budgets["choose"] == 1120
    # 출력 상한: 그룹 설명 1024, 다수결 512(결정 224, 이전 400·200)
    max_tokens = {s: m for s, _, m, _ in vlm.calls}
    assert max_tokens["describe"] == 1024 and max_tokens["vote"] == 512
    # 진행률: CLUSTERING 시작 0, 호출마다 오르되 total 직전에서 멈추고, 끝에 total
    clustering = [done for step, done in progress if step is ProcessStep.CLUSTERING]
    assert clustering[0] == 0 and clustering[-1] == 6
    assert 0 < max(clustering[1:-1]) <= 5
    # 전국 검색과 여행 지역 좌표 20 km 검색
    assert ("찻집", None, None, "accuracy") in kakao.keyword_calls
    assert ("찻집", HOME, 20_000, "accuracy") in kakao.keyword_calls


def test_gps_folder_gets_name_found_in_gps_record_step(tmp_path, loop):
    """4단계에서 GPS 장소 레코드를 고르면 그 이름이 GPS 폴더의 place_name이 된다."""
    vlm = ScriptedVlm({"choose": {"choice": "1", "reason": "간판"}})
    result, _ = run_scene(tmp_path, loop, vlm, kakao=FakeKakao(found=[CAFE]))
    gps_place = next(p for p in result.places if 1 in {a.trip_attachment_id for a in p.attachments})
    assert gps_place.place_name == "정선 찻집"
    assert json.loads(result.model_dump_json())["places"][0]["place_name"] == "정선 찻집"


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (None, None),
        # 4단계에서 이름을 못 찾은 GPS 레코드는 자리표시 이름만 있다
        ({"id": "gps:0", "name": "GPS 장소 1", "gps": True}, None),
        ({"id": "gps:0", "name": "정선 찻집", "gps": True, "place_name": "정선 찻집"}, "정선 찻집"),
        ({"id": "kakao:77", "name": "  정선   5일장 \n"}, "정선 5일장"),
        ({"id": "kakao:77", "name": "   "}, None),
        ({"id": "kakao:77", "name": "가" * 60}, "가" * 50),
    ],
)
def test_record_place_name(record, expected):
    assert organize.record_place_name(record) == expected


def test_record_folder_with_only_blurry_photos_is_released(tmp_path, loop):
    rows = [row[:4] + (row[0] in (4, 5),) for row in SCENE]
    vlm = ScriptedVlm({"choose": NAME_ONLY})
    result, _ = run_scene(tmp_path, loop, vlm, rows=rows, kakao=FakeKakao(found=[CAFE]))
    assert sorted(a.trip_attachment_id for p in result.places for a in p.attachments) == [1, 2, 3]
    released = {u.trip_attachment_id: u for u in result.unclassified}
    assert released[4].issue in (Issue.BLURRY, Issue.DUPLICATED) and released[4].place_id is None
    assert released[4].region_origin is RegionOrigin.UNKNOWN


def test_cancel_during_workflow_stops_waiting_model_calls(tmp_path, loop, monkeypatch):
    monkeypatch.setattr(organize, "POLL_SEC", 0.01)
    monkeypatch.setattr(organize, "CANCEL_CHECK_SEC", 0.01)
    started, cancelled = threading.Event(), []

    class Blocking:
        async def chat(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

    with pytest.raises(PipelineCancelled):
        run_scene(tmp_path, loop, Blocking(), cancelled=lambda: started.is_set())
    deadline = time.monotonic() + 5
    while not cancelled and time.monotonic() < deadline:
        time.sleep(0.01)
    assert cancelled


def test_deadline_raises_workflow_timeout(tmp_path, loop, monkeypatch):
    monkeypatch.setattr(organize, "POLL_SEC", 0.01)

    class Slow:
        async def chat(self, *args, **kwargs):
            await asyncio.sleep(30)

    with pytest.raises(WorkflowTimeout) as caught:
        run_scene(tmp_path, loop, Slow(), deadline=time.monotonic() + 0.2)
    assert caught.value.code is ErrorCode.TIMEOUT


def test_missing_workflow_thresholds_fail(tmp_path, loop):
    request, settings, engine = pipeline(tmp_path, SCENE)
    ctx = replace(build_context(settings, engine), workflow=resources(loop, ScriptedVlm()))
    del ctx.thresholds["workflow_time_veto_min"]
    with pytest.raises(PipelineError, match="workflow_time_veto_min"):
        run(request, ctx, lambda *args: None, lambda: False)


# ---------- 2단계와 1단계 규칙
def trip_of(rows, seed=61106, regions=("강원특별자치도 정선군",)):
    request = ProcessRequest.model_validate(
        dict(
            trip_name="정선",
            period={"start_date": "2026-10-02", "end_date": "2026-10-03"},
            regions=[{"latitude": HOME[0], "longitude": HOME[1]}],
            attachments=[
                dict(
                    trip_attachment_id=i + 1,
                    analyze_storage_key=f"{i}.jpg",
                    taken_at=t.isoformat() if t else None,
                    latitude=gps[0] if gps else None,
                    longitude=gps[1] if gps else None,
                    device_model=dev,
                )
                for i, (gps, t, dev, _) in enumerate(rows)
            ],
        )
    )
    photos = build_photo_states(request)
    for photo, row in zip(photos, rows, strict=True):
        photo.embedding = unit(*row[3])
    return Trip(
        photos,
        {},
        name="정선",
        regions=list(regions),
        period=request.period,
        photo_seed=seed,
        group_min_sim=0.9,
        font=load_font(),
    )


def test_step1_groups_by_single_linkage_and_trip_text_has_period():
    trip = trip_of(
        [
            (None, None, None, (1, 0, 0)),
            (None, None, None, (0.95, 0.31, 0)),
            (None, None, None, (0.8, 0.6, 0)),  # 첫 사진과는 0.8이지만 둘째와 0.95라 이어진다
            (None, None, None, (0, 0, 1)),
        ]
    )
    sizes = sorted(len(g) for g in trip.groups)
    assert sizes == [1, 3]
    assert json.loads(trip.trip_text()) == {
        "name": "정선",
        "regions": ["강원특별자치도 정선군"],
        "period": "2026-10-02 ~ 2026-10-03 (2일)",
    }


def test_clock_uses_single_mutual_pair_without_min_pairs():
    """실험 시각 보정: 기준 기기와 서로 가장 닮은 쌍 하나로도 일수를 정한다(서비스 CLOCK과 다름)."""
    t = BASE
    rows = [
        {"gps": True, "t": t, "dev": "Ref"},
        {"gps": False, "t": t - timedelta(days=3, minutes=5), "dev": "Cam"},
        {"gps": False, "t": t, "dev": "Lonely"},
        {"gps": False, "t": t, "dev": None},
    ]
    X = np.stack([unit(1, 0, 0), unit(1, 0.1, 0), unit(0, 0, 1), unit(0, 1, 0)])
    T, offs = clock(rows, X, 0.7)
    assert offs == {"Ref": 0, "Cam": 3, "Lonely": None, None: None}
    assert T[1] == rows[1]["t"] + timedelta(days=3) and T[2] is None and T[3] is None


def test_stage2_links_gps_groups_within_100m_and_inserts_by_time():
    rows = [
        (HOME, BASE, "A", (1, 0, 0, 0)),
        ((HOME[0] + 0.0005, HOME[1]), BASE + timedelta(minutes=20), "A", (0, 1, 0, 0)),
        (None, BASE + timedelta(minutes=10), "A", (0, 0, 1, 0)),
        ((HOME[0] + 0.05, HOME[1]), BASE + timedelta(hours=3), "A", (0, 0, 0, 1)),
    ]
    trip = trip_of(rows)
    s = stage2(trip, eps=100, gap_min=30, pair_min_sim=0.7)

    def ids(pl):
        return sorted(trip.photos[i].source.trip_attachment_id for g in pl for i in trip.groups[g])

    assert sorted(ids(pl) for pl in s["places"]) == [[1, 2, 3], [4]]
    assert s["rest"] == [] and len(s["inserted"]) == 1


def test_photo_order_uses_seed_not_request_order():
    rows = [(None, None, None, unit(*np.eye(6)[i])) for i in range(6)]
    a, b = trip_of(rows, seed=61106), trip_of(rows, seed=7)
    order = lambda t: [p.source.trip_attachment_id for p in t.photos]  # noqa: E731
    assert order(a) != order(b) and sorted(order(a)) == list(range(1, 7))


# ---------- 단계 규칙
def test_same_place_records_and_parse_card():
    a = {"id": "kakao:1", "name": "화암동굴(정선)", "lat": 37.0, "lon": 128.0}
    b = {"id": "kb:x", "name": "화암 동굴", "lat": 37.001, "lon": 128.0}
    far = dict(b, lat=37.01)
    assert steps.same_record(a, b) and not steps.same_record(a, far)
    assert steps.parse_card("B", ["A", "B"], ["x", "y"]) == 1
    assert steps.parse_card("B: y", ["A", "B"], ["x", "y"]) == 1
    assert steps.parse_card("Bridge", ["A", "B"], ["x", "Bridge"]) == 1
    assert steps.parse_card("none", ["A", "B"], ["x", "y"]) is None


def test_searcher_filters_by_sigungu_and_strong_index():
    async def go():
        places = Places(resources(None, None, FakeKakao(found=[CAFE, OTHER])))
        srch = steps.Searcher(places, [HOME], ["강원특별자치도 정선군"])
        found = await srch.kakao("찻집")
        return found

    found = asyncio.run(go())
    assert [x["id"] for x in found] == ["kakao:77"]
    items = [{"name": "정선"}, {"name": "정선 아리랑시장"}]
    assert steps.strong_index("정선 아리랑", items) == [{"name": "정선 아리랑시장"}]


def test_prompts_match_experiment_text():
    """실험 코드의 최종 프롬프트와 글자 단위로 같은지 해시로 고정한다(바꾸면 실험과 비교 불가)."""
    digests = {
        name: hashlib.sha256(getattr(prompts, name).encode()).hexdigest()[:16]
        for name in PROMPT_DIGESTS
    }
    assert digests == PROMPT_DIGESTS


PROMPT_DIGESTS = {  # 실험 코드에서 직접 계산한 값(2026-11-10)
    "DESCRIBE": "d6f3611fc31280d6",
    "PROPOSE_V2": "b7b425e4f2206dca",
    "AUDIT_V2": "e8963cdea0cc2863",
    "QUERY2": "e74077323b34899e",
    "CHOOSE3": "1fdd60ba11b8e724",
    "CHOOSE4": "89be903a2c919002",
    "GPS_CHOOSE": "d00f269b7dcb7cb5",
    "FACILITY": "9dfc573beb6d6f98",
    "SHORTLIST10_PREFIX": "1216e8e91d1078d7",
    "IMAGE_PICK2": "6bbfb6ba84d7323b",
    "REASSIGN2": "59d22d26d728d7d1",
    "PURGE": "0f39d99001381cc9",
    "SELECT": "a6448e9ab3acf513",
}


# ---------- 클라이언트 확장
@pytest.mark.parametrize("center", [None, (37.3365, 128.7836)])
def test_kakao_search_keyword_without_or_with_center(center):
    def handler(request):
        params = dict(request.url.params)
        if center is None:
            assert params == {"query": "카페", "size": "15", "sort": "accuracy"}
        else:
            assert params["x"] == "128.783600" and params["radius"] == "20000"
            assert params["sort"] == "distance"
        docs = [{"id": "1", "place_name": "카페", "address_name": "강원 정선군",
                 "road_address_name": "강원 정선군 길", "x": "128.1", "y": "37.1"}]  # fmt: skip
        return httpx.Response(200, json={"documents": docs})

    async def go():
        async with KakaoLocalClient("k", transport=httpx.MockTransport(handler)) as client:
            options = {} if center is None else dict(radius_m=99_000, sort="distance")
            return await client.search_keyword("카페", KakaoBudget(1), center=center, **options)

    (place,) = asyncio.run(go())
    assert place.address_name == "강원 정선군" and place.road_address_name == "강원 정선군 길"
    assert place.address == "강원 정선군 길"


def test_kakao_search_category_pages_and_budget():
    def handler(request):
        params = dict(request.url.params)
        assert request.url.path.endswith("/search/category.json")
        assert params["page"] == "2" and params["size"] == "15"
        return httpx.Response(200, json={"documents": [], "meta": {"is_end": False}})

    async def go():
        budget = KakaoBudget(1)
        async with KakaoLocalClient("k", transport=httpx.MockTransport(handler)) as client:
            first = await client.search_category("AT4", 37.0, 128.0, 20_000, budget, page=2)
            second = await client.search_category("AT4", 37.0, 128.0, 20_000, budget)
        return first, second

    assert asyncio.run(go()) == (([], False), None)


def test_vllm_extra_body_is_sent_and_cannot_override_body():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        message = {"role": "assistant", "content": "ok"}
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "stop"}]}
        )

    async def go():
        async with VllmClient(
            "http://vlm.test/v1", "m", transport=httpx.MockTransport(handler)
        ) as c:
            await c.chat([{"role": "user", "content": "t"}], [], 5, extra_body={"x": {"y": 1}})
            with pytest.raises(ValueError, match="max_tokens"):
                await c.chat(
                    [{"role": "user", "content": "t"}], [], 5, extra_body={"max_tokens": 9}
                )

    asyncio.run(go())
    assert seen["x"] == {"y": 1} and seen["max_tokens"] == 5


# ---------- 정리 실행기 연결
async def _agent_only(cfg):
    from app.jobs.process_handler import agent_resources

    async with agent_resources(cfg, None) as agent:
        return agent


def test_workflow_settings_missing_fail_job():
    from app.jobs.config import JobsSettings
    from app.jobs.process_handler import workflow_resources

    cfg = JobsSettings(
        _env_file=None, API_KEY="t", IMAGE_SOURCE="local", PROCESS_WORKFLOW_ENABLED=True
    )

    async def go():
        async with workflow_resources(cfg, None):
            pass

    with pytest.raises(PipelineError) as caught:
        asyncio.run(go())
    assert caught.value.code is ErrorCode.MODEL_NOT_READY


class _Opened:
    def __init__(self, value):
        self.value = value
        self.args = None

    def __call__(self, *args, **kwargs):
        self.args = (args, kwargs)
        return self

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc_info):
        return None


@scenario
async def test_process_handler_runs_workflow(configured, monkeypatch):  # noqa: F811
    from app.jobs import process_handler

    cfg, _ = configured
    for name, value in {
        "PROCESS_WORKFLOW_ENABLED": True,
        "PROCESS_AGENT_ENABLED": True,  # 워크플로우를 켜면 에이전트는 쓰지 않는다
        "VLM_BASE_URL": "http://vlm.test/v1",
        "VLM_MODEL": "gemma-4",
        "PLACE_INDEX_PATH": "/tmp/places.sqlite",
        "KAKAO_REST_API_KEY": SecretStr("kakao-key"),
        "TOURAPI_RAW_PATH": "/tmp/tourapi.json",
    }.items():
        setattr(cfg, name, value)
    places = FakePlaces()
    places.close = lambda: None
    vlm = ScriptedVlm()
    opened = _Opened(vlm)
    monkeypatch.setattr(process_handler, "PlaceIndex", lambda path: places)
    monkeypatch.setattr(process_handler, "load_tour_raw", lambda path: [])
    monkeypatch.setattr(process_handler, "VllmClient", opened)
    monkeypatch.setattr(process_handler, "KakaoLocalClient", _Opened(FakeKakao()))
    gates = []

    def gate(redis, cfg_, kind, wait_sec=None):
        gates.append((kind, wait_sec))

    monkeypatch.setattr(process_handler, "generation_gate", gate)
    monkeypatch.setattr(process_handler, "GatedVlm", lambda client, gate, timeout: client)
    result = await process_handler.run(process_request(), reporter())
    # 워크플로우 정리도 정리 슬롯 게이트(기능별 상한, 검색 우선)를 쓴다(#183).
    assert gates == [("process", cfg.PROCESS_VLM_WAIT_SEC)]
    assert result["places"] == []
    assert [u["trip_attachment_id"] for u in result["unclassified"]] == [101]
    assert opened.args[1]["timeout_sec"] == cfg.PROCESS_WORKFLOW_VLM_TIMEOUT_SEC
    assert [c[0] for c in vlm.calls] == ["describe", "propose"]


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


def test_tour_pick_shows_representative_images_and_returns_record():
    tour = [
        {"contentid": "1", "contenttypeid": "12", "title": "화암동굴", "addr1": "정선군",
         "mapx": "128.78", "mapy": "37.33", "lDongRegnCd": "51", "lDongSignguCd": "770",
         "firstimage": "http://img.test/1.jpg"},
        {"contentid": "2", "contenttypeid": "39", "title": "식당", "addr1": "정선군",
         "mapx": "128.70", "mapy": "37.38", "lDongRegnCd": "51", "lDongSignguCd": "770"},
    ]  # fmt: skip

    class Remote:
        async def get(self, url):
            return _png((10, 20, 30))

    answers = {
        "shortlist": lambda text: {"candidates": [1]},
        "imagepick": lambda text: {"choice": re.search(r"후보: (\d+)\.", text).group(1)},
    }
    vlm = ScriptedVlm(answers)

    async def go():
        res = replace(resources(None, vlm), tour=tour, remote_images=Remote())
        places = Places(res)
        srch = steps.Searcher(places, [HOME], ["강원특별자치도 정선군"])
        trip = trip_of([(None, None, None, (1, 0))])
        trip._images = {1: _png((1, 2, 3))}
        img = await steps.grid(trip, [(0, "")], 3, 341)
        return await steps.tour_pick2(Pod(vlm, 1, 1), trip, srch, img, "요약", "name_big")

    record = asyncio.run(go())
    assert record == {
        "id": "kb:tourapi:1", "name": "화암동굴", "category": "tourapi:12", "address": "정선군",
        "lat": 37.33, "lon": 128.78,
    }  # fmt: skip
    # 목록은 종류 12·14·28·32만(39 음식점은 빠짐)
    assert [c[3] for c in vlm.calls] == [1, 2]


def test_gated_vlm_passes_extra_body():
    from app.jobs.gpu_requests import GatedVlm

    class Gate:
        async def run(self, factory, timeout):
            return await factory()

    class Client:
        async def chat(self, messages, urls, max_tokens, response_format=None, **options):
            return options

    vlm = GatedVlm(Client(), Gate(), 1)
    extra = {"mm_processor_kwargs": {"max_soft_tokens": 560}}
    options = asyncio.run(vlm.chat([], [], 1, temperature=0, stream=True, extra_body=extra))
    assert options == {"temperature": 0, "stream": True, "extra_body": extra}
    options = asyncio.run(vlm.chat([], [], 1, allow_truncated=True))
    assert options == {"allow_truncated": True}
    assert asyncio.run(vlm.chat([], [], 1)) == {}


def test_zero_time_veto_turns_time_rules_off_like_experiment(tmp_path, loop):
    """실험은 tveto_min이 0이면 시각 떼기와 시각 거부를 하지 않았다."""
    request, settings, engine = pipeline(tmp_path, SCENE)
    ctx = replace(build_context(settings, engine), workflow=resources(loop, ScriptedVlm()))
    ctx.thresholds.update(blur_var_min=10, phash_max_dist=0, workflow_time_veto_min=0)
    result = run(request, ctx, lambda *args: None, lambda: False)
    assert sorted(a.trip_attachment_id for p in result.places for a in p.attachments) == [1, 2, 3]


# ---------- 결정 221: 실패 처리, 기간 밖·지역 밖, 기본값, TourAPI 원자료 받기
def test_pod_retries_transient_failures_then_fails_job(monkeypatch):
    from app.pipeline.workflow import client

    monkeypatch.setattr(client, "RETRY_WAIT_SEC", 0)

    class Flaky:
        def __init__(self, error, ok_after):
            self.error, self.ok_after, self.calls = error, ok_after, []

        async def chat(self, messages, urls, max_tokens, **options):
            self.calls.append(options)
            if len(self.calls) <= self.ok_after:
                raise self.error
            return ChatTurn(content="{}")

    text = [{"type": "text", "text": "t"}]
    ok = Flaky(TimeoutError(), 2)
    assert asyncio.run(Pod(ok, 2, 3).chat(text, 10, budget=560)) == {"text": "{}", "finish": "stop"}
    assert len(ok.calls) == 3
    assert ok.calls[0]["extra_body"] == {"mm_processor_kwargs": {"max_soft_tokens": 560}}
    # 다시 보낸 뒤에도 실패하면 작업 실패. 시간 초과는 TIMEOUT, 그 밖은 MODEL_NOT_READY
    for error, cls, code in (
        (TimeoutError(), WorkflowTimeout, ErrorCode.TIMEOUT),
        (VLMTransientError("x"), WorkflowUnavailable, ErrorCode.MODEL_NOT_READY),
    ):
        vlm = Flaky(error, 99)
        with pytest.raises(cls) as caught:
            asyncio.run(Pod(vlm, 1, 2).chat(text, 10))
        assert caught.value.code is code and len(vlm.calls) == 3
    # 불완전 응답과 GPU 슬롯 대기 초과(VLMUnavailable)는 다시 보내지 않고 바로 실패
    broken = Flaky(VLMUnavailable("슬롯"), 99)
    pod = Pod(broken, 1, 3)
    with pytest.raises(WorkflowUnavailable):
        asyncio.run(pod.chat(text, 10))
    assert len(broken.calls) == 1
    # 한 호출이 실패하면 뒤 호출은 보내지 않는다
    with pytest.raises(WorkflowUnavailable):
        asyncio.run(pod.chat(text, 10))
    assert len(broken.calls) == 1


def test_model_failure_fails_the_job(tmp_path, loop, monkeypatch):
    from app.pipeline.workflow import client

    monkeypatch.setattr(client, "RETRY_WAIT_SEC", 0)
    vlm = ScriptedVlm(fail=VLMTransientError("x"))
    with pytest.raises(WorkflowUnavailable) as caught:
        run_scene(tmp_path, loop, vlm, vlm_retries=1)
    assert caught.value.code is ErrorCode.MODEL_NOT_READY
    assert len(vlm.calls) <= 2 * 3  # 첫 실패 뒤에는 새 호출을 보내지 않는다


# ---------- 결정 224: 잘린 응답은 실패가 아니라 잘린 텍스트로 받는다
TRUNCATED = '{"place_type": "카페",\n  "distinctive": "창가 자리 창가 자리 창가'


def test_pod_returns_truncated_text_without_failing():
    class Truncating:
        def __init__(self):
            self.calls = []

        async def chat(self, messages, urls, max_tokens, **options):
            self.calls.append(options)
            return ChatTurn(content=TRUNCATED, finish_reason="length")

    vlm = Truncating()
    pod = Pod(vlm, 1, 3)
    text = [{"type": "text", "text": "t"}]
    assert asyncio.run(pod.chat(text, 10, stage="vote0")) == {"text": TRUNCATED, "finish": "length"}
    assert asyncio.run(pod.chat(text, 10, stage="describe"))["finish"] == "length"
    assert len(vlm.calls) == 2 and all(c["allow_truncated"] is True for c in vlm.calls)
    assert pod.failed is None and pod.stats["errors"] == 0
    assert pod.stats["truncated"] == 2
    assert pod.stats["truncated:vote0"] == pod.stats["truncated:describe"] == 1


def test_truncated_description_goes_into_propose_as_one_line():
    line = steps.desc_line(3, {"truncated": TRUNCATED})
    assert line == '[3] (잘림) {"place_type": "카페", "distinctive": "창가 자리 창가 자리 창가'
    long = steps.desc_line(1, {"truncated": "가" * 1000})
    assert long == "[1] (잘림) " + "가" * steps.TRUNCATED_DESC_CHARS
    # 잘리지 않았는데 읽지 못한 설명은 이전처럼 설명 없음
    assert steps.desc_line(2, {"error": None, "raw": "깨짐"}) == "[2] (설명 없음)"


def test_unreadable_but_finished_description_stays_without_description():
    """stop으로 끝났는데 JSON이 아니면 잘린 설명이 아니라 이전처럼 설명 없음으로 둔다."""

    class Pod_:
        def __init__(self, turn):
            self.turn = turn

        async def chat(self, content, max_tokens, budget=None, stage=""):
            return self.turn

    class Trip_:
        G = 1

        def diverse(self, g, k):
            return [0]

        def trip_text(self):
            return ""

        def photo_uri(self, i, side):
            return "data:image/png;base64,"

    stop = asyncio.run(steps.describe(Pod_({"text": "설명 문장", "finish": "stop"}), Trip_()))
    assert stop == [{"error": None, "raw": "설명 문장"}]
    assert steps.desc_line(1, stop[0]) == "[1] (설명 없음)"
    cut = asyncio.run(steps.describe(Pod_({"text": "설명 문", "finish": "length"}), Trip_()))
    assert cut == [{"truncated": "설명 문"}]


def test_truncated_describe_and_vote_do_not_fail_the_job(tmp_path, loop, caplog):
    class TruncatingVlm(ScriptedVlm):
        def __init__(self):
            super().__init__({"choose": NAME_ONLY})
            self.prompts = {}

        async def chat(self, messages, image_urls, max_tokens, response_format=None, **options):
            assert options["allow_truncated"] is True
            turn = await super().chat(messages, image_urls, max_tokens, response_format, **options)
            stage = self.calls[-1][0]
            self.prompts.setdefault(stage, messages[0]["content"][0]["text"])
            if stage in ("describe", "vote"):
                return ChatTurn(content=TRUNCATED, finish_reason="length")
            return turn

    caplog.set_level("INFO")
    vlm = TruncatingVlm()
    result, _ = run_scene(tmp_path, loop, vlm, kakao=FakeKakao(found=[CAFE, OTHER]))
    placed = sorted(a.trip_attachment_id for p in result.places for a in p.attachments)
    assert placed  # 작업이 실패하지 않고 결과를 낸다
    n_describe = len([c for c in vlm.calls if c[0] == "describe"])
    n_vote = len([c for c in vlm.calls if c[0] == "vote"])
    assert n_describe and n_vote
    # 잘린 설명이 묶음 제안 지시문에 그룹마다 한 줄로 들어간다
    lines = [x for x in vlm.prompts["propose"].splitlines() if "(잘림)" in x]
    assert len(lines) == n_describe
    assert all(x.endswith('"distinctive": "창가 자리 창가 자리 창가') for x in lines)
    (summary,) = [r for r in caplog.records if getattr(r, "event", None) == "ai_workflow_summary"]
    assert summary.model_errors == 0
    assert summary.model_truncated == n_describe + n_vote
    assert summary.model_truncated_by_stage["describe"] == n_describe
    assert sum(v for k, v in summary.model_truncated_by_stage.items() if k.startswith("vote")) == (
        n_vote
    )


def test_out_of_period_photos_are_unclassified_before_workflow(tmp_path, loop):
    """LOCATING과 같은 기준으로 기간 밖 사진을 뺀다. GPS 사진은 원래 좌표를 단 채 미분류로."""
    rows = SCENE + [(7, (HOME[0] + 0.2, HOME[1]), -60 * 24 * 5, (0, 0, 0, 1, 0), False)]
    vlm = ScriptedVlm({"choose": NAME_ONLY})
    result, _ = run_scene(tmp_path, loop, vlm, rows=rows, kakao=FakeKakao(found=[CAFE]))
    assert 7 not in {a.trip_attachment_id for p in result.places for a in p.attachments}
    (left,) = [u for u in result.unclassified if u.trip_attachment_id == 7]
    assert left.issue is Issue.UNCLEAR_LOCATION and left.region_origin is RegionOrigin.EXIF
    assert len([c for c in vlm.calls if c[0] == "describe"]) == 3  # 사진 7은 모델에 가지 않음


def test_gps_place_outside_trip_regions_is_unclassified(tmp_path, loop):
    """잠실처럼 요청 지역(정선군) 밖인 GPS 장소는 사진째 미분류로 뺀다."""
    rows = SCENE + [(7, SEOUL, 120, (0, 0, 0, 1, 0), False)]
    kakao = FakeKakao()
    result, _ = run_scene(tmp_path, loop, ScriptedVlm(), rows=rows, kakao=kakao)
    placed = {a.trip_attachment_id for p in result.places for a in p.attachments}
    assert placed == {1, 2, 3}
    (left,) = [u for u in result.unclassified if u.trip_attachment_id == 7]
    assert left.issue is Issue.UNCLEAR_LOCATION and left.region_origin is RegionOrigin.EXIF
    assert (37.51, 127.1) in kakao.region_calls


def test_gps_place_is_kept_when_region_lookup_fails(tmp_path, loop):
    rows = SCENE + [(7, SEOUL, 120, (0, 0, 0, 1, 0), False)]
    kakao = FakeKakao(failing={(37.51, 127.1)})
    result, _ = run_scene(tmp_path, loop, ScriptedVlm(), rows=rows, kakao=kakao)
    assert 7 in {a.trip_attachment_id for p in result.places for a in p.attachments}


def test_region_rule_is_off_when_trip_region_cannot_be_resolved(tmp_path, loop):
    rows = SCENE + [(7, SEOUL, 120, (0, 0, 0, 1, 0), False)]
    kakao = FakeKakao(failing={(round(HOME[0], 2), round(HOME[1], 2))})
    result, _ = run_scene(tmp_path, loop, ScriptedVlm(), rows=rows, kakao=kakao)
    assert 7 in {a.trip_attachment_id for p in result.places for a in p.attachments}


def test_record_outside_trip_regions_is_treated_as_unnamed(tmp_path, loop):
    """GPS 없는 묶음이 고른 레코드가 지역 밖(강남구)이면 이름이 없는 것으로 다뤄 미분류가 된다."""
    outside = replace(
        OTHER, address_name="강원 정선군 1"
    )  # 시·군 거르기는 통과하지만 좌표는 강남구
    kakao = FakeKakao(found=[outside])
    result, _ = run_scene(tmp_path, loop, ScriptedVlm({"choose": NAME_ONLY}), kakao=kakao)
    assert sorted(a.trip_attachment_id for p in result.places for a in p.attachments) == [1, 2, 3]
    assert {u.trip_attachment_id for u in result.unclassified} == {4, 5, 6}
    assert (37.5, 127.0) in kakao.region_calls


def test_region_key_treats_general_district_as_city():
    """'성남시 분당구'는 '성남시'로 비교한다(정리 에이전트의 region_key 재사용)."""

    class Kakao(FakeKakao):
        async def region(self, latitude, longitude, budget):
            return KakaoRegion("경기도", "성남시 분당구")

    places = Places(resources(None, None, Kakao()))
    keys = {("경기도", "성남시")}
    assert asyncio.run(organize._in_region(places, 37.38, 127.11, keys)) is True
    assert asyncio.run(organize._in_region(places, 37.38, 127.11, {("경기도", "용인시")})) is False


def test_workflow_is_on_by_default_and_turns_agent_off():
    from app.jobs.config import JobsSettings
    from app.jobs.process_handler import workflow_resources

    on = JobsSettings(_env_file=None, API_KEY="t", IMAGE_SOURCE="local")
    assert on.PROCESS_WORKFLOW_ENABLED is True and on.PROCESS_AGENT_ENABLED is True
    assert asyncio.run(_agent_only(on)) is None
    off = JobsSettings(
        _env_file=None, API_KEY="t", IMAGE_SOURCE="local", PROCESS_WORKFLOW_ENABLED=False
    )

    async def go():
        async with workflow_resources(off, None) as workflow:
            return workflow

    assert asyncio.run(go()) is None
    with pytest.raises(ValueError, match="PROCESS_WORKFLOW_CONCURRENCY"):
        JobsSettings(
            _env_file=None, API_KEY="t", IMAGE_SOURCE="local",
            PROCESS_WORKFLOW_CONCURRENCY=9, GPU_GENERATION_INFLIGHT=8,
        )  # fmt: skip


@pytest.mark.parametrize(
    "fields",
    [
        {"TOURAPI_RAW_S3_URI": "s3://bucket/tourapi.json"},
        {"TOURAPI_RAW_S3_URI": "s3://bucket/tourapi.json", "TOURAPI_RAW_PATH": "/tmp/t"},
        {"TOURAPI_RAW_SHA256": "a" * 64, "TOURAPI_RAW_PATH": "/tmp/t"},
        {"TOURAPI_RAW_S3_URI": "https://b/t.json", "TOURAPI_RAW_SHA256": "a" * 64},
    ],
)
def test_invalid_tourapi_settings_are_rejected(fields):
    from pydantic import ValidationError

    from app.jobs.config import JobsSettings

    with pytest.raises(ValidationError):
        JobsSettings(_env_file=None, API_KEY="t", IMAGE_SOURCE="local", **fields)


def test_prepare_tour_raw_downloads_once_and_checks_hash(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.pipeline.workflow import resources as res

    data = json.dumps({"items": [{"contentid": "1"}]}).encode()
    digest = hashlib.sha256(data).hexdigest()
    calls = []

    def downloader(uri, region):
        def download(target):
            calls.append(uri)
            target.write_bytes(data)

        return download

    monkeypatch.setattr(res, "s3_downloader", downloader)
    path = tmp_path / "raw/tourapi.json"
    cfg = SimpleNamespace(
        PROCESS_WORKFLOW_ENABLED=True,
        TOURAPI_RAW_PATH=str(path),
        TOURAPI_RAW_S3_URI="s3://bucket/tourapi.json",
        TOURAPI_RAW_SHA256=digest,
        AWS_REGION="ap-northeast-2",
    )
    assert res.prepare_tour_raw(cfg) == path and res.prepare_tour_raw(cfg) == path
    assert calls == ["s3://bucket/tourapi.json"]  # 해시 표지가 맞으면 다시 받지 않는다
    assert res.load_tour_raw(str(path)) == [{"contentid": "1"}]
    with pytest.raises(ValueError, match="해시"):
        res.prepare_tour_raw(replace_ns(cfg, TOURAPI_RAW_PATH=str(tmp_path / "b.json"),
                                        TOURAPI_RAW_SHA256="0" * 64))  # fmt: skip
    assert res.prepare_tour_raw(replace_ns(cfg, PROCESS_WORKFLOW_ENABLED=False)) is None


def replace_ns(ns, **fields):
    from types import SimpleNamespace

    return SimpleNamespace(**{**vars(ns), **fields})


def test_several_trip_regions_keep_places_in_any_of_them(tmp_path, loop):
    """요청 지역이 여럿(정선군, 평창군)이면 어느 한 곳 안이면 남기고 모두 밖이면 뺀다(결정 222)."""
    rows = SCENE + [
        (7, PYEONGCHANG, 200, (0, 0, 0, 1, 0), False),
        (8, SEOUL, 300, (0, 0, 0, 0.1, 1), False),
    ]
    kakao = FakeKakao()
    result, _ = run_scene(
        tmp_path, loop, ScriptedVlm(), rows=rows, kakao=kakao, regions=(HOME, PYEONGCHANG)
    )
    placed = {a.trip_attachment_id for p in result.places for a in p.attachments}
    assert {1, 2, 3, 7} <= placed and 8 not in placed
    (left,) = [u for u in result.unclassified if u.trip_attachment_id == 8]
    assert left.issue is Issue.UNCLEAR_LOCATION
    # 같은 사진이어도 평창군을 요청 지역에서 빼면 사진 7도 지역 밖이 된다
    result, _ = run_scene(tmp_path, loop, ScriptedVlm(), rows=rows, kakao=FakeKakao())
    placed = {a.trip_attachment_id for p in result.places for a in p.attachments}
    assert 7 not in placed and 8 not in placed
