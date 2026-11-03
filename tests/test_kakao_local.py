"""실제 카카오 호출 없이 키워드 검색, 행정구역 변환, 작업별 호출 상한을 확인한다."""

import httpx
import pytest

from app.infra.kakao_local import (
    KakaoBudget,
    KakaoLocalClient,
    KakaoPlace,
    KakaoRegion,
    KakaoUnavailable,
)

pytestmark = pytest.mark.anyio
SECRET = "rest-key-secret"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def client_for(handler):
    return KakaoLocalClient(SECRET, transport=httpx.MockTransport(handler))


async def test_keyword_sends_bounded_query_and_parses_places():
    def handler(request):
        assert request.url.path == "/v2/local/search/keyword.json"
        assert request.headers["authorization"] == f"KakaoAK {SECRET}"
        params = dict(request.url.params)
        assert params == {
            "query": "화암동굴", "x": "128.783600", "y": "37.336500", "radius": "20000",
            "size": "15", "sort": "accuracy",
        }  # fmt: skip
        docs = [{"id": "8", "place_name": "화암동굴", "category_name": "여행 > 관광",
                 "road_address_name": "", "address_name": "강원 정선군 화암면",
                 "x": "128.7837", "y": "37.3366", "distance": "12"}]  # fmt: skip
        return httpx.Response(200, json={"documents": docs})

    budget = KakaoBudget(5)
    async with client_for(handler) as client:
        places = await client.keyword("화암동굴", 37.3365, 128.7836, 99_000, budget, size=99)
    assert places == [
        KakaoPlace(
            "kakao:8", "화암동굴", "여행 > 관광", "강원 정선군 화암면", 37.3366, 128.7837, 12
        )
    ]
    assert budget.used == 1


async def test_region_uses_legal_district_names():
    def handler(request):
        assert request.url.path == "/v2/local/geo/coord2regioncode.json"
        docs = [
            {
                "region_type": "H",
                "region_1depth_name": "강원특별자치도",
                "region_2depth_name": "정선군",
            },
            {
                "region_type": "B",
                "region_1depth_name": "강원특별자치도",
                "region_2depth_name": "정선군",
            },
        ]
        return httpx.Response(200, json={"documents": docs})

    async with client_for(handler) as client:
        assert await client.region(37.3365, 128.7836, KakaoBudget(1)) == KakaoRegion(
            "강원특별자치도", "정선군"
        )


async def test_budget_stops_calls_without_network():
    def handler(request):
        pytest.fail("상한을 넘으면 호출하면 안 됨")

    budget = KakaoBudget(0)
    async with client_for(handler) as client:
        assert await client.keyword("카페", 37.0, 127.0, 1000, budget) is None
        assert await client.region(37.0, 127.0, budget) is None
        assert await client.keyword(" ", 37.0, 127.0, 1000, budget) == []
    assert budget.used == 0


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"message": "auth"}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"documents": [{"id": "1"}]}),
        httpx.Response(200, json=[]),
    ],
)
async def test_failures_map_to_kakao_unavailable_without_secret(response):
    async with client_for(lambda _: response) as client:
        with pytest.raises(KakaoUnavailable) as caught:
            await client.keyword("카페", 37.0, 127.0, 1000, KakaoBudget(1))
    assert SECRET not in str(caught.value)


async def test_timeout_maps_to_timeout_error():
    def handler(request):
        raise httpx.ReadTimeout("느림", request=request)

    async with client_for(handler) as client:
        with pytest.raises(TimeoutError):
            await client.region(37.0, 127.0, KakaoBudget(1))


def test_invalid_construction_is_rejected():
    with pytest.raises(ValueError):
        KakaoLocalClient(" ")
    with pytest.raises(ValueError):
        KakaoBudget(-1)
