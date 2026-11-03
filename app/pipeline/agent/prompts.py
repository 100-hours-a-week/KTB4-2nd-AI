"""정리 에이전트의 지시문과 도구 정의. 평가 도구(2026-10-28 agent.py)와 같은 문장이다.

확신도 기준은 결정 134(폴더 묶기)와 137(장소 추정), high만 반영은 127이다.
"""

# 한 대화에서 모델에게 보낼 수 있는 사진 수. 생성 서버의 요청당 사진 상한(결정 138)
MAX_IMAGES = 30
# 사진 한 장 추정과 종합의 모델 응답 상한(10/26 실험과 같음). 묶기는 폴더가 많아 두 배
MAX_CALLS = 12
MAX_MERGE_CALLS = 24
MAX_TOKENS = 1200
MAX_MERGE_TOKENS = 2000
# read_photo, read_place_photos 한 번에 불러오는 사진 수
READ_LIMIT = 4

CONF_EXISTING = "\n".join(
    [
        "확신도 기준(기존 장소에 넣기)",
        "- high: 대상 사진에 그 장소의 사진과 같은 시설, 간판, 건물, 특징적 사물이 보이고, "
        "반대 근거가 없다.",
        "- medium: 겉모습, 촬영 시각, 장면 종류 같은 정황 근거만 있다.",
        "- low: 근거가 부족하거나, 다른 장소의 간판·업종이 보이거나, "
        "촬영 시각이 그 장소의 방문 시간과 멀리 떨어져 있다.",
        "확신도 기준(새 장소 만들기)",
        "- high: 사진에 장소를 특정하는 단서(간판의 상호, 고유한 시설)가 있고, "
        "외부 검색에서 여행 지역 안이면서 앞뒤 사진의 위치와 맞는 장소가 하나로 확인되며, "
        "기존 장소일 가능성이 배제된다.",
        "- medium: 외부 검색 후보가 여럿이거나 위치가 확인되지 않는다.",
        "- low: 장소를 특정할 단서가 없다.",
    ]
)

CONF_MERGE = "\n".join(
    [
        "확신도 기준(폴더 묶기)",
        "- high: 두 폴더 사진에 같은 시설·간판·건물이 보이거나, "
        "외부 장소 자료에서 두 좌표가 같은 시설의 범위에 속하고, 반대 근거가 없다.",
        "- medium: 가까움, 시간상 이어짐, 비슷한 장면 같은 정황 근거만 있다.",
        "- low: 근거가 부족하거나 다른 상호·업종 같은 반대 근거가 있다.",
    ]
)


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_NEAR = {
    "type": "string",
    "description": "검색 중심: 장소 ID(예: p3), 사진 ID(예: S5770), 또는 trip(여행 전체 중심)",
}
_CONF = {"type": "string", "enum": ["high", "medium", "low"]}

READ_PHOTO = _fn(
    "read_photo",
    f"사진 ID로 사진을 불러와 직접 봅니다. 한 번에 최대 {READ_LIMIT}장입니다.",
    {"photo_ids": {"type": "array", "items": {"type": "string"}}},
    ["photo_ids"],
)
FIND_SIMILAR = _fn(
    "find_similar",
    "임베딩 유사도로 주어진 사진과 닮은 사진을 찾습니다. "
    "사진 ID, 장소 ID, 유사도, 촬영 시각을 돌려줍니다.",
    {"photo_id": {"type": "string"}, "k": {"type": "integer", "description": "1~10"}},
    ["photo_id"],
)
SEARCH_PUBLIC_PLACES = _fn(
    "search_public_places",
    "공공 장소 자료(관광 정보, 상가, 지도 자료)에서 이름이나 분류로 장소를 찾습니다. "
    "후보 ID, 이름, 분류, 좌표, 검색 중심에서의 거리를 돌려줍니다.",
    {
        "query": {"type": "string"},
        "near": _NEAR,
        "radius_m": {"type": "integer", "description": "100~30000"},
    },
    ["query", "near"],
)
KAKAO_KEYWORD_SEARCH = _fn(
    "kakao_keyword_search",
    "카카오 지도에서 키워드로 장소를 찾습니다. "
    "후보 ID, 이름, 분류, 좌표, 검색 중심에서의 거리를 돌려줍니다.",
    {
        "query": {"type": "string"},
        "near": _NEAR,
        "radius_m": {"type": "integer", "description": "100~20000"},
    },
    ["query", "near"],
)
DECIDE = _fn(
    "decide",
    "최종 답입니다. 기존 장소에 넣거나(existing), 외부 검색으로 찾은 새 장소를 냅니다(new).",
    {
        "decision": {"type": "string", "enum": ["existing", "new"]},
        "place_id": {"type": "string", "description": "existing일 때 장소 ID"},
        "candidate_id": {"type": "string", "description": "new일 때 외부 검색 결과의 후보 ID"},
        "new_place_name": {"type": "string", "description": "new일 때 장소 이름"},
        "confidence": _CONF,
        "reason": {"type": "string"},
    },
    ["decision", "confidence", "reason"],
)
FINALIZE = _fn(
    "finalize",
    "종합 결과입니다.",
    {
        "accept": {"type": "array", "items": {"type": "string"}, "description": "받아들일 사진 ID"},
        "reject": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"photo_id": {"type": "string"}, "reason": {"type": "string"}},
            },
        },
    },
    ["accept"],
)
READ_PLACE_PHOTOS = _fn(
    "read_place_photos",
    f"폴더의 사진을 불러와 직접 봅니다. 한 번에 최대 {READ_LIMIT}장입니다.",
    {"place_id": {"type": "string"}, "n": {"type": "integer", "description": f"1~{READ_LIMIT}"}},
    ["place_id"],
)
MERGE = _fn(
    "merge",
    "묶기 결과입니다.",
    {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "place_ids": {"type": "array", "items": {"type": "string"}},
                    "confidence": _CONF,
                    "reason": {"type": "string"},
                },
            },
        }
    },
    ["groups"],
)

ESTIMATE_TOOLS = [READ_PHOTO, FIND_SIMILAR, SEARCH_PUBLIC_PLACES, KAKAO_KEYWORD_SEARCH, DECIDE]
AGGREGATE_TOOLS = [READ_PHOTO, FINALIZE]
MERGE_TOOLS = [READ_PLACE_PHOTOS, READ_PHOTO, SEARCH_PUBLIC_PLACES, KAKAO_KEYWORD_SEARCH, MERGE]

ESTIMATE_SYSTEM = "\n".join(
    [
        "당신은 여행 사진 정리를 돕는 에이전트입니다. "
        "위치 정보(GPS)가 없어 장소가 정해지지 않은 사진 한 장이 주어집니다.",
        "같은 여행의 다른 사진들은 장소 폴더로 묶여 있습니다. "
        "대상 사진이 어느 장소에서 찍혔는지 판단해 decide 도구로 답하세요.",
        "- 여행 안의 기존 장소에서 찍혔으면 decision=existing과 그 장소 ID를 냅니다.",
        "- 기존 장소가 아닌 곳에서 찍혔고 외부 검색으로 그 장소를 찾았으면 "
        "decision=new와 후보 ID, 장소 이름을 냅니다.",
        "- 사진 속 글자, 간판, 건물, 실내 구조, 물건, 풍경 같은 단서를 비교하세요. "
        "촬영 시각이 있으면 앞뒤 사진의 장소와 시간 차이도 근거가 됩니다.",
        "- 잘못 넣은 사진은 사용자가 직접 찾아 고쳐야 하므로, "
        "확신이 없으면 확신도를 낮게 적으세요. 확신도 high인 답만 반영됩니다.",
        f"- 도구는 최대 {MAX_CALLS - 1}번 쓸 수 있고, "
        f"한 대화에서 볼 수 있는 사진은 처음 받은 사진을 포함해 {MAX_IMAGES}장입니다.",
        "",
        CONF_EXISTING,
    ]
)

AGGREGATE_SYSTEM = "\n".join(
    [
        "당신은 여행 사진 정리의 종합 단계입니다. "
        "사진마다 따로 추정한 결과 가운데 확신도 high인 제안을 받습니다.",
        "제안을 다시 확인해 받아들일 사진과 거절할 사진을 finalize 도구로 정하세요.",
        "- 서로 거의 같은 사진(연속 촬영)인데 다른 장소로 제안된 경우, "
        "사진을 직접 보고 하나로 정리하거나 모두 거절합니다.",
        "- 같은 새 장소를 가리키는 제안은 한 장소로 묶습니다(같은 후보 ID나 같은 이름).",
        "- 근거가 약하거나 사진과 맞지 않는 제안은 거절합니다. 확신이 high인 제안만 받아들입니다.",
        f"- read_photo 도구로 사진을 직접 볼 수 있습니다(한 대화에서 {MAX_IMAGES}장까지).",
    ]
)

MERGE_SYSTEM = "\n".join(
    [
        "당신은 여행 사진 정리의 폴더 묶기 단계입니다. "
        "한 장소가 여러 폴더로 나뉘어 있을 수 있습니다.",
        "폴더 목록과 폴더 사이의 거리·시간 관계를 보고, "
        "같은 장소인 폴더들을 merge 도구로 묶으세요.",
        "- read_place_photos로 폴더의 사진을, read_photo로 특정 사진을 직접 볼 수 있습니다"
        f"(한 대화에서 {MAX_IMAGES}장까지).",
        "- search_public_places와 kakao_keyword_search로 "
        "두 폴더가 같은 시설의 범위에 속하는지 확인할 수 있습니다.",
        "- 확신도 high인 묶음만 반영됩니다. 묶을 것이 없으면 빈 목록을 냅니다.",
        "",
        CONF_MERGE,
    ]
)
