"""사진 선별과 실제 스토리 핸들러의 정상 및 오류 경계를 검증한다."""

import io
import json
import logging
from contextlib import nullcontext
from datetime import datetime
from typing import get_args
from unittest.mock import AsyncMock

import numpy as np
import pytest
from PIL import Image

from app.core.errors import ErrorCode, PipelineCancelled, PipelineError
from app.core.tracing import end_trace, start_trace
from app.infra.vlm import ChatTurn, VLMUnavailable
from app.jobs.models import StoryMessage
from app.story.pipeline import StoryContext, StoryOptions, finalize, run, validate_request
from app.story.prompts import (
    EXAMPLES,
    LENGTH,
    MOOD_VOICE,
    PERCEIVE_SYSTEM,
    SPEECH,
    PhotoNotes,
    StoryDraft,
    perceive_format,
    response_format,
    sentence_plan,
    system_prompt,
    voice_of,
)
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
                    "about": "풍경을 본 일",
                    "detail_summary": "사진 속 풍경",
                    "memo": "풍경을 사진에 남겼다.",
                }
                for i in ids
            ],
        }
    )


def photo_notes(ids):
    return json.dumps(
        {"notes": [{"trip_attachment_id": i, "shows": f"실외, 맑음, 풍경 {i}"} for i in ids]}
    )


def pinned_ids(response_format, field):
    items = response_format["json_schema"]["schema"]["properties"][field]["prefixItems"]
    return [item["properties"]["trip_attachment_id"]["const"] for item in items]


def respond(story=None, notes=None):
    """사진 확인 호출에는 확인 메모를, 집필 호출에는 스토리 초안을 돌려준다."""

    async def chat(messages, urls, max_tokens, response_format=None, **kwargs):
        if response_format["json_schema"]["name"] == "photo_notes":
            ids = pinned_ids(response_format, "notes")
            return ChatTurn(photo_notes(ids) if notes is None else notes)
        return ChatTurn(draft() if story is None else story)

    return chat


def context():
    vectors = AsyncMock()
    vectors.fetch.return_value = {101: np.array([1.0, 0]), 102: np.array([0.0, 1.0])}
    vlm = AsyncMock()
    vlm.chat.side_effect = respond()
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
async def test_perceive_then_write_calls_and_code_derived_metadata():
    ctx = context()
    reporter = AsyncMock()
    result = await run(request_two_places(), reporter, ctx, StoryOptions())
    assert ctx.vlm.chat.await_count == 2
    first, args = ctx.vlm.chat.call_args_list
    # 사진 확인 호출은 같은 사진을 같은 순서로 보내고, 분위기와 장소 메모는 주지 않는다.
    assert first.args[0][0] == {"role": "system", "content": PERCEIVE_SYSTEM}
    assert first.args[1] == args.args[1] and first.args[2] == 1024
    assert pinned_ids(first.kwargs["response_format"], "notes") == [101, 102]
    seen = json.loads(first.args[0][-1]["content"][-1]["text"])
    assert "mood" not in seen and "place_notes" not in seen
    assert all("shows" not in photo for photo in seen["photos"])
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
    ctx.vlm.chat.side_effect = respond(draft(expected))
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
    data = json.loads(content[4]["text"])
    photos = data["photos"]
    assert [(p["image_number"], p["trip_attachment_id"]) for p in photos] == [(1, 101), (2, 102)]
    # 사진 확인 메모는 사진 항목에 붙고, 장소 메모 제공자가 없으면 장소 메모는 빈 목록이다.
    assert [p["shows"] for p in photos] == ["실외, 맑음, 풍경 101", "실외, 맑음, 풍경 102"]
    assert data["place_notes"] == []
    assert len(content) == 5


@scenario
async def test_pinned_photo_order_follows_image_order_when_place_and_time_order_differ():
    # 뒤 장소의 사진이 더 이르면 이미지, 입력 photos, 고정한 사진 ID가 모두 촬영 시각순이어야 한다
    # (결정 64번: 위치마다 ID를 고정하므로 이미지 순서와 어긋나면 다른 사진의 글이 붙는다).
    request = request_two_places()
    request["places"][0]["attachments"][0]["taken_at"] = "2026-10-13T15:00:00+09:00"
    request["places"][1]["attachments"][0]["taken_at"] = "2026-10-12T09:00:00+09:00"
    ctx = context()
    await run(request, AsyncMock(), ctx, StoryOptions())
    args, kwargs = ctx.vlm.chat.call_args
    content = args[0][-1]["content"]
    labels = [p["text"] for p in content if p["type"] == "text"][:-1]
    assert labels == [
        "image_number 1, trip_attachment_id 102:",
        "image_number 2, trip_attachment_id 101:",
    ]
    assert args[1] == ["https://images.test/b.jpg", "https://images.test/a.jpg"]
    photos = json.loads(content[-1]["text"])["photos"]
    assert [p["trip_attachment_id"] for p in photos] == [102, 101]
    pinned = kwargs["response_format"]["json_schema"]["schema"]["properties"]["blocks"]
    assert [b["properties"]["trip_attachment_id"]["const"] for b in pinned["prefixItems"]] == [
        102,
        101,
    ]


@pytest.mark.parametrize("mood", ["PLAIN", "HUMOROUS", "LITERARY"])
@scenario
async def test_system_prompt_is_adopted_text_with_requested_mood_rule_only(mood):
    ctx = context()
    await run({**request_two_places(), "mood": mood}, AsyncMock(), ctx, StoryOptions())
    system, user = ctx.vlm.chat.call_args.args[0]
    voice = voice_of(story_request()["trip_id"])
    assert system == {"role": "system", "content": system_prompt(mood, voice)}
    for other, rule in MOOD_VOICE.items():
        assert (rule in system["content"]) == (other == mood)
    # 말씨는 분위기가 아니라 여행으로 정하고, 예시는 요청한 분위기와 말씨의 것만 넣는다.
    for (other, speech), example in EXAMPLES.items():
        assert (example in system["content"]) == ((other, speech) == (mood, voice))
    assert SPEECH[voice] in system["content"]
    assert "직접 겪은 일처럼" in system["content"] and "과거형" in system["content"]
    # 입력 JSON의 mood도 system의 규칙과 같은 값이다.
    assert json.loads(user["content"][-1]["text"])["mood"] == mood


@scenario
async def test_each_photo_is_given_a_sentence_count_in_the_writing_call_only():
    ctx = context()
    await run(request_two_places(), AsyncMock(), ctx, StoryOptions())
    first, second = ctx.vlm.chat.call_args_list
    assert all(
        "sentences" not in p for p in json.loads(first.args[0][-1]["content"][-1]["text"])["photos"]
    )
    photos = json.loads(second.args[0][-1]["content"][-1]["text"])["photos"]
    assert [p["sentences"] for p in photos] == sentence_plan(story_request()["trip_id"], 2)
    assert LENGTH in second.args[0][0]["content"]


def test_sentence_plan_varies_block_length_and_is_stable_per_trip():
    plan = sentence_plan(33, 10)
    # 열 장이면 한 문장 3, 두 문장 5, 세 문장 2이고, 같은 여행은 다시 만들어도 같다.
    assert sorted(plan) == [1, 1, 1, 2, 2, 2, 2, 2, 3, 3] and plan == sentence_plan(33, 10)
    assert plan != sentence_plan(40, 10)
    assert len(sentence_plan(33, 30)) == 30 and set(sentence_plan(33, 30)) == {1, 2, 3}
    assert sentence_plan(33, 1) in ([1], [2], [3])


@pytest.mark.parametrize(
    "options,expected",
    [
        (StoryOptions(), (1024, 2048)),
        # 설정한 상한이 사진 수에 비해 작으면 사진 한 장에 150, 200토큰으로 늘린다.
        (StoryOptions(perceive_max_tokens=100, max_tokens=150), (300, 400)),
    ],
)
@scenario
async def test_output_token_limits_grow_with_the_number_of_photos(options, expected):
    ctx = context()
    await run(request_two_places(), AsyncMock(), ctx, options)
    first, second = ctx.vlm.chat.call_args_list
    assert (first.args[2], second.args[2]) == expected


@scenario
async def test_output_token_limits_follow_photos_not_places():
    # 장소는 둘이고 사진은 넷이다. 상한은 사진 수로 계산한다.
    ctx = context()
    ctx.vectors.fetch.return_value = {
        101: np.array([1.0, 0]),
        102: np.array([0, 1.0]),
        103: np.array([0.999, 0.04]),
        104: np.array([0.6, 0.8]),
    }
    ctx.vlm.chat.side_effect = respond(draft([101, 103, 104, 102]))
    options = StoryOptions(perceive_max_tokens=100, max_tokens=150, dedupe_z=100.0)
    result = await run(request_near_duplicates(), AsyncMock(), ctx, options)
    assert len(result["blocks"]) == 4
    first, second = ctx.vlm.chat.call_args_list
    assert (first.args[2], second.args[2]) == (600, 800)


def test_story_request_moods_match_prompt_rules():
    # 요청이 받는 분위기와 프롬프트 규칙이 어긋나면 KeyError로 내부 오류가 난다.
    moods = set(get_args(StoryMessage.model_fields["mood"].annotation))
    assert moods == set(MOOD_VOICE)
    assert set(EXAMPLES) == {(mood, voice) for mood in moods for voice in SPEECH}


def test_prompt_keeps_the_rules_found_by_evaluation():
    # 판정에서 드러난 문제를 막으려고 넣은 문장들이다(11/10 실험). 문구가 빠지면 알 수 있게 둔다.
    literary = MOOD_VOICE["LITERARY"]
    # 서정적으로: 감각을 지어내지 않고, 다듬은 표현을 쓰지 않으며, 느낌이나 맺음말로 끝내지 않는다.
    assert "shows나 place_notes에 없는 소리나 냄새를 지어내지 말고" in literary
    assert "다듬은 표현과 비유는" in literary and "두세 블록에만" in literary
    assert "나머지 블록은 한 일과 본 것만으로 끝내세요" in literary
    assert "느낌을 정리하는 문장으로 블록을 끝내지 마세요" in literary
    assert "'-했음' 같은 말끝도 쓰지 않습니다" in literary
    # "-요" 말씨: "-습니다"가 되풀이되거나 "-다"가 섞이지 않게 한다.
    assert "'-습니다'로 끝나는 문장은 스토리 전체에서 서너 번을 넘기지 말고" in SPEECH["yo"]
    assert "'-다'로 끝나는 문장은 쓰지 않습니다" in SPEECH["yo"]
    assert "'-습니다'로 끝나는 문장은" not in SPEECH["da"]
    # 사진 확인 호출: 가장 눈에 띄는 대상을 먼저 적게 한다.
    assert "가장 눈에 띄는 대상" in PERCEIVE_SYSTEM
    for mood in MOOD_VOICE:
        system = system_prompt(mood, "da")
        assert "촬영 시각에 맞게 쓰세요" in system
        assert "마지막 사진이라고 해서 여행이나 행사가 끝났다고 쓰지 마세요" in system
        # 맺음말의 예를 넣으면 그 구절이 그대로 쓰였다. 예 없이 조건만 적는다.
        assert "끝났다는 말은 그 사진의 촬영 시각이 저녁 이후일 때만 씁니다" in system
        assert "일정을 마쳤다" not in system and "마무리했다" not in system
        assert "묘한 분위기" not in system and "시간이 훌쩍" not in system
        # 그곳에 오래 다닌 사람의 일은 빌리지 않고, 사진 확인 메모의 수는 쓰지 않는다.
        assert "그 사진을 찍은 날 그 자리에서 할 수 있었던 일만 고르고" in system
        assert "몇 명, 몇 잔 같은 수는 쓰지 않습니다" in system
        # 장소 메모가 사진의 장면을 밀어내지 않게 하고, 사진 확인 메모의 문구는 옮기지 않는다.
        assert "그것을 본 일이 그 블록에 한 번은 나오게 하세요" in system
        assert "shows의 문구(사람 수, 색, 배경)를 그대로 옮기지도 않습니다" in system
        assert "사진을 찍었다는 말은 스토리 전체에서 두 번까지만 쓰세요" in system
        assert "about에 그 블록에서 이야기할 일을 한 줄로 정하세요" in system
        assert "음식 이름 없이 먹었다는 것만 쓰세요" in system
        assert "모든 문장은 지난 여행을 돌아보는 과거형으로 쓰세요" in system
        assert LENGTH in system


def test_photo_note_allows_eighty_characters():
    # 주된 대상과 배경을 함께 적게 하면서 상한을 60자에서 80자로 올렸다(지시문은 70자 이내).
    assert "70자 이내" in PERCEIVE_SYSTEM
    schema = perceive_format([101])["json_schema"]["schema"]
    note = schema["properties"]["notes"]["prefixItems"][0]
    assert note["properties"]["shows"]["maxLength"] == 80
    PhotoNotes.model_validate({"notes": [{"trip_attachment_id": 101, "shows": "가" * 80}]})
    with pytest.raises(ValueError):
        PhotoNotes.model_validate({"notes": [{"trip_attachment_id": 101, "shows": "가" * 81}]})


def test_mood_is_not_tied_to_sentence_endings():
    # 분위기 문단은 말끝을 정하지 않고, 말끝은 여행마다 정한 말씨 문단이 정한다(결정 120번).
    for rule in MOOD_VOICE.values():
        assert "분위기는 말끝이 아니라 글 전체에서 드러나야 합니다." in rule
        assert "'-다'로 끝나는" not in rule and "'-요'로 끝나는" not in rule
    assert voice_of(33) == "da" and voice_of(40) == "yo"
    ban = "detail_summary와 story_summary에는 웃음 표시를 쓰지 않습니다."
    for mood, rule in MOOD_VOICE.items():
        assert (ban in rule) == (mood == "HUMOROUS")


def test_about_is_written_before_the_text_and_left_out_of_the_result():
    schema = response_format([101])["json_schema"]["schema"]
    block = schema["properties"]["blocks"]["prefixItems"][0]
    assert list(block["properties"]) == ["trip_attachment_id", "about", "detail_summary", "memo"]
    assert set(block["required"]) == set(block["properties"])
    request = validate_request(story_request())
    photo = request.places[0].attachments[0]
    result = finalize(draft([photo.trip_attachment_id]), request, [(5001, photo)])
    assert "about" not in result["blocks"][0]


@pytest.mark.parametrize("mood", ["EMOTIONAL", "CALM"])
@scenario
async def test_removed_moods_are_invalid_requests(mood):
    # BE·FE가 세 값만 보내므로 대체 없이 거절한다(스토리 트랙 결정 80번).
    ctx = context()
    with pytest.raises(PipelineError) as error:
        await run({**request_two_places(), "mood": mood}, AsyncMock(), ctx, StoryOptions())
    assert error.value.code == ErrorCode.INVALID_REQUEST
    ctx.vlm.chat.assert_not_called()


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
    # 위치마다 그 자리의 사진 ID만 받는다(결정 64번).
    assert blocks["items"] is False
    assert [b["properties"]["trip_attachment_id"]["const"] for b in blocks["prefixItems"]] == ids
    assert "$defs" not in schema
    block = blocks["prefixItems"][0]["properties"]
    assert schema["properties"]["story_summary"]["maxLength"] == 40
    assert block["detail_summary"]["maxLength"] == 40
    assert block["memo"]["maxLength"] == 125


def test_response_schemas_do_not_share_mutable_state_between_requests():
    ids = [101, 102]
    first = response_format(ids)["json_schema"]["schema"]
    ids.append(103)
    second = response_format([201])["json_schema"]["schema"]
    first["properties"]["blocks"]["maxItems"] = 99
    first["properties"]["blocks"]["prefixItems"][0]["properties"]["memo"]["maxLength"] = 1
    pinned = second["properties"]["blocks"]["prefixItems"]
    assert second["properties"]["blocks"]["maxItems"] == 1
    assert [b["properties"]["trip_attachment_id"]["const"] for b in pinned] == [201]
    assert pinned[0]["properties"]["memo"]["maxLength"] == 125
    assert StoryDraft.model_json_schema()["properties"]["blocks"]["maxItems"] == 30
    assert response_format([101, 102])["json_schema"]["schema"] != first


@pytest.mark.parametrize(
    "fault,code",
    [
        (VLMUnavailable("secret endpoint"), ErrorCode.VLM_UNAVAILABLE),
        (TimeoutError("secret endpoint"), ErrorCode.TIMEOUT),
    ],
)
@pytest.mark.parametrize("failing_call", ["photo_notes", "travel_story"])
@scenario
async def test_generation_error_mapping(fault, code, failing_call):
    # 사진 확인 호출과 집필 호출 어느 쪽의 서버 오류도 같은 실패로 알린다.
    ctx = context()
    normal = respond()

    async def chat(messages, urls, max_tokens, response_format=None, **kwargs):
        if response_format["json_schema"]["name"] == failing_call:
            raise fault
        return await normal(messages, urls, max_tokens, response_format, **kwargs)

    ctx.vlm.chat.side_effect = chat
    with pytest.raises(PipelineError) as error:
        await run(request_two_places(), AsyncMock(), ctx, StoryOptions())
    assert error.value.code == code and "secret" not in str(error.value)


@pytest.mark.parametrize(
    "notes",
    [
        "not json",
        json.dumps({"notes": [{"trip_attachment_id": 101, "shows": "한 장만 적음"}]}),
        photo_notes([101, 101]),
    ],
)
@scenario
async def test_invalid_photo_notes_are_skipped_without_failing_the_story(notes, caplog):
    ctx = context()
    ctx.vlm.chat.side_effect = respond(notes=notes)
    caplog.set_level(logging.WARNING)
    result = await run(request_two_places(), AsyncMock(), ctx, StoryOptions())
    assert len(result["blocks"]) == 2
    data = json.loads(ctx.vlm.chat.call_args.args[0][-1]["content"][-1]["text"])
    assert all("shows" not in photo for photo in data["photos"])
    assert len(events(caplog, "ai_story_perceive_invalid")) == 1


class Notes:
    def __init__(self, found=None, fault=None):
        self.found, self.fault, self.asked = found or {}, fault, None

    async def notes_for(self, places):
        self.asked = places
        if self.fault:
            raise self.fault
        return self.found


@scenario
async def test_place_notes_are_passed_to_the_writing_call_only():
    original = context()
    notes = Notes({5001: ["셔틀을 타고 올라감", "맛은 평범했음"], 5002: [], 9999: ["없는 장소"]})
    ctx = StoryContext(original.images, original.vectors, original.urls, original.vlm, notes)
    await run(request_two_places(), AsyncMock(), ctx, StoryOptions())
    assert [name for _, name in notes.asked] == ["해변", "공원"]
    first, second = ctx.vlm.chat.call_args_list
    assert "place_notes" not in json.loads(first.args[0][-1]["content"][-1]["text"])
    data = json.loads(second.args[0][-1]["content"][-1]["text"])
    # 메모가 빈 장소와 요청에 없는 장소는 넣지 않는다.
    assert data["place_notes"] == [
        {
            "trip_place_id": 5001,
            "place_name": "해변",
            "notes": ["셔틀을 타고 올라감", "맛은 평범했음"],
        }
    ]


@scenario
async def test_place_notes_failure_does_not_fail_the_story(caplog):
    original = context()
    notes = Notes(fault=RuntimeError("secret search key"))
    ctx = StoryContext(original.images, original.vectors, original.urls, original.vlm, notes)
    caplog.set_level(logging.WARNING)
    result = await run(request_two_places(), AsyncMock(), ctx, StoryOptions())
    assert len(result["blocks"]) == 2
    failed = events(caplog, "ai_story_place_notes_failed")
    # 예외의 문구는 로그의 어느 칸에도 남기지 않는다.
    assert len(failed) == 1 and "secret" not in str(failed[0].__dict__)
    assert failed[0].error_type == "RuntimeError"


@pytest.mark.parametrize("after,calls", [("place_notes", 0), ("photo_notes", 1)])
@scenario
async def test_cancel_between_writing_steps_stops_before_the_next_call(after, calls):
    original, reporter = context(), AsyncMock()

    class Cancelling(Notes):
        async def notes_for(self, places):
            if after == "place_notes":
                reporter.check_cancel.side_effect = PipelineCancelled()
            return {}

    normal = respond()

    async def chat(messages, urls, max_tokens, response_format=None, **kwargs):
        if response_format["json_schema"]["name"] == after:
            reporter.check_cancel.side_effect = PipelineCancelled()
        return await normal(messages, urls, max_tokens, response_format, **kwargs)

    original.vlm.chat.side_effect = chat
    ctx = StoryContext(original.images, original.vectors, original.urls, original.vlm, Cancelling())
    with pytest.raises(PipelineCancelled):
        await run(request_two_places(), reporter, ctx, StoryOptions())
    assert original.vlm.chat.await_count == calls


NOTES_ON = {"STORY_PLACE_NOTES_ENABLED": True}
TAVILY, FIRECRAWL = ("tavily", "search-key"), ("firecrawl", "fire-key")
TWO_KEYS = {"TAVILY_API_KEY": "search-key", "FIRECRAWL_API_KEY": "fire-key"}


@pytest.mark.parametrize(
    "changes,searches",
    [
        ({}, None),
        (NOTES_ON, None),
        (TWO_KEYS, None),
        ({**NOTES_ON, "TAVILY_API_KEY": " ", "FIRECRAWL_API_KEY": ""}, None),
        ({**NOTES_ON, "TAVILY_API_KEY": "search-key"}, [TAVILY]),
        ({**NOTES_ON, "FIRECRAWL_API_KEY": " fire-key "}, [FIRECRAWL]),
        ({**NOTES_ON, **TWO_KEYS}, [TAVILY, FIRECRAWL]),
        ({**NOTES_ON, **TWO_KEYS, "STORY_SEARCH_PRIMARY": "firecrawl"}, [FIRECRAWL, TAVILY]),
        # 주 검색으로 정한 쪽의 키가 비어 있으면 키가 있는 검색만 쓴다.
        (
            {**NOTES_ON, "TAVILY_API_KEY": "search-key", "STORY_SEARCH_PRIMARY": "firecrawl"},
            [TAVILY],
        ),
    ],
    ids=[
        "default",
        "no_key",
        "disabled",
        "blank_keys",
        "tavily_only",
        "firecrawl_only",
        "both",
        "firecrawl_first",
        "primary_without_key",
    ],
)
@scenario
async def test_handler_builds_place_notes_only_with_setting_and_key(changes, searches, monkeypatch):
    from app.jobs.config import JobsSettings
    from app.story import handler

    # 실행 환경에 검색 키나 설정이 있어도 이 테스트의 값만 쓰게 한다.
    names = ("TAVILY_API_KEY", "FIRECRAWL_API_KEY", "STORY_SEARCH_PRIMARY")
    for name in (*names, "STORY_PLACE_NOTES_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    cfg = JobsSettings(
        _env_file=None,
        API_KEY="test",
        IMAGE_SOURCE="local",
        STORY_PLACE_NOTES_TIMEOUT_SEC=12,
        STORY_VLM_TIMEOUT_SEC=77,
        **changes,
    )
    seen, pipeline, reporter = [], AsyncMock(return_value={}), AsyncMock()

    def client(settings):
        seen.append(settings.VLM_TIMEOUT_SEC)
        return AsyncMock()

    monkeypatch.setattr(handler, "JobsSettings", lambda: cfg)
    monkeypatch.setattr(handler.VllmClient, "from_settings", client)
    monkeypatch.setattr(handler, "AsyncQdrantVectors", lambda *a, **k: AsyncMock())
    monkeypatch.setattr(handler.S3ImageResolver, "create", AsyncMock(return_value=AsyncMock()))
    monkeypatch.setattr(handler, "run_pipeline", pipeline)
    await handler.run(story_request(), reporter)
    ctx = pipeline.call_args.args[2]
    # 스토리의 생성 호출은 검색과 다른 제한 시간을 쓴다.
    assert seen == [77] and ctx.vlm.timeout_sec == 77
    assert (ctx.notes is not None) == (searches is not None)
    if searches:
        notes = ctx.notes
        # 주 검색이 앞에 오고, 키가 없는 검색은 빠진다.
        assert notes.searches == searches and notes.timeout_sec == 12
        assert notes.vlm is ctx.vlm and notes.redis is reporter.store.redis
        assert (
            notes.key_prefix == cfg.redis_prefix and notes.ttl_sec == cfg.STORY_PLACE_NOTES_TTL_SEC
        )
        # 블로그 글을 받는 클라이언트는 리디렉션을 따르지 않고, 실행이 끝나면 닫힌다.
        assert notes.http.follow_redirects is False and notes.http.is_closed
        # 환경 변수의 프록시 설정을 쓰지 않는다.
        assert notes.http.trust_env is False


@pytest.mark.parametrize(
    "value,primary",
    [
        ("firecrawl", "firecrawl"),
        (" Firecrawl ", "firecrawl"),
        ("TAVILY", "tavily"),
        ("", "tavily"),
    ],
)
def test_story_search_primary_ignores_case_and_blank_means_the_default(value, primary):
    from app.jobs.config import JobsSettings

    cfg = JobsSettings(
        _env_file=None, API_KEY="test", IMAGE_SOURCE="local", STORY_SEARCH_PRIMARY=value
    )
    assert cfg.STORY_SEARCH_PRIMARY == primary


def test_story_search_primary_must_be_a_known_search():
    from pydantic import ValidationError

    from app.jobs.config import JobsSettings

    with pytest.raises(ValidationError) as caught:
        JobsSettings(
            _env_file=None, API_KEY="test", IMAGE_SOURCE="local", STORY_SEARCH_PRIMARY="bing"
        )
    assert [error["loc"] for error in caught.value.errors()] == [("STORY_SEARCH_PRIMARY",)]


def test_story_step_time_limits_must_fit_in_the_job_time_limit():
    from pydantic import ValidationError

    from app.jobs.config import JobsSettings

    def settings(**changes):
        return JobsSettings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local", **changes)

    assert settings(STORY_PLACE_NOTES_ENABLED=True).STORY_VLM_TIMEOUT_SEC == 90
    with pytest.raises(ValidationError):
        settings(STORY_PLACE_NOTES_ENABLED=True, STORY_PLACE_NOTES_TIMEOUT_SEC=150)
    with pytest.raises(ValidationError):
        settings(STORY_PLACE_NOTES_ENABLED=True, STORY_VLM_TIMEOUT_SEC=120)
    # 생성 호출 하나의 상한은 제한 시간 + 슬롯 대기(2초) + 여유(5초)다.
    # 60 + 2 × (113 + 7) = 300은 거부한다.
    with pytest.raises(ValidationError):
        settings(STORY_PLACE_NOTES_ENABLED=True, STORY_VLM_TIMEOUT_SEC=113)
    assert settings(STORY_PLACE_NOTES_ENABLED=True, STORY_VLM_TIMEOUT_SEC=112.9)
    with pytest.raises(ValidationError):
        settings(STORY_PLACE_NOTES_ENABLED=True, STORY_VLM_TIMEOUT_SEC=110, GPU_WAIT_SEC=5)
    # 장소 메모를 끈 배포는 이전처럼 짧은 작업 제한으로도 시작할 수 있다.
    assert settings(JOBS_TIMEOUT_SEC=10).JOBS_TIMEOUT_SEC == 10


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

    normal = respond()

    async def generate(messages, urls, max_tokens, response_format=None, **kwargs):
        # 집필 호출이 끝난 뒤에 취소가 들어온 경우만 본다.
        if response_format["json_schema"]["name"] == "travel_story":
            reporter.check_cancel.side_effect = PipelineCancelled()
        return await normal(messages, urls, max_tokens, response_format, **kwargs)

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
    monkeypatch.setattr(handler, "AsyncQdrantVectors", lambda *args, **kwargs: vectors)
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
    monkeypatch.setattr(handler, "AsyncQdrantVectors", lambda *args, **kwargs: AsyncMock())
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
        content = body["messages"][-1]["content"]
        assert [part["type"] for part in content] == ["text", "image_url"] * 2 + ["text"]
        if body["response_format"]["json_schema"]["name"] == "photo_notes":
            assert body["max_tokens"] == 1024
            reply = photo_notes(pinned_ids(body["response_format"], "notes"))
            return httpx.Response(
                200, json={"choices": [{"finish_reason": "stop", "message": {"content": reply}}]}
            )
        assert body["response_format"]["json_schema"]["name"] == "travel_story"
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["blocks"]["minItems"] == 2
        assert schema["properties"]["blocks"]["maxItems"] == 2
        pinned = schema["properties"]["blocks"]["prefixItems"]
        assert [b["properties"]["trip_attachment_id"]["const"] for b in pinned] == [101, 102]
        assert body["max_tokens"] == 2048
        assert "40자 이내의 짧은 한 구절" in body["messages"][0]["content"]
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
            assert len(requests) == 2
