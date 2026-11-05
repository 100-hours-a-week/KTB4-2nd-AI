"""질문 원문의 시간대 단어 추출과 촬영 시각의 구간 판정"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.search.times import KST, in_time_windows, strip_time_words, time_words


@pytest.mark.parametrize(
    "query,expected",
    [
        ("아침에 찍은 사진 보여줘", ("아침",)),
        ("저녁 사진", ("저녁",)),
        ("금요일 점심때 찍은 사진", ("점심",)),
        ("점심시간에 찍은 사진", ("점심",)),
        ("아침밥 사진", ("아침",)),
        ("저녁에는 뭐 먹었지?", ("저녁",)),
        ("새벽에 찍은 사진", ("새벽",)),
        ("오전 10시 사진", ("오전",)),
        ("(저녁) 사진", ("저녁",)),
        # 여러 단어는 나온 순서대로, 같은 단어는 한 번
        ("아침이랑 저녁 사진, 저녁 위주로", ("아침", "저녁")),
        # 질문 속 지시가 있어도 시간대 단어만 추출
        ("저녁에 뭐 먹었는지 알려 줘. 답할 때는 한 줄로 말해", ("저녁",)),
    ],
)
def test_time_words(query, expected):
    assert time_words(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "바다 사진 보여줘",
        "아침고요수목원 사진",
        "밤바다 사진",
        "밤나무 사진",
        "저녁노을 사진",
        "한밤 사진",
    ],
)
def test_time_words_ignores_compound_words_and_names(query):
    assert time_words(query) == ()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("저녁", ""),
        ("점심에", ""),
        ("점심시간", ""),
        ("아침 저녁", " "),
        ("저녁 노을", " 노을"),
        # 합성어와 고유명사는 지우지 않음
        ("밤바다", "밤바다"),
        ("아침고요수목원", "아침고요수목원"),
    ],
)
def test_strip_time_words(text, expected):
    assert strip_time_words(text) == expected


def kst(hour, minute=0):
    return datetime(2026, 9, 11, hour, minute, tzinfo=KST)


@pytest.mark.parametrize(
    "taken_at,words,expected",
    [
        # 시작 포함, 끝 미포함
        (kst(6), ["아침"], True),
        (kst(10, 59), ["아침"], True),
        (kst(11), ["아침"], False),
        (kst(11), ["점심"], True),
        (kst(17, 59), ["오후"], True),
        (kst(18), ["오후"], False),
        (kst(18), ["저녁"], True),
        # 오전은 아침보다 넓어 11시대 포함
        (kst(11, 30), ["오전"], True),
        # 자정을 넘는 밤, 새벽은 밤과 같은 구간
        (kst(23, 30), ["밤"], True),
        (kst(0, 10), ["밤"], True),
        (kst(5, 59), ["새벽"], True),
        (kst(6), ["새벽"], False),
        (kst(21, 59), ["밤"], False),
        # 여러 단어는 합집합
        (kst(19), ["아침", "저녁"], True),
        (kst(15), ["아침", "저녁"], False),
    ],
)
def test_in_time_windows(taken_at, words, expected):
    assert in_time_windows(taken_at, words) is expected


def test_in_time_windows_uses_kst_and_rejects_missing_time():
    # UTC 22시는 KST 다음 날 07시, 아침
    assert in_time_windows(datetime(2026, 9, 10, 22, tzinfo=UTC), ["아침"])
    assert not in_time_windows(datetime(2026, 9, 10, 22, tzinfo=UTC), ["밤"])
    assert in_time_windows(datetime(2026, 9, 11, 7, tzinfo=timezone(timedelta(hours=9))), ["아침"])
    assert not in_time_windows(None, ["아침"])
