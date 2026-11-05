"""질문 원문의 시간대 표현과 촬영 시각의 구간 판정

시간대 단어는 닫힌 집합이라 모델 없이 코드가 판정, 질의 해석이 fallback이어도 동작
BE는 해석 결과와 무관하게 원문 query를 검색 요청에 실어 보내므로 계약 변경 없이 사용 가능
10/22 검색 테스트셋 M0에서 시간대 유형 F1 0.093, "저녁 사진"이 시간대와 무관하게 검색된 문제 대응
"""

import re
from collections.abc import Iterable
from datetime import datetime, time, timedelta, timezone

KST = timezone(timedelta(hours=9))

# 단어별 KST 구간(시작 포함, 끝 미포함), 끝이 시작보다 이르면 자정을 넘는 구간
# 결정기록 10/22 확정값, 오전과 새벽은 11/4 추가. 시간대는 겹쳐도 되는 필터라 오전과 점심이 겹침
TIME_WINDOWS: dict[str, tuple[time, time]] = {
    "아침": (time(6), time(11)),
    "오전": (time(6), time(12)),
    "점심": (time(11), time(14)),
    "오후": (time(14), time(18)),
    "저녁": (time(18), time(22)),
    "밤": (time(22), time(6)),
    "새벽": (time(22), time(6)),
}

# 단어 뒤에 붙어도 시간대로 인정하는 말, 그 밖의 글자가 붙으면 다른 단어로 보고 제외
# "아침고요수목원", "밤바다", "밤나무"처럼 시간대 단어로 시작하는 고유명사와 합성어를 거르기 위함
_NOUN_SUFFIXES = ("시간", "식사", "무렵", "때", "쯤", "밥")
_PARTICLES = (
    "에는", "에도", "부터", "까지", "이랑", "이나", "하고",
    "에", "엔", "의", "은", "는", "도", "이", "가", "을", "를", "과", "와", "랑", "나",
)  # fmt: skip
_PATTERN = re.compile(
    rf"(?<![가-힣])({'|'.join(TIME_WINDOWS)})"
    rf"(?:{'|'.join(_NOUN_SUFFIXES)})?(?:{'|'.join(_PARTICLES)})?(?![가-힣])"
)


def time_words(query: str) -> tuple[str, ...]:
    """질문에 나온 시간대 단어, 나온 순서대로 중복 없이"""
    return tuple(dict.fromkeys(match.group(1) for match in _PATTERN.finditer(query)))


def strip_time_words(text: str) -> str:
    """시간대 단어와 뒤에 붙은 말을 지운 문자열, 합성어와 고유명사는 남김"""
    return _PATTERN.sub("", text)


def in_time_windows(taken_at: datetime | None, words: Iterable[str]) -> bool:
    """촬영 시각이 단어 구간 중 하나에 들면 참, 촬영 시각이 없으면 거짓(못 믿으면 붙이지 않음)"""
    if taken_at is None:
        return False
    clock = taken_at.astimezone(KST).time()
    return any(_contains(TIME_WINDOWS[word], clock) for word in words)


def _contains(window: tuple[time, time], clock: time) -> bool:
    start, end = window
    if start < end:
        return start <= clock < end
    return clock >= start or clock < end
