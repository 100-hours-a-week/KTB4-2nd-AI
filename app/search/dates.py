"""질의 해석의 상대 날짜 표현을 기준일로 계산

모델은 상대 날짜 표현을 단위로 분류만 하고 달력 계산은 코드가 담당
10/22 검색 테스트셋 M0에서 모델이 "지난주 금요일"을 한 주 앞 날짜로 계산한 오류 대응
상대 날짜가 없으면 null 대신 unit none, null을 허용하면 실모델이 매번 null을 고름(10/26 확인)
"""

from calendar import monthrange
from datetime import date, timedelta
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

Unit = Literal["none", "day", "week", "weekday", "month", "year", "season"]
Weekday = Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
Season = Literal["SPRING", "SUMMER", "FALL", "WINTER"]

UNITS: tuple[str, ...] = get_args(Unit)
# date.weekday()와 같은 순서, 월요일 0
WEEKDAYS: tuple[str, ...] = get_args(Weekday)
SEASONS: tuple[str, ...] = get_args(Season)
# 계절의 시작 월, 계절마다 석 달
SEASON_START_MONTH: dict[str, int] = {"SPRING": 3, "SUMMER": 6, "FALL": 9, "WINTER": 12}


class RelativeDate(BaseModel):
    """모델이 분류한 상대 날짜 표현

    unit none은 상대 날짜 없음. offset은 이번 0, 지난 -1, 어제 -1, 3일 전 -3
    단위가 쓰지 않는 필드는 무시
    """

    model_config = ConfigDict(extra="forbid")

    unit: Unit
    offset: int | None
    weekday: Weekday | None
    season: Season | None


def resolve_relative_date(expr: RelativeDate, today: date) -> tuple[date, date] | None:
    """표현이 가리키는 날짜 범위, 단위에 필요한 필드가 없거나 달력 범위를 넘으면 None

    오늘 뒤로 넘어가는 범위의 정리는 호출부의 날짜 검증이 담당
    """
    try:
        return _resolve(expr, today)
    except (OverflowError, ValueError):
        # offset이 지나치게 커서 date 범위(1년에서 9999년)를 넘는 경우
        return None


def _resolve(expr: RelativeDate, today: date) -> tuple[date, date] | None:
    match expr.unit:
        case "none":
            return None
        case "day":
            if expr.offset is None:
                return None
            day = today + timedelta(days=expr.offset)
            return day, day
        case "week":
            # 한 주는 월요일부터 일요일
            if expr.offset is None:
                return None
            monday = today - timedelta(days=today.weekday()) + timedelta(weeks=expr.offset)
            if expr.weekday is None:
                return monday, monday + timedelta(days=6)
            day = monday + timedelta(days=WEEKDAYS.index(expr.weekday))
            return day, day
        case "weekday":
            # 요일만 쓴 표현은 오늘을 포함해 가장 최근의 그 요일
            if expr.weekday is None:
                return None
            day = today - timedelta(days=(today.weekday() - WEEKDAYS.index(expr.weekday)) % 7)
            return day, day
        case "month":
            if expr.offset is None:
                return None
            return _month_range(today.year, today.month + expr.offset, months=1)
        case "year":
            if expr.offset is None:
                return None
            year = today.year + expr.offset
            return date(year, 1, 1), date(year, 12, 31)
        case "season":
            if expr.season is None:
                return None
            start_month = SEASON_START_MONTH[expr.season]
            if expr.offset is not None:
                # 겨울의 연도는 12월이 속한 해, 작년 겨울은 작년 12월부터
                year = today.year + expr.offset
            else:
                # 연도 없는 계절은 오늘 이전에 시작한 가장 최근의 그 계절
                year = today.year if date(today.year, start_month, 1) <= today else today.year - 1
            return _month_range(year, start_month, months=3)


def _month_range(year: int, month: int, *, months: int) -> tuple[date, date]:
    """year년 month월 1일부터 months개월 뒤 말일까지, month는 1에서 12 밖이어도 연도로 넘김"""
    start_year, start_index = divmod(year * 12 + month - 1, 12)
    end_year, end_index = divmod(year * 12 + month - 1 + months - 1, 12)
    end_month = end_index + 1
    return (
        date(start_year, start_index + 1, 1),
        date(end_year, end_month, monthrange(end_year, end_month)[1]),
    )
