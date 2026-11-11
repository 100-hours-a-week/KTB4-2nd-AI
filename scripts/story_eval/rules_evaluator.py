"""Langfuse 코드 평가기: 스토리 결과의 규칙 검사와 자동 지표.

이 파일 전체를 Langfuse 코드 평가기 소스로 올린다(setup_langfuse.py). Langfuse 런타임에는 표준
라이브러리만 있고 2초 안에 끝나야 하므로 앱 코드를 가져오지 않는다.

대상은 스토리 작업 트레이스의 최상위 관측(이름 "story")이다. 입력은 작업 요청(StoryMessage),
출력은 BE에 보낸 결과(StoryResult)다.

규칙 검사(스토리 4단계 rules.py와 10/26 rulecheck.py를 옮김)
- story.rule_time: 시간 표현이 촬영 시각·일차·순서·여행 기간과 맞지 않는 곳의 수
- story.rule_markup: 글에 남은 HTML 태그 조각의 수
- story.rule_repeat: 세 블록 이상이 같은 첫 어절로 시작하거나 같은 마지막 어절로 끝나는 블록 수
- story.rule_admin_name: 행정 구역 이름에 번호가 붙은 이름("정선군 12")이 남은 곳의 수

자동 지표(스토리 4단계 auto_metrics4.py를 옮김, 블록 글 memo 기준)
- story.auto_open_move: 이동·시간 표현으로 시작하는 블록의 비율
- story.auto_open_place: 첫 어절에 그 블록의 장소 이름(첫 낱말)이 든 블록의 비율
- story.auto_end_top: 가장 흔한 끝맺음(마지막 두 글자)이 차지하는 비율
- story.auto_end_types: 서로 다른 끝맺음 수 / 문장 수
- story.auto_len_cv: memo 길이의 변동 계수
- story.auto_intens: 강조어가 든 블록의 비율
- story.auto_sent_per_block: 블록당 문장 수
- story.auto_chars: memo 평균 글자 수
"""

# from __future__ import annotations를 쓰지 않는다. 소스를 모듈 없이 실행하면 dataclass가
# 문자열 주석을 풀 모듈을 찾지 못한다.
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

# Langfuse 런타임이 계약 클래스를 미리 정의하지 않은 경우를 위해 같은 모양으로 둔다.
try:
    Score  # type: ignore[used-before-def]  # noqa: B018
except NameError:

    @dataclass
    class Score:
        name: str
        value: int | float | str | bool
        data_type: str
        comment: str | None = None
        config_id: str | None = None
        metadata: dict[str, Any] | None = None


try:
    EvaluationResult  # type: ignore[used-before-def]  # noqa: B018
except NameError:

    @dataclass
    class EvaluationResult:
        scores: list


# 한국은 일광 절약 시간이 없다. 런타임에 시간대 자료가 없어도 되도록 고정 오프셋을 쓴다.
KST = timezone(timedelta(hours=9))
COMMENT_LIMIT = 500

# ---- 시간 표현(rulecheck.py) ----

# 시각대 표현과 허용 시간(시 단위, [시작, 끝)). 경계는 넉넉하게 잡아 오탐을 줄인다.
TIME_WORDS: dict[str, tuple[int, int]] = {
    "새벽": (0, 7),
    "이른 아침": (4, 9),
    "아침": (5, 11),
    "오전": (5, 12),
    "한낮": (11, 16),
    "점심": (10, 15),
    "정오": (11, 14),
    "낮": (9, 18),
    "오후": (12, 19),
    "해 질 녘": (16, 21),
    "해질녘": (16, 21),
    "저녁": (16, 22),
    "밤": (18, 29),  # 24시 이후는 다음 날 0~5시로 본다
    "야경": (17, 29),
}
# "낮은"(높이), "밤나무"처럼 시각대가 아닌 쓰임은 뒤 글자로, "하룻밤", "어젯밤"은 앞 글자로 거른다.
TIME_RE = re.compile(
    r"(이른 아침|해 질 녘|해질녘|새벽|아침|오전|한낮|점심|정오|오후|저녁|야경|낮(?![은게아고다지았추춘춰])"  # noqa: E501
    r"|(?<!하룻)(?<!알)(?<!군)(?<!어젯)(?<!전날 )(?<!전날)밤(?![나송톨껍색알]))"
)
ORDINAL_DAY = {
    "첫날": 1, "첫째날": 1, "1일째": 1, "1일차": 1,
    "둘째날": 2, "2일째": 2, "2일차": 2,
    "셋째날": 3, "사흗날": 3, "3일째": 3, "3일차": 3,
    "넷째날": 4, "4일째": 4, "4일차": 4,
}  # fmt: skip
DAY_RE = re.compile(
    r"(첫째 ?날|첫날|둘째 ?날|셋째 ?날|넷째 ?날|사흗날|이튿날|[1-9]일 ?(?:째|차)"
    r"|마지막 날|마지막날|다음 날|다음날|전날|같은 날)"
)
SEQ_FIRST = re.compile(r"(?:^|\s)(첫 일정|시작은)")
SEQ_LAST = re.compile(r"(마지막으로|마지막 일정|끝으로)")
SEQ_TRIP_END = re.compile(r"(여행의 끝|여행의 마지막(?! ?날))")
LENGTH_RE = re.compile(r"(\d)박 ?(\d)일|(이틀|사흘|나흘)")
LENGTH_WORD = {"이틀": 2, "사흘": 3, "나흘": 4}
MARKUP_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9]*[^<>]{0,40}>?")

# ---- 반복과 행정 구역 이름(rules.py) ----

# 숫자 뒤에 단위("2명", "3번", "1층")가 붙으면 행정 구역 번호 이름이 아니다.
ADMIN_NUMBERED = re.compile(
    r"[가-힣]{1,6}(?:시|군|구|읍|면|동)\s?\d+"
    r"(?!\d|\s?(?:명|개|번|층|시|분|초|그릇|잔|인|일|박|살|년|월|차|장|곳|원|km|m))"
)
STRIP = re.compile(r"[\s\"'“”‘’.,!?~…()\[\]ㅋㅎ]+$")
RULE_SENT = re.compile(r"(?<=[.!?])\s+")

# ---- 자동 지표(auto_metrics4.py) ----

OPEN = re.compile(
    r"^(다음\s?날|이튿날|마지막\s?날|첫날|둘째\s?날|이후|이어서|그리고|그 뒤|그다음|오전|오후|아침|점심|저녁"  # noqa: E501
    r"|밤|정오|낮|새벽|식사 후|[^ ]+(으로|로) (이동|넘어|향)|[^ ]+에 (도착|들렀|갔|왔|들어))"
)
INTENS = re.compile(r"정말|너무|참 |아주|진짜|완전|엄청|굉장히")
SENT = re.compile(r"[^.!?…~]+[.!?…~]*")


@dataclass
class Finding:
    block: int | None
    kind: str
    text: str
    expected: str = ""


@dataclass
class Story:
    blocks: list[dict]
    summary: str
    taken: dict[int, datetime]
    places: dict[int, str]
    start: date
    end: date
    findings: list[Finding] = field(default_factory=list)


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _kst(value: str) -> datetime:
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t.replace(tzinfo=KST) if t.tzinfo is None else t.astimezone(KST)


def load(request: Any, result: Any) -> Story:
    request, result = _json(request), _json(result)
    if not isinstance(request, dict) or not isinstance(result, dict):
        raise ValueError("입력이나 출력이 JSON 객체가 아님")
    taken, places = {}, {}
    for place in request["places"]:
        places[place["trip_place_id"]] = place.get("place_name") or ""
        for photo in place["attachments"]:
            taken[photo["trip_attachment_id"]] = _kst(photo["taken_at"])
    blocks = sorted(result["blocks"], key=lambda b: b["order_number"])
    if not blocks:
        raise ValueError("블록이 없음")
    return Story(
        blocks=blocks,
        summary=result.get("story_summary") or "",
        taken=taken,
        places=places,
        start=date.fromisoformat(request["period"]["start_date"]),
        end=date.fromisoformat(request["period"]["end_date"]),
    )


def _hour_ok(t: datetime, rng: tuple[int, int]) -> bool:
    h = t.hour + t.minute / 60
    lo, hi = rng
    if hi > 24:
        return h >= lo or h < hi - 24
    return lo <= h < hi


def _ordinal(token: str) -> int | None:
    if token in ORDINAL_DAY:
        return ORDINAL_DAY[token]
    m = re.match(r"([1-9])일(?:째|차)", token)
    return int(m.group(1)) if m else None


def check_time(story: Story) -> None:
    """rulecheck.check의 findings 부분. 촬영 시각은 한국 시각으로 본다."""
    info = []
    for b in story.blocks:
        t = story.taken[b["trip_attachment_id"]]
        info.append((b, t, (t.date() - story.start).days + 1))
    last_day = max(d for _, _, d in info)
    trip_days = (story.end - story.start).days + 1
    add = story.findings.append
    prev = None
    for idx, (b, t, d) in enumerate(info):
        n = b["order_number"]
        for fld in ("detail_summary", "memo"):
            text = b.get(fld) or ""
            for m in TIME_RE.finditer(text):
                if not _hour_ok(t, TIME_WORDS[m.group(1)]):
                    add(Finding(n, "time", m.group(1), f"촬영 {t:%H:%M}"))
            for m in DAY_RE.finditer(text):
                tok = m.group(1).replace(" ", "")
                if text[m.end() : m.end() + 1] == "의":  # "첫날의 피로"처럼 되돌아보는 쓰임
                    continue
                if tok == "마지막날":
                    if d != last_day:
                        add(Finding(n, "time", m.group(1), f"{d}째 날"))
                elif tok in ("다음날", "이튿날"):
                    if prev is None or d - prev[2] != 1:
                        add(Finding(n, "time", m.group(1), f"{d}째 날"))
                elif tok == "전날":
                    if d <= 1:
                        add(Finding(n, "time", m.group(1), "여행 첫날"))
                elif tok == "같은날":
                    if prev is None or prev[2] != d:
                        add(Finding(n, "time", m.group(1), f"{d}째 날"))
                else:
                    day = _ordinal(tok)
                    if day is not None and day != d:
                        add(Finding(n, "time", m.group(1), f"{d}째 날"))
            first = SEQ_FIRST.search(text)
            if first and prev is not None and prev[2] == d:
                add(Finding(n, "time", first.group(1), "같은 날 앞 블록 있음"))
            last = SEQ_LAST.search(text)
            if last and not (idx == len(info) - 1 or info[idx + 1][2] != d):
                add(Finding(n, "time", last.group(1), "같은 날 뒤 블록 있음"))
            trip_end = SEQ_TRIP_END.search(text)
            if trip_end and idx != len(info) - 1:
                add(Finding(n, "time", trip_end.group(1), "마지막 블록 아님"))
        prev = (b, t, d)
    for m in LENGTH_RE.finditer(story.summary):
        if m.group(1):
            nights, days = int(m.group(1)), int(m.group(2))
            if days != trip_days or nights != trip_days - 1:
                add(Finding(None, "time", m.group(0), f"{trip_days - 1}박 {trip_days}일"))
        elif LENGTH_WORD[m.group(3)] != trip_days:
            add(Finding(None, "time", m.group(0), f"{trip_days}일"))


def check_markup(story: Story) -> None:
    for b in story.blocks:
        for fld in ("detail_summary", "memo"):
            m = MARKUP_RE.search(b.get(fld) or "")
            if m:
                story.findings.append(Finding(b["order_number"], "markup", m.group(0)))
    m = MARKUP_RE.search(story.summary)
    if m:
        story.findings.append(Finding(None, "markup", m.group(0)))


def _rule_sentences(memo: str) -> list[str]:
    return [s.strip() for s in RULE_SENT.split(memo) if s.strip()]


def _first_word(text: str) -> str:
    words = text.split()
    return words[0] if words else ""


def _last_word(text: str) -> str:
    words = STRIP.sub("", text).split()
    return words[-1] if words else ""


def check_repeat(story: Story) -> None:
    firsts = {b["order_number"]: _first_word(b.get("memo") or "") for b in story.blocks}
    lasts = {}
    for b in story.blocks:
        sents = _rule_sentences(b.get("memo") or "")
        if sents:
            lasts[b["order_number"]] = _last_word(sents[-1])
    for words, where in ((firsts, "첫 어절"), (lasts, "마지막 어절")):
        counts = Counter(w for w in words.values() if w)
        for n, word in words.items():
            if word and counts[word] >= 3:
                story.findings.append(Finding(n, "repeat", word, where))


def check_admin_name(story: Story) -> None:
    for b in story.blocks:
        for fld in ("detail_summary", "memo"):
            m = ADMIN_NUMBERED.search(b.get(fld) or "")
            if m:
                story.findings.append(Finding(b["order_number"], "admin_name", m.group(0)))
    m = ADMIN_NUMBERED.search(story.summary)
    if m:
        story.findings.append(Finding(None, "admin_name", m.group(0)))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.findall(text) if len(re.sub(r"[^가-힣A-Za-z0-9]", "", s)) >= 2]


def _ending(s: str) -> str:
    core = re.sub(r"[^가-힣]", "", s)
    return core[-2:] if len(core) >= 2 else core


def auto_metrics(story: Story) -> dict[str, float]:
    memos = [b.get("memo") or "" for b in story.blocks]
    names = [story.places.get(b.get("trip_place_id"), "") for b in story.blocks]
    sents = [s for m in memos for s in _sentences(m)]
    ends = Counter(_ending(s) for s in sents)
    lens = [len(m) for m in memos]
    mean_len = statistics.mean(lens)
    return {
        "open_move": sum(bool(OPEN.match(m)) for m in memos) / len(memos),
        "open_place": sum(
            bool(name.split()) and name.split()[0] in m.split(" ")[0]
            for name, m in zip(names, memos, strict=True)
        )
        / len(memos),
        "end_top": max(ends.values()) / len(sents) if sents else 0.0,
        "end_types": len(ends) / len(sents) if sents else 0.0,
        "len_cv": statistics.pstdev(lens) / mean_len if len(lens) > 1 and mean_len else 0.0,
        "intens": sum(bool(INTENS.search(m)) for m in memos) / len(memos),
        "sent_per_block": len(sents) / len(memos),
        "chars": float(mean_len),
    }


RULES = {
    "time": "시간 표현이 촬영 시각·일차·순서·기간과 맞지 않음",
    "markup": "HTML 태그 조각",
    "repeat": "세 블록 이상이 같은 첫 어절이나 마지막 어절을 씀",
    "admin_name": "행정 구역 번호 이름",
}


def _comment(found: list[Finding]) -> str:
    parts = []
    for f in found:
        where = f"블록 {f.block}" if f.block is not None else "요약"
        parts.append(f"{where}: {f.text}" + (f"({f.expected})" if f.expected else ""))
    return "; ".join(parts)[:COMMENT_LIMIT]


def score_story(request: Any, result: Any) -> list[Score]:
    story = load(request, result)
    check_time(story)
    check_markup(story)
    check_repeat(story)
    check_admin_name(story)
    scores = []
    for kind, label in RULES.items():
        found = [f for f in story.findings if f.kind == kind]
        scores.append(
            Score(
                name=f"story.rule_{kind}",
                value=float(len(found)),
                data_type="NUMERIC",
                comment=_comment(found) or None,
                metadata={"rule": label},
            )
        )
    for name, value in auto_metrics(story).items():
        scores.append(Score(name=f"story.auto_{name}", value=round(value, 4), data_type="NUMERIC"))
    return scores


def evaluate(ctx) -> EvaluationResult:
    observation = ctx.observation
    try:
        scores = score_story(observation.input, observation.output)
    except Exception as exc:  # 입력 형식이 다르면 점수 대신 이유를 남긴다.
        scores = [
            Score(
                name="story.rule_error",
                value=type(exc).__name__,
                data_type="CATEGORICAL",
                comment=str(exc)[:COMMENT_LIMIT],
            )
        ]
    return EvaluationResult(scores=scores)
