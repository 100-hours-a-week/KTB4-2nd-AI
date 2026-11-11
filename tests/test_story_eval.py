"""Langfuse 스토리 평가 구성: 코드 평가기 소스, 판정 평가기 정의, 설정 스크립트."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.observability.langfuse_tracing import REVIEW_QUEUE
from scripts.story_eval import judges, setup_langfuse
from scripts.story_eval import rules_evaluator as rules

SOURCE = Path(rules.__file__)


def request(taken=("2026-10-12T06:00:00+09:00",), period=("2026-10-12", "2026-10-13")):
    return {
        "type": "story",
        "trip_id": 1,
        "execution_id": "s1",
        "trip_name": "정선 여행",
        "period": {"start_date": period[0], "end_date": period[1]},
        "mood": "PLAIN",
        "places": [
            {
                "trip_place_id": 10,
                "place_name": "정선 아리랑시장",
                "attachments": [
                    {"trip_attachment_id": 100 + i, "analyze_storage_key": "k", "taken_at": t}
                    for i, t in enumerate(taken)
                ],
            }
        ],
    }


def result(*memos, summary="정선 여행"):
    return {
        "story_summary": summary,
        "blocks": [
            {
                "order_number": i + 1,
                "trip_place_id": 10,
                "trip_attachment_id": 100 + i,
                "day_label": "1일차, 정선 아리랑시장",
                "detail_summary": "시장",
                "memo": memo,
            }
            for i, memo in enumerate(memos)
        ],
        "verification": {"passed": True, "retry_count": 0},
    }


def scores(req, res) -> dict:
    return {s.name: s for s in rules.score_story(req, res)}


def test_source_uses_only_the_standard_library():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert modules - {"__future__"} <= sys.stdlib_module_names


def test_uploaded_source_runs_on_its_own():
    """Langfuse가 소스만 실행하는 것처럼 모듈 밖에서 실행해 evaluate()를 부른다."""
    space: dict = {"__name__": "langfuse_evaluator"}
    exec(compile(SOURCE.read_text(encoding="utf-8"), "evaluator.py", "exec"), space)
    ctx = SimpleNamespace(
        observation=SimpleNamespace(
            input=json.dumps(request(), ensure_ascii=False),
            output=result("시장에 들렀다."),
            metadata=None,
            tool_calls=[],
        ),
        experiment=None,
    )
    out = space["evaluate"](ctx)
    names = {s.name for s in out.scores}
    assert {"story.rule_time", "story.rule_repeat", "story.auto_len_cv"} <= names
    assert all(s.data_type == "NUMERIC" for s in out.scores)


def test_time_words_that_contradict_the_photo_time_are_counted():
    found = scores(request(), result("저녁에 시장 국수를 먹었다."))
    assert found["story.rule_time"].value == 1.0
    assert "저녁" in found["story.rule_time"].comment
    assert scores(request(), result("아침에 시장 국수를 먹었다."))["story.rule_time"].value == 0.0


def test_utc_offsets_are_read_as_korean_time():
    # 21:00Z는 한국 시각 다음 날 06:00이다. "아침"은 맞고 첫날이 아니라 둘째 날이다.
    found = scores(request(taken=("2026-10-11T21:00:00Z",)), result("첫날 아침 시장에 갔다."))
    assert found["story.rule_time"].value == 0.0


def test_day_order_and_trip_length():
    req = request(taken=("2026-10-12T09:00:00+09:00", "2026-10-13T09:00:00+09:00"))
    res = result("둘째 날 시장에 갔다.", "다음 날 다시 갔다.", summary="3박 4일 정선")
    found = scores(req, res)
    # "둘째 날"은 첫날 블록이고, 요약의 "3박 4일"은 1박 2일 여행과 맞지 않는다.
    assert found["story.rule_time"].value == 2.0


def test_repeats_admin_names_and_markup():
    found = scores(
        request(taken=("2026-10-12T09:00:00+09:00",) * 3),
        result("드디어 정선군 12에 왔다.", "드디어 2명이 걸었다.", "드디어 끝났다.</div>"),
    )
    assert found["story.rule_repeat"].value == 3.0
    assert found["story.rule_admin_name"].value == 1.0
    assert found["story.rule_markup"].value == 1.0


def test_auto_metrics_follow_stage4_definitions():
    found = scores(
        request(taken=("2026-10-12T09:00:00+09:00",) * 2),
        result("오전에 시장에 갔다. 정말 좋았다.", "정선 아리랑시장 국수가 맛있었다."),
    )
    assert found["story.auto_open_move"].value == 0.5
    assert found["story.auto_open_place"].value == 0.5
    assert found["story.auto_intens"].value == 0.5
    assert found["story.auto_sent_per_block"].value == 1.5
    # 끝맺음은 "갔다", "았다", "었다"로 모두 다르다.
    assert found["story.auto_end_top"].value == pytest.approx(1 / 3, abs=1e-4)
    assert found["story.auto_end_types"].value == 1.0


def test_unexpected_input_is_reported_as_one_score():
    ctx = SimpleNamespace(observation=SimpleNamespace(input="not json", output=None))
    out = rules.evaluate(ctx)
    assert [(s.name, s.data_type) for s in out.scores] == [("story.rule_error", "CATEGORICAL")]


def test_judges_cover_the_stage4_items_once_each():
    names = [j.name for j in judges.JUDGES]
    assert len(names) == len(set(names)) == 11
    assert {j.data_type for j in judges.JUDGES} == {"NUMERIC", "BOOLEAN"}
    for judge in judges.JUDGES:
        prompt = judges.user_prompt(judge)
        assert prompt.count("{{input}}") == 1 and prompt.count("{{output}}") == 1
    assert "{{" not in judges.SYSTEM


def cursor_page(items, limit=None, cursor=None):
    """SDK의 커서 방식 목록처럼 두 개씩 나눠 준다(meta.cursor가 다음 쪽)."""
    start = int(cursor or 0)
    rest = start + 2 < len(items)
    meta = SimpleNamespace(cursor=str(start + 2) if rest else None)
    return SimpleNamespace(data=list(items[start : start + 2]), meta=meta)


class FakeApi:
    """설정 스크립트가 부르는 Langfuse API만 흉내 낸다."""

    def __init__(self, connections=()):
        self.configs, self.queues, self.rules = [], [], []
        self.updates = []
        self.connections = list(connections)
        page = lambda items: SimpleNamespace(data=list(items), meta=SimpleNamespace())  # noqa: E731
        self.score_configs = SimpleNamespace(
            get=lambda **_: page(self.configs), create=self._config
        )
        self.annotation_queues = SimpleNamespace(
            list_queues=lambda **_: page(self.queues), create_queue=self._queue
        )
        self.evaluators = SimpleNamespace(
            list=lambda **kw: cursor_page(self.evaluator_items, **kw), create=self._evaluator,
            update=lambda i, request: self.updates.append((i, request)),
        )  # fmt: skip
        self.evaluator_items = []
        self.evaluation_rules = SimpleNamespace(
            list=lambda **kw: cursor_page(self.rules, **kw), create=self._rule,
            update=lambda i, **kw: self.updates.append((i, kw)),
        )  # fmt: skip
        self.llm_connections = SimpleNamespace(list=lambda **_: page(self.connections))

    def _config(self, **kw):
        item = SimpleNamespace(id=f"c{len(self.configs)}", is_archived=False, **kw)
        self.configs.append(item)
        return item

    def _queue(self, **kw):
        self.queues.append(SimpleNamespace(id="q", **kw))

    def _evaluator(self, request):
        item = SimpleNamespace(
            id=f"e{len(self.evaluator_items)}",
            name=request.name,
            source_code=getattr(request, "source_code", None),
            prompt=getattr(request, "prompt", None),
            model_config_=getattr(request, "model_config_", None),
            output_definition=getattr(request, "output_definition", None),
            variable_mapping=getattr(request, "variable_mapping", None),
        )
        self.evaluator_items.append(item)
        return item

    def _rule(self, **kw):
        assigned = [
            SimpleNamespace(evaluator_id=a.evaluator_id) for a in kw["evaluator_assignments"]
        ]
        rule = SimpleNamespace(id=f"r{len(self.rules)}", name=kw["name"], kw=kw)
        rule.evaluator_assignments = assigned
        rule.enabled, rule.sampling, rule.filter = kw["enabled"], kw["sampling"], kw["filter"]
        self.rules.append(rule)


def run_setup(api):
    setup_langfuse.Setup(SimpleNamespace(api=api), dry_run=False).run()


def test_setup_without_llm_connection_builds_everything_but_judges(capsys):
    api = FakeApi()
    run_setup(api)
    assert [c.name for c in api.configs] == [j.name for j in judges.JUDGES]
    numeric = [c for c in api.configs if c.data_type == "NUMERIC"]
    assert numeric and all(c.max_value == judges.MAX_BLOCKS for c in numeric)
    assert [(q.name, len(q.score_config_ids)) for q in api.queues] == [(REVIEW_QUEUE, 11)]
    assert [e.name for e in api.evaluator_items] == [setup_langfuse.RULES_EVALUATOR]
    assert [r.name for r in api.rules] == [setup_langfuse.CODE_RULE]
    columns = {f.column for f in api.rules[0].kw["filter"]}
    assert columns == {"traceName", "level", "isRootObservation"}
    assert "LLM Connections" in capsys.readouterr().out


def test_setup_with_connection_adds_judges_and_is_idempotent():
    google = SimpleNamespace(provider="gemini", adapter="google-ai-studio")
    api = FakeApi(connections=[google])
    run_setup(api)
    judge_items = api.evaluator_items[1:]
    assert [e.name for e in judge_items] == [j.name for j in judges.JUDGES]
    assert {(e.model_config_.provider, e.model_config_.model) for e in judge_items} == {
        ("gemini", setup_langfuse.JUDGE_MODEL)
    }
    rule = next(r for r in api.rules if r.name == setup_langfuse.JUDGE_RULE)
    assert len(rule.evaluator_assignments) == 11
    names = {f.column: f.value for f in rule.kw["filter"]}
    assert names["name"] == [setup_langfuse.WRITE_GENERATION]
    assert names["traceName"] == [setup_langfuse.STORY_TRACE]

    # 같은 정의로 다시 실행하면 아무것도 새로 만들거나 고치지 않는다.
    counts = (len(api.configs), len(api.queues), len(api.evaluator_items), len(api.rules))
    for item in api.evaluator_items[:1]:
        item.source_code = setup_langfuse.RULES_SOURCE.read_text(encoding="utf-8")
    run_setup(api)
    assert (len(api.configs), len(api.queues), len(api.evaluator_items), len(api.rules)) == counts
    assert api.updates == []


def test_setup_restores_a_rule_that_drifted():
    api = FakeApi()
    run_setup(api)
    api.rules[0].enabled = False
    api.rules[0].filter = api.rules[0].filter[:1]
    run_setup(api)
    (rule_id, update), *_ = api.updates
    assert rule_id == api.rules[0].id and update["enabled"] is True
    assert len(update["filter"]) == 3
