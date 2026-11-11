"""스토리 평가 구성을 Langfuse 프로젝트에 맞춘다.

여러 번 실행해도 이름이 같은 항목은 새로 만들지 않는다.

만드는 것
- 사람 평가 점수 항목 11개(judges.JUDGES와 같은 이름)와 대기열 "스토리 평가"
- 코드 평가기 "story.rules"(rules_evaluator.py)와 평가 규칙 "스토리 규칙 검사"
  (스토리 작업 트레이스의 최상위 관측)
- 판정 평가기 11개(Gemini 3.5 Flash Lite)와 평가 규칙 "스토리 판정"(스토리 집필 호출)

판정 평가기는 Langfuse 프로젝트에 Google AI Studio LLM 연결이 있어야 만든다. 없으면 건너뛴다.
이미 있는 평가기는 지침이나 소스가 바뀌었을 때만 새 버전을 올린다.

사용: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY를 환경 변수로 두고
    uv run python -m scripts.story_eval.setup_langfuse [--dry-run]
"""

import argparse
from pathlib import Path

from langfuse import Langfuse
from langfuse.api import (
    CreateEvaluatorRequest_Code,
    CreateEvaluatorRequest_LlmAsJudge,
    EvaluationRuleEvaluatorAssignmentInput,
    EvaluationRuleFilter_Boolean,
    EvaluationRuleFilter_StringOptions,
    EvaluatorChatMessage,
    EvaluatorModelConfig,
    EvaluatorOutputDefinition_Boolean,
    EvaluatorOutputDefinition_Numeric,
    PromptVariableMappingInput,
    UpdateCodeEvaluatorRequest,
    UpdateLlmAsJudgeEvaluatorRequest,
)

from app.observability.langfuse_tracing import REVIEW_QUEUE
from scripts.story_eval.judges import JUDGES, MAX_BLOCKS, SYSTEM, Judge, user_prompt

JUDGE_MODEL = "gemini-3.5-flash-lite"
JUDGE_ADAPTER = "google-ai-studio"
STORY_TRACE = "story"
WRITE_GENERATION = "writing.generate"
RULES_EVALUATOR = "story.rules"
RULES_SOURCE = Path(__file__).with_name("rules_evaluator.py")
JUDGE_RULE = "스토리 판정"
CODE_RULE = "스토리 규칙 검사"


def _all(fetch) -> list:
    """커서 방식 목록을 끝까지 읽는다."""
    items, cursor = [], None
    while True:
        page = fetch(limit=100, cursor=cursor)
        items.extend(page.data)
        cursor = getattr(page.meta, "cursor", None)
        if not cursor:
            return items


def _pages(fetch) -> list:
    """쪽 번호 방식 목록을 끝까지 읽는다."""
    items, page = [], 1
    while True:
        result = fetch(page=page, limit=100)
        items.extend(result.data)
        if page >= (getattr(result.meta, "total_pages", 1) or 1):
            return items
        page += 1


def _output(judge: Judge):
    if judge.data_type == "NUMERIC":
        return EvaluatorOutputDefinition_Numeric(
            min_value=0.0,
            max_value=float(MAX_BLOCKS),
            score_value_instructions=judge.value,
            score_reasoning_instructions=judge.reasoning,
        )
    return EvaluatorOutputDefinition_Boolean(
        score_value_instructions=judge.value,
        score_reasoning_instructions=judge.reasoning,
    )


def _prompt(judge: Judge) -> list[EvaluatorChatMessage]:
    return [
        EvaluatorChatMessage(role="system", content=SYSTEM),
        EvaluatorChatMessage(role="user", content=user_prompt(judge)),
    ]


MAPPING = [
    PromptVariableMappingInput(variable="input", source="input"),
    PromptVariableMappingInput(variable="output", source="output"),
]


def _mapping(items) -> set:
    return {(m.variable, str(m.source), m.json_path) for m in items or []}


def _filters(items) -> set:
    """규칙 필터를 비교할 수 있는 모양으로 바꾼다. 읽기 응답과 생성 요청의 타입이 다르다."""
    out = set()
    for f in items or []:
        value = f.value
        value = tuple(value) if isinstance(value, list) else value
        out.add((f.column, str(f.operator), value, getattr(f, "key", None)))
    return out


def _same_judge(current, judge: Judge, model: EvaluatorModelConfig) -> bool:
    """이미 있는 판정 평가기가 지금 정의와 같은지. 다르면 새 버전을 올린다."""
    try:
        prompt = [(m.role, m.content) for m in current.prompt]
        output = current.output_definition
        return (
            prompt == [(m.role, m.content) for m in _prompt(judge)]
            and current.model_config_ is not None
            and (current.model_config_.provider, current.model_config_.model)
            == (model.provider, model.model)
            and output.data_type == judge.data_type
            and output.score_value_instructions == judge.value
            and output.score_reasoning_instructions == judge.reasoning
            and getattr(output, "max_value", None) == getattr(_output(judge), "max_value", None)
            and _mapping(current.variable_mapping) == _mapping(MAPPING)
        )
    except AttributeError:
        return False


class Setup:
    def __init__(self, lf: Langfuse, dry_run: bool) -> None:
        self.api = lf.api
        self.dry_run = dry_run

    def say(self, action: str, name: str) -> None:
        print(f"{'(dry-run) ' if self.dry_run else ''}{action}: {name}")

    def score_configs(self) -> list[str]:
        existing = {
            c.name: c
            for c in _pages(lambda **kw: self.api.score_configs.get(**kw))
            if not c.is_archived
        }
        ids = []
        for judge in JUDGES:
            config = existing.get(judge.name)
            if config is None:
                self.say("점수 항목 생성", judge.name)
                if self.dry_run:
                    continue
                extra = {"min_value": 0.0, "max_value": float(MAX_BLOCKS)}
                config = self.api.score_configs.create(
                    name=judge.name,
                    data_type=judge.data_type,
                    description=judge.title,
                    **(extra if judge.data_type == "NUMERIC" else {}),
                )
            ids.append(config.id)
        return ids

    def queue(self, config_ids: list[str]) -> None:
        queues = _pages(lambda **kw: self.api.annotation_queues.list_queues(**kw))
        current = next((q for q in queues if q.name == REVIEW_QUEUE), None)
        if current is not None:
            missing = set(config_ids) - set(current.score_config_ids)
            if missing:
                print(f"확인 필요: 대기열 '{REVIEW_QUEUE}'에 점수 항목 {len(missing)}개 없음")
            return
        self.say("대기열 생성", REVIEW_QUEUE)
        if not self.dry_run:
            self.api.annotation_queues.create_queue(
                name=REVIEW_QUEUE,
                description="스토리 작업이 끝날 때마다 AI가 자동으로 넣는 스토리 판정 사람 평가",
                score_config_ids=config_ids,
            )

    def code_evaluator(self, evaluators: dict) -> str | None:
        source = RULES_SOURCE.read_text(encoding="utf-8")
        current = evaluators.get(RULES_EVALUATOR)
        if current is None:
            self.say("코드 평가기 생성", RULES_EVALUATOR)
            if self.dry_run:
                return None
            return self.api.evaluators.create(
                request=CreateEvaluatorRequest_Code(
                    name=RULES_EVALUATOR,
                    description="스토리 규칙 검사와 자동 지표",
                    source_code=source,
                    source_code_language="PYTHON",
                )
            ).id
        if getattr(current, "source_code", None) != source:
            self.say("코드 평가기 새 버전", RULES_EVALUATOR)
            if not self.dry_run:
                self.api.evaluators.update(
                    current.id,
                    request=UpdateCodeEvaluatorRequest(
                        source_code=source, source_code_language="PYTHON"
                    ),
                )
        return current.id

    def judge_model(self) -> EvaluatorModelConfig | None:
        connections = _pages(lambda **kw: self.api.llm_connections.list(**kw))
        google = [c for c in connections if c.adapter == JUDGE_ADAPTER]
        if not google:
            return None
        return EvaluatorModelConfig(provider=google[0].provider, model=JUDGE_MODEL)

    def judges(self, evaluators: dict, model: EvaluatorModelConfig) -> list[str]:
        ids = []
        for judge in JUDGES:
            current = evaluators.get(judge.name)
            if current is None:
                self.say("판정 평가기 생성", judge.name)
                if self.dry_run:
                    continue
                current = self.api.evaluators.create(
                    request=CreateEvaluatorRequest_LlmAsJudge(
                        name=judge.name,
                        description=judge.title,
                        prompt=_prompt(judge),
                        model_config_=model,
                        variable_mapping=MAPPING,
                        output_definition=_output(judge),
                    )
                )
            elif not _same_judge(current, judge, model):
                self.say("판정 평가기 새 버전", judge.name)
                if not self.dry_run:
                    self.api.evaluators.update(
                        current.id,
                        request=UpdateLlmAsJudgeEvaluatorRequest(
                            prompt=_prompt(judge),
                            model_config_=model,
                            variable_mapping=MAPPING,
                            output_definition=_output(judge),
                        ),
                    )
            ids.append(current.id)
        return ids

    def rule(self, rules: dict, name: str, evaluator_ids: list[str], filters: list) -> None:
        assignments = [
            EvaluationRuleEvaluatorAssignmentInput(evaluator_id=i) for i in evaluator_ids
        ]
        current = rules.get(name)
        if current is None:
            self.say("평가 규칙 생성", name)
            if not self.dry_run:
                self.api.evaluation_rules.create(
                    name=name,
                    enabled=True,
                    sampling=1.0,
                    filter=filters,
                    evaluator_assignments=assignments,
                )
            return
        assigned = {a.evaluator_id for a in current.evaluator_assignments}
        same = (
            assigned == set(evaluator_ids)
            and current.enabled
            and current.sampling == 1.0
            and _filters(current.filter) == _filters(filters)
        )
        if not same:
            self.say("평가 규칙 갱신", name)
            if not self.dry_run:
                self.api.evaluation_rules.update(
                    current.id,
                    enabled=True,
                    sampling=1.0,
                    filter=filters,
                    evaluator_assignments=assignments,
                )

    def run(self) -> None:
        config_ids = self.score_configs()
        if not self.dry_run:
            self.queue(config_ids)
        evaluators = {e.name: e for e in _all(lambda **kw: self.api.evaluators.list(**kw))}
        rules = {r.name: r for r in _all(lambda **kw: self.api.evaluation_rules.list(**kw))}
        story_only = [
            EvaluationRuleFilter_StringOptions(
                column="traceName", operator="any of", value=[STORY_TRACE]
            ),
            # 실패하거나 취소된 기록(ERROR, WARNING)은 평가하지 않는다. 최상위 관측은 작업 상태를
            # 따른다. 집필 호출은 뒤의 검증 단계에서 작업이 실패해도 DEFAULT로 남아 판정된다.
            EvaluationRuleFilter_StringOptions(
                column="level", operator="any of", value=["DEFAULT"]
            ),
        ]
        code_id = self.code_evaluator(evaluators)
        if code_id is not None:
            root = EvaluationRuleFilter_Boolean(
                column="isRootObservation", operator="=", value=True
            )
            self.rule(rules, CODE_RULE, [code_id], [*story_only, root])
        model = self.judge_model()
        if model is None:
            print(
                "건너뜀: Google AI Studio LLM 연결이 없어 판정 평가기를 만들지 않음. "
                "Langfuse Settings > LLM Connections에서 연결을 추가한 뒤 다시 실행"
            )
            return
        judge_ids = self.judges(evaluators, model)
        if judge_ids and not self.dry_run:
            generation = [
                EvaluationRuleFilter_StringOptions(
                    column="type", operator="any of", value=["GENERATION"]
                ),
                EvaluationRuleFilter_StringOptions(
                    column="name", operator="any of", value=[WRITE_GENERATION]
                ),
            ]
            self.rule(rules, JUDGE_RULE, judge_ids, [*story_only, *generation])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="만들 항목만 출력")
    args = parser.parse_args()
    # 평가기 갱신 응답이 SDK 기본 시간 제한(5초)을 넘는 일이 있어 넉넉히 둔다.
    lf = Langfuse(timeout=60)
    try:
        Setup(lf, args.dry_run).run()
    finally:
        lf.shutdown()


if __name__ == "__main__":
    main()
