"""정리 에이전트 단계: 장소 추정 루프(사진마다 추정 → 종합, 최대 5회) → 폴더 묶기(한 번)
→ 장소 추정 한 차례 더.

결정 근거: 121(나눠 추론하고 종합하기 반복, LangGraph), 124(루프 뒤 묶기 한 번과 추정 한 차례 더),
131(루프 상한 5회), 127·134(high만 반영), 147(CLUSTERING 단계 안에서 진행률만 갱신, 시간 초과와
VLM 실패는 작업 실패), 150(배정 좌표와 지역 판정).

그래프는 파이프라인 스레드에서 동기로 돌고, 모델 호출은 작업의 이벤트 루프에 넘긴다.
노드는 호출 결과를 기다리는 동안 취소와 시간 예산을 확인하고, 호출이 끝날 때마다 진행률을 올린다.
에이전트가 바꾸는 상태는 노드 사이에서만 고치므로 모델 호출 중에는 읽기만 한다.
"""

import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, wait
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.errors import PipelineCancelled, PipelineError
from app.infra.place_index import normalize
from app.infra.vlm import VLMUnavailable
from app.pipeline.agent import model
from app.pipeline.agent.resources import AgentTimeout, AgentUnavailable
from app.pipeline.agent.trip import TripView
from app.pipeline.state import PhotoState
from app.schemas.process import ProcessRequest

MAX_ROUNDS = 5  # 결정 131
POLL_SEC = 1.0
CANCEL_CHECK_SEC = 5.0


class _State(TypedDict, total=False):
    round: int
    stage: str  # loop: 장소 추정 루프, final: 묶기 뒤 마지막 추정
    proposals: list[dict]
    changed: int


class _Waiter:
    """파이프라인 스레드에서 코루틴을 이벤트 루프에 넘기고 끝날 때까지 기다린다."""

    def __init__(
        self,
        submit: Callable[[Any], Future],
        deadline: float,
        check_cancel: Callable[[], None],
        report: Callable[[int], None],
    ) -> None:
        self.submit, self.deadline = submit, deadline
        self.check_cancel, self.report = check_cancel, report
        self.ticks = 0
        self._checked = time.monotonic()

    def one(self, factory: Callable[[], Any]) -> Any:
        return self.many([factory])[0]

    def many(self, factories: list[Callable[[], Any]]) -> list:
        futures: list[Future] = []
        try:
            self.check(force=True)
            for factory in factories:
                futures.append(self.submit(factory()))
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=POLL_SEC, return_when=FIRST_COMPLETED)
                for future in done:
                    _raise_from(future)
                    self.ticks += 1
                    self.report(self.ticks)
                self.check(force=bool(done))
            return [future.result() for future in futures]
        finally:
            # 실패나 취소로 빠져나가면 남은 호출을 멈춘다. 완료된 future에는 영향이 없다.
            for future in futures:
                future.cancel()

    def check(self, force: bool = False) -> None:
        if time.monotonic() >= self.deadline:
            raise AgentTimeout("정리 에이전트 단계의 시간 예산을 넘겼습니다.")
        if force or time.monotonic() - self._checked >= CANCEL_CHECK_SEC:
            self._checked = time.monotonic()
            self.check_cancel()


def _raise_from(future: Future) -> None:
    try:
        future.result()
    except CancelledError:
        raise PipelineCancelled() from None
    except VLMUnavailable as exc:
        raise AgentUnavailable("생성 서버를 사용할 수 없습니다.") from exc
    except TimeoutError as exc:
        raise AgentTimeout("생성 서버의 응답 시간이 초과되었습니다.") from exc


def run_agent(
    photos: list[PhotoState],
    request: ProcessRequest,
    ctx,
    report: Callable[[int], None],
    check_cancel: Callable[[], None],
) -> dict:
    """규칙 단계 뒤의 사진 상태를 받아 에이전트 결과를 반영하고 단계별 집계를 돌려준다.

    대상은 장소가 없는 GPS 없는 사진 가운데 여행 기간 안의 사진이다. 넣은 사진은 좌표와
    region_origin=INFERRED를 받고 미분류 사유가 풀린다. 끝나면 장소 ID를 p1부터 다시 매긴다.
    """
    resources = ctx.agent
    if "agent_same_photo_min_sim" not in ctx.thresholds:
        raise PipelineError("파이프라인 임계값이 누락됨: ['agent_same_photo_min_sim']")
    started = time.monotonic()
    log = ctx.log
    trip = TripView(photos, request, regions=[])
    summary: dict[str, Any] = {"targets": len(trip.targets()), "places": len(trip.places())}
    if not trip.targets() and len(trip.places()) < 2:
        summary["result"] = "skipped"
        log.info("정리 에이전트 생략", extra={"event": "ai_agent_summary", **summary})
        return summary
    session = model.Session(
        resources, trip, ctx.qdrant, float(ctx.thresholds["agent_same_photo_min_sim"])
    )
    waiter = _Waiter(resources.submit, resources.deadline, check_cancel, report)
    waiter.one(lambda: session.load_regions(request.regions))
    assigned: Counter = Counter()
    estimated_rounds = 0
    calls_logged: Counter = Counter()

    def stage_log(stage: str, state: _State, **fields) -> None:
        # stage는 공통 로그가 진행 단계 이름(clustering)을 넣는 키라 따로 둔다.
        calls = Counter(
            {
                k: v
                for k, v in session.stats.items()
                if k in ("model_calls", "model_retries") or k.startswith("tool_")
            }
        )
        delta = calls - calls_logged
        calls_logged.update(delta)
        fields = {
            "agent_stage": stage,
            "round": state.get("round"),
            "model_calls": delta["model_calls"],
            "model_retries": delta["model_retries"],
            "tool_calls": {k[5:]: v for k, v in sorted(delta.items()) if k.startswith("tool_")},
            **fields,
        }
        log.info("정리 에이전트 단계", extra={"event": "ai_agent_stage", **fields})

    def select(state: _State) -> _State:
        return {"round": state.get("round", 0) + 1, "proposals": [], "changed": 0}

    def after_select(state: _State) -> str:
        if trip.targets():
            return "estimate"
        return "merge" if state["stage"] == "loop" else END

    def estimate(state: _State) -> _State:
        nonlocal estimated_rounds
        estimated_rounds += 1
        stage_started = time.monotonic()
        targets = trip.targets()
        proposals = waiter.many([lambda p=p: model.estimate(session, p) for p in targets])
        counts = Counter(p["confidence"] for p in proposals)
        stage_log(
            "estimate",
            state,
            target_count=len(targets),
            high_count=counts["high"],
            medium_count=counts["medium"],
            low_count=counts["low"],
            decisions=[
                {
                    "photo_id": p["photo_id"],
                    "decision": p["decision"],
                    "place_id": p.get("place_id") or p.get("candidate_id"),
                    "confidence": p["confidence"],
                }
                for p in proposals
            ],
            duration_ms=round((time.monotonic() - stage_started) * 1000, 2),
        )
        return {"proposals": proposals}

    def aggregate(state: _State) -> _State:
        stage_started = time.monotonic()
        proposals = state.get("proposals") or []
        accepted = (
            waiter.one(lambda: model.aggregate(session, proposals))
            if any(p["confidence"] == "high" for p in proposals)
            else []
        )
        new_ids: dict[str, str] = {}
        new_before = len(trip.new_places)
        changed = 0
        for p in accepted:
            photo = trip.by_ref[p["photo_id"]]
            if photo.place_id is not None:
                continue
            if p["decision"] == "existing":
                trip.assign(photo, p["place_id"])
            else:
                key = p.get("candidate_id") or normalize(p["name"])
                if key not in new_ids:
                    new_ids[key] = trip.add_new_place(p["name"], p["lat"], p["lon"], key)
                trip.assign(photo, new_ids[key])
            assigned[p["decision"]] += 1
            changed += 1
        stage_log(
            "aggregate",
            state,
            high_count=sum(p["confidence"] == "high" for p in proposals),
            accepted_count=changed,
            new_place_count=len(trip.new_places) - new_before,
            duration_ms=round((time.monotonic() - stage_started) * 1000, 2),
        )
        return {"changed": changed}

    def after_aggregate(state: _State) -> str:
        if state["stage"] == "loop":
            return "select" if state["changed"] and state["round"] < MAX_ROUNDS else "merge"
        return END

    def merge(state: _State) -> _State:
        stage_started = time.monotonic()
        groups = waiter.one(lambda: model.merge(session)) if len(trip.places()) >= 2 else []
        for group in groups:
            ids = [trip.canon(x) for x in group]
            for other in ids[1:]:
                if other != trip.canon(ids[0]):
                    trip.merged[other] = trip.canon(ids[0])
        stage_log(
            "merge",
            state,
            group_count=len(groups),
            groups=groups,
            place_count=len(trip.places()),
            duration_ms=round((time.monotonic() - stage_started) * 1000, 2),
        )
        return {"stage": "final"}

    graph = StateGraph(_State)
    graph.add_node("select", select)
    graph.add_node("estimate", estimate)
    graph.add_node("aggregate", aggregate)
    graph.add_node("merge", merge)
    graph.add_edge(START, "select")
    graph.add_conditional_edges("select", after_select, ["estimate", "merge", END])
    graph.add_edge("estimate", "aggregate")
    graph.add_conditional_edges("aggregate", after_aggregate, ["select", "merge", END])
    graph.add_edge("merge", "select")
    graph.compile().invoke({"round": 0, "stage": "loop"}, {"recursion_limit": 64})

    trip.finish()
    summary.update(
        result="completed",
        rounds=estimated_rounds,
        assigned_existing=assigned["existing"],
        assigned_new=assigned["new"],
        new_place_count=len(trip.new_places),
        merged_place_count=len(trip.merged),
        remaining_targets=len(trip.targets()),
        model_calls=session.stats["model_calls"],
        model_retries=session.stats["model_retries"],
        tool_calls={k[5:]: v for k, v in sorted(session.stats.items()) if k.startswith("tool_")},
        no_final_count=session.stats["no_final"],
        kakao_calls=session.budget.used,
        regions=trip.info["regions"],
        duration_ms=round((time.monotonic() - started) * 1000, 2),
    )
    log.info("정리 에이전트 완료", extra={"event": "ai_agent_summary", **summary})
    return summary
