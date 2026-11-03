"""CLUSTERING 규칙 단계 뒤의 정리 에이전트(#153). 진입점은 run_agent, 자원은 AgentResources"""

from app.pipeline.agent.resources import AgentResources, AgentTimeout, AgentUnavailable

__all__ = ["MAX_ROUNDS", "AgentResources", "AgentTimeout", "AgentUnavailable", "run_agent"]


def __getattr__(name: str):
    # LangGraph를 불러오는 그래프는 에이전트를 켠 작업에서 처음 쓸 때 불러온다.
    if name in ("MAX_ROUNDS", "run_agent"):
        from app.pipeline.agent import graph

        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
