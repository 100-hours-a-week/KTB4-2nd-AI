from unittest.mock import Mock

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api.services.query_service import QueryService
from app.jobs.execution import execute
from app.jobs.kafka import RequestReceiver
from app.jobs.mock import run
from app.observability.metrics import Metrics
from app.observability.quality import EvaluationInput, evaluate, publish
from app.schemas.query import ParseRequest
from app.search.mock import create_mock_context
from app.v2_main import create_app
from tests.test_v2_jobs import environment, message, process_request, scenario


def samples(content):
    return [
        sample
        for family in text_string_to_metric_families(content.decode())
        for sample in family.samples
    ]


def shared(content):
    """복제본 사이에 같아야 하는 Redis 공유 지표. process_*는 인스턴스별 값이라 제외"""
    return [sample for sample in samples(content) if not sample.name.startswith("process_")]


@scenario
async def test_terminal_metrics_are_atomic_and_not_duplicated_by_publication_or_replica():
    async with environment() as (cfg, a, b, producer):
        await RequestReceiver(a, producer).accept(message(cfg, process_request()))
        record = await a.claim("process", "runner")
        await execute(a, record.identity, record.owner, run)
        assert not await b.finish(record.identity, record.owner, "COMPLETED", result={})
        claim = await b.claim_result(record.identity, "publisher")
        await b.acknowledge_result(record.identity, "publisher")
        one = await Metrics(a.redis, a.prefix).render()
        two = await Metrics(b.redis, b.prefix).render()
        assert shared(one) == shared(two)
        values = samples(one)
        count = [
            s.value
            for s in values
            if s.name == "yeodam_ai_operation_duration_seconds_count"
            and s.labels["phase"] == "total"
        ]
        assert sum(count) == 1
        assert claim is not None
        assert all("trip_id" not in s.labels and "execution_id" not in s.labels for s in values)
        assert any(
            s.name == "yeodam_ai_photos_total"
            and s.labels["result"] == "unclassified"
            and s.value == 1
            for s in values
        )


@scenario
async def test_queued_cancel_is_counted_without_inventing_execution_time():
    async with environment() as (cfg, a, _, producer):
        await RequestReceiver(a, producer).accept(message(cfg, process_request()))
        await a.cancel(77, "process")
        values = samples(await Metrics(a.redis, a.prefix).render())
        assert any(
            s.name == "yeodam_ai_operation_duration_seconds_sum"
            and s.labels["phase"] == "execution"
            and s.value == 0
            for s in values
        )
        assert all(s.labels.get("outcome", "CANCELED") == "CANCELED" for s in values if s.value)


@scenario
async def test_metrics_requires_auth_and_two_apis_show_same_snapshot():
    async with environment() as (cfg, a, b, _):
        await Metrics(a.redis, a.prefix).query("search", "success", 0.123)
        for store in [a, b]:
            app = create_app(cfg, store, start_kafka=False)
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://api.test"
                ) as client,
            ):
                assert (await client.get("/metrics")).status_code == 401
                response = await client.get("/metrics", headers={"Authorization": "Bearer test"})
                assert response.status_code == 200
                assert "text/plain" in response.headers["content-type"]
                assert 'phase="request"' in response.text
                assert "yeodam_ai_evaluation{" not in response.text


@scenario
async def test_metrics_failure_does_not_turn_successful_search_into_error():
    broken = Mock()
    broken.pipeline.side_effect = RedisConnectionError()
    service = QueryService(create_mock_context(), metrics=Metrics(broken, "test:"))
    result = await service.parse(ParseRequest(query="바다 사진", reference_date="2026-10-14"))
    assert result.intent == "SEARCH"


@pytest.mark.parametrize(
    "feature,items,metric,expected",
    [
        (
            "process",
            [
                {"expected_place": "a", "predicted_place": "x"},
                {"expected_place": "a", "predicted_place": "x"},
            ],
            "bcubed_f1",
            1,
        ),
        (
            "search",
            [{"k": 2, "relevant_ids": [1, 2], "retrieved_ids": [2, 3]}],
            "precision_at_k",
            0.5,
        ),
        (
            "story",
            [{"factual": False, "all_places_present": True, "mood_matched": True}],
            "factual_rate",
            0,
        ),
    ],
)
@scenario
async def test_labeled_evaluation_is_computed_and_separate_from_live_metrics(
    feature, items, metric, expected
):
    report = evaluate(
        EvaluationInput(feature=feature, dataset="synthetic-labels", model="test", items=items)
    )
    assert report["metrics"][metric] == expected
    async with environment() as (_, a, _, _):
        await publish(a.redis, a.prefix, report)
        values = samples(await Metrics(a.redis, a.prefix).render())
        assert any(
            s.name == "yeodam_ai_evaluation"
            and s.labels
            == {
                "feature": feature,
                "dataset": "synthetic-labels",
                "unclassified_policy": "none",
                "metric": metric,
            }
            and s.value == expected
            for s in values
        )
        assert any(
            s.name == "yeodam_ai_evaluation_info"
            and s.labels["dataset"] == "synthetic-labels"
            and s.labels["model"] == "test"
            and s.labels["k"] == ("2" if feature == "search" else "none")
            for s in values
        )


def test_story_quality_does_not_accept_automatic_verification_as_truth():
    with pytest.raises((ValueError, KeyError)):
        evaluate(
            EvaluationInput(feature="story", dataset="test", model="test", items=[{"passed": True}])
        )


@pytest.mark.parametrize("policy,recall,samples_count", [("singleton", 0.5, 2), ("exclude", 1, 1)])
def test_unclassified_policy_is_explicit_and_coverage_keeps_original_population(
    policy, recall, samples_count
):
    data = EvaluationInput(
        feature="process",
        dataset="labeled",
        model="test",
        items=[
            {"expected_place": "a", "predicted_place": "x"},
            {"expected_place": "a", "predicted_place": None},
        ],
    )
    with pytest.raises(ValueError, match="미분류 평가 방식"):
        evaluate(data)
    report = evaluate(data.model_copy(update={"unclassified_policy": policy}))
    assert report["metrics"]["coverage"] == 0.5
    assert report["metrics"]["bcubed_recall"] == recall
    assert report["samples"] == samples_count and report["total_samples"] == 2


@scenario
async def test_first_error_changes_existing_zero_counter():
    async with environment() as (_, a, _, _):
        metrics = Metrics(a.redis, a.prefix)

        def count(content):
            return next(
                s.value
                for s in samples(content)
                if s.name == "yeodam_ai_operation_duration_seconds_count"
                and s.labels == {"operation": "search", "outcome": "error", "phase": "request"}
            )

        assert count(await metrics.render()) == 0
        await metrics.query("search", "error", 0.1)
        assert count(await metrics.render()) == 1


@scenario
async def test_invalid_body_logs_completion_once(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="app.observability.metrics")
    async with environment() as (cfg, a, _, producer):
        request = process_request()
        request["attachments"] = []
        receiver = RequestReceiver(a, producer)
        await receiver.accept(message(cfg, request))
        await receiver.accept(message(cfg, request))
        entries = [r for r in caplog.records if getattr(r, "event", None) == "ai_job_finished"]
        assert len(entries) == 1
        assert entries[0].error_code == "INVALID_REQUEST"
