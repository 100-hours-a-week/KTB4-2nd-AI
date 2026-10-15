"""Redis에 공유 집계하고 Prometheus가 읽을 수 있는 스냅샷으로 노출한다."""

import asyncio
import json
from collections import defaultdict

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily
from redis.exceptions import RedisError

from app.core.logging import get_logger

BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, float("inf"))
log = get_logger(__name__)


def log_terminal(record, now):
    log.info(
        "AI 작업 종료",
        extra={
            "event": "ai_job_finished",
            "operation": record.kind,
            "trip_id": record.trip_id,
            "job_id": record.execution_id,
            "result": str(record.status),
            "duration_ms": round(max(0, now - record.created_at) * 1000, 2),
            "error_code": str(record.event.error.code)
            if record.event and record.event.error
            else None,
        },
    )


def observe_commands(pipe, key, operation, outcome, phase, seconds):
    prefix = f"{operation}|{outcome}|{phase}|"
    seconds = max(0, seconds)
    pipe.hincrby(key, prefix + "count", 1)
    pipe.hincrbyfloat(key, prefix + "sum", seconds)
    for bucket in BUCKETS:
        if seconds <= bucket:
            pipe.hincrby(key, prefix + str(bucket), 1)


def terminal_commands(pipe, key, record, now):
    outcome = str(record.status)
    for phase, seconds in [
        ("total", now - record.created_at),
        ("queue", (record.started_at or now) - record.created_at),
        ("execution", now - record.started_at if record.started_at else 0),
    ]:
        observe_commands(pipe, key, record.kind, outcome, phase, seconds)
    result = record.event.result if record.event else None
    if outcome == "COMPLETED" and result:
        if record.kind == "process":
            classified = sum(len(p["attachments"]) for p in result["places"])
            pipe.hincrby(key, "photos|classified", classified)
            pipe.hincrby(key, "photos|unclassified", len(result["unclassified"]))
        else:
            pipe.hincrby(key, "story|blocks", len(result["blocks"]))


class Metrics:
    def __init__(self, redis, prefix):
        self.redis, self.prefix = redis, prefix
        self.key = prefix + "metrics:operations"

    async def query(self, operation, outcome, seconds):
        # 관측 실패가 이미 계산한 검색 응답을 바꾸지 않도록 분리한다.
        try:
            async with asyncio.timeout(0.2):
                async with self.redis.pipeline(transaction=True) as pipe:
                    observe_commands(pipe, self.key, operation, outcome, "request", seconds)
                    await pipe.execute()
        except (TimeoutError, RedisError):
            log.warning("검색 지표 기록 실패", extra={"event": "ai_metrics_write_failed"})

    async def render(self):
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hgetall(self.key)
            for name in ["queue:process", "queue:story", "running", "pending"]:
                pipe.zcard(self.prefix + name)
            pipe.hgetall(self.prefix + "metrics:quality")
            pipe.hgetall(self.prefix + "metrics:quality_context")
            values = await pipe.execute()
        histogram = HistogramMetricFamily(
            "yeodam_ai_operation_duration_seconds",
            "AI operation duration; total includes time waiting in the job queue.",
            labels=["operation", "outcome", "phase"],
        )
        groups = defaultdict(dict)
        for operation in ["process", "story"]:
            for outcome in ["COMPLETED", "FAILED", "CANCELED"]:
                for phase in ["total", "queue", "execution"]:
                    groups[(operation, outcome, phase)] = {}
        for operation in ["parse", "search"]:
            for outcome in ["success", "fallback", "partial", "error"]:
                groups[(operation, outcome, "request")] = {}
        for field, value in values[0].items():
            parts = field.split("|")
            if len(parts) == 4:
                groups[tuple(parts[:3])][parts[3]] = float(value)
        for labels, stats in groups.items():
            histogram.add_metric(
                list(labels),
                buckets=[
                    ("+Inf" if bucket == float("inf") else str(bucket), stats.get(str(bucket), 0))
                    for bucket in BUCKETS
                ],
                sum_value=stats.get("sum", 0),
            )
        queue = GaugeMetricFamily("yeodam_ai_jobs", "Shared Redis job counts.", labels=["state"])
        for state, value in zip(
            ["queued_process", "queued_story", "running", "pending_result"],
            values[1:5],
            strict=True,
        ):
            queue.add_metric([state], value)
        photos = CounterMetricFamily(
            "yeodam_ai_photos", "Successfully processed photos, not accuracy.", labels=["result"]
        )
        for name in ["classified", "unclassified"]:
            photos.add_metric([name], int(values[0].get("photos|" + name, 0)))
        blocks = CounterMetricFamily(
            "yeodam_ai_story_blocks", "Validated story blocks, not factual accuracy."
        )
        blocks.add_metric([], int(values[0].get("story|blocks", 0)))

        quality = GaugeMetricFamily(
            "yeodam_ai_evaluation",
            "Latest labeled evaluation, not live accuracy.",
            labels=["feature", "metric"],
        )
        for field, value in values[5].items():
            quality.add_metric(field.split("|"), float(value))
        evaluation_info = GaugeMetricFamily(
            "yeodam_ai_evaluation_info",
            "Conditions for the latest labeled evaluation.",
            labels=["feature", "dataset", "model", "k", "unclassified_policy"],
        )
        for feature, raw in values[6].items():
            metadata = json.loads(raw)
            evaluation_info.add_metric(
                [feature]
                + [
                    str(metadata[name]) if metadata[name] is not None else "none"
                    for name in ["dataset", "model", "k", "unclassified_policy"]
                ],
                1,
            )

        class Snapshot:
            def collect(self):
                return iter([histogram, queue, photos, blocks, quality, evaluation_info])

        registry = CollectorRegistry()
        registry.register(Snapshot())
        return generate_latest(registry)
