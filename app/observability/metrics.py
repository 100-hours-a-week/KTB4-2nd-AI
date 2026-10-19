"""Redis에 공유 집계하고 Prometheus가 읽을 수 있는 스냅샷으로 노출한다."""

import asyncio
import json
from collections import defaultdict

from prometheus_client import CollectorRegistry, ProcessCollector, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily
from redis.exceptions import RedisError

from app.core.logging import (
    FAILURE_INTAKE,
    FAILURE_QUEUE,
    FAILURE_STARTUP,
    get_logger,
    stage_name,
)

BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, float("inf"))
# 데이터셋별 평가 형식은 새 키에 둔다. 이전 버전(22d2bd6)의 렌더러는 metrics:quality를 기능|지표
# 형식으로 읽으므로, 같은 키에 쓰면 롤백하거나 버전이 섞일 때 /metrics 전체가 실패한다.
QUALITY_KEY = "metrics:quality_v2"
QUALITY_CONTEXT_KEY = "metrics:quality_context_v2"
log = get_logger(__name__)


def _ms(seconds):
    return round(max(0, seconds) * 1000, 2)


def log_terminal(record, now, previous_step=None, *, intake=False):
    """종료 상태로 처음 바뀔 때 한 번. previous_step은 종료 직전의 진행 단계

    intake는 Kafka 접수 때 이미 실패로 등록된 작업(형식 오류)이다.
    """
    error = record.event.error if record.event else None
    failure_stage = None
    if str(record.status) != "COMPLETED":
        if intake:
            failure_stage = FAILURE_INTAKE
        elif record.started_at is None:
            failure_stage = FAILURE_QUEUE
        else:
            failure_stage = stage_name(previous_step) or FAILURE_STARTUP
    log.info(
        "AI 작업 종료",
        extra={
            "event": "ai_job_finished",
            "operation": record.kind,
            "trip_id": record.trip_id,
            "job_id": record.execution_id,
            "result": str(record.status),
            "duration_ms": _ms(now - record.created_at),
            "queue_ms": _ms((record.started_at or now) - record.created_at),
            "execution_ms": _ms(now - record.started_at) if record.started_at else 0,
            "failure_stage": failure_stage,
            "error_code": str(error.code) if error else None,
            "error_reason": error.detail.get("reason") if error and error.detail else None,
            "progress_done": record.progress.done,
            "progress_total": record.progress.total,
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

    async def query(self, operation, outcome, seconds, stages=()):
        # 관측 실패가 이미 계산한 검색 응답을 바꾸지 않도록 분리한다.
        # 단계는 phase에 단계 이름, outcome에 단계 결과(success, error, timeout, cancelled)를 둔다.
        try:
            async with asyncio.timeout(0.2):
                async with self.redis.pipeline(transaction=True) as pipe:
                    observe_commands(pipe, self.key, operation, outcome, "request", seconds)
                    for stage in stages:
                        observe_commands(
                            pipe, self.key, operation, stage.result, stage.full_name, stage.seconds
                        )
                    await pipe.execute()
        except (TimeoutError, RedisError):
            log.warning("검색 지표 기록 실패", extra={"event": "ai_metrics_write_failed"})

    async def render(self):
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hgetall(self.key)
            for name in ["queue:process", "queue:story", "running", "pending"]:
                pipe.zcard(self.prefix + name)
            pipe.hgetall(self.prefix + QUALITY_KEY)
            pipe.hgetall(self.prefix + QUALITY_CONTEXT_KEY)
            pipe.hgetall(self.prefix + "metrics:resources")
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
            "Latest labeled evaluation per dataset and unclassified policy, not live accuracy.",
            labels=["feature", "dataset", "unclassified_policy", "metric"],
        )
        for field, value in values[5].items():
            parts = field.split("|")
            # 데이터셋 구분 전의 기록(기능|지표)은 조건을 알 수 없어 노출하지 않는다.
            if len(parts) == 4:
                quality.add_metric(parts, float(value))
        evaluation_info = GaugeMetricFamily(
            "yeodam_ai_evaluation_info",
            "Conditions for the latest labeled evaluation.",
            labels=[
                "feature",
                "dataset",
                "unclassified_policy",
                "model",
                "k",
                "release",
                "source",
            ],
        )
        for series, raw in values[6].items():
            parts = series.split("|")
            if len(parts) != 3:
                continue
            metadata = json.loads(raw)
            evaluation_info.add_metric(
                parts
                + [
                    str(metadata.get(name)) if metadata.get(name) is not None else "none"
                    for name in ["model", "k", "release", "source"]
                ],
                1,
            )

        memory = GaugeMetricFamily(
            "yeodam_ai_job_peak_memory_bytes",
            "Peak resident memory of the most recent job process, per operation.",
            labels=["operation"],
        )
        for field, value in values[7].items():
            operation, name = field.split("|", 1)
            if name == "peak_memory_bytes":
                memory.add_metric([operation], float(value))

        class Snapshot:
            def collect(self):
                return iter([histogram, queue, photos, blocks, quality, evaluation_info, memory])

        registry = CollectorRegistry()
        registry.register(Snapshot())
        # 이 API 인스턴스의 메모리와 CPU. Redis 공유값이 아니므로 인스턴스별로 본다.
        ProcessCollector(registry=registry)
        return generate_latest(registry)
