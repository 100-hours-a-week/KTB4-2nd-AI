"""자원과 처리량 기록. 작업 프로세스 최대 메모리, 처리량, 자원 지표 노출을 확인한다."""

from app.jobs.execution import execute
from app.jobs.kafka import RequestReceiver
from app.jobs.mock import run as mock_run
from app.observability.metrics import Metrics
from tests.observability_helpers import events, info_logs  # noqa: F401
from tests.test_v2_jobs import environment, message, process_request, scenario, story_request


@scenario
async def test_job_summary_records_peak_memory_and_throughput(caplog):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")

        async def steps(request, reporter):
            await reporter.report("DOWNLOADING", 1, 1)
            await reporter.report("EMBEDDING", 1, 1)
            return await mock_run(request, reporter)

        await execute(a, job.identity, job.owner, steps)
        summary = events(caplog, "ai_job_summary")[0]
        assert summary.max_rss_mb > 0 and summary.cpu_user_sec >= 0
        download = next(r for r in events(caplog, "ai_stage") if r.stage == "download")
        assert download.items_per_sec is not None
        resources = await a.redis.hgetall(a.key("metrics:resources"))
        assert int(resources["process|peak_memory_bytes"]) > 0
        text = (await Metrics(a.redis, a.prefix).render()).decode()
        assert 'yeodam_ai_job_peak_memory_bytes{operation="process"}' in text


@scenario
async def test_story_progress_has_no_photo_throughput(caplog):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, story_request()))
        job = await a.claim("story", "worker")
        await execute(a, job.identity, job.owner, mock_run)
        assert all(r.items_per_sec is None for r in events(caplog, "ai_stage"))
