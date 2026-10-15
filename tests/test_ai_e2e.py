"""Kafka와 Redis, API 두 프로세스, 정리 실행기 및 실제 SDK HTTP 연결을 검증한다."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from pydantic import SecretStr

from app.embedding.client import MODEL_VERSION
from scripts.check_v2_e2e import close_clients, run, validate_case
from tests.e2e_upstream import Upstream, input_case, serve
from tests.test_v2_jobs import environment, scenario
from tests.test_v2_jobs_integration import eventually, free_port, stop


def test_e2e_input_rejects_unrelated_candidate_before_publishing():
    case = input_case()
    validate_case(case)
    case["search"]["candidates"][0]["trip_attachment_id"] = 999
    with pytest.raises(ValueError, match="검색 후보"):
        validate_case(case)


@pytest.mark.parametrize("feature", ["search", "story"])
def test_same_photo_id_must_use_same_bytes_across_features(feature):
    case = input_case()
    photos = (
        case["search"]["candidates"]
        if feature == "search"
        else case["story"]["places"][0]["attachments"]
    )
    photos[0]["analyze_storage_key"], photos[1]["analyze_storage_key"] = (
        photos[1]["analyze_storage_key"],
        photos[0]["analyze_storage_key"],
    )
    with pytest.raises(ValueError, match="같은 사진 ID"):
        validate_case(case)


@scenario
async def test_client_shutdown_is_bounded_and_attempts_both_clients():
    from unittest.mock import AsyncMock

    first = AsyncMock()
    first.stop.side_effect = RuntimeError()
    second = AsyncMock()
    cancelled = asyncio.Event()

    async def hung():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    second.stop.side_effect = hung
    errors = await asyncio.wait_for(close_clients([first, second], timeout=0.01), 1)
    assert errors == ["RuntimeError", "TimeoutError"]
    assert cancelled.is_set()
    first.stop.assert_awaited_once()
    second.stop.assert_awaited_once()


@scenario
async def test_missing_topics_fail_before_request_producer_creation(monkeypatch):
    from unittest.mock import AsyncMock, Mock

    from scripts import check_v2_e2e

    admin = AsyncMock()
    admin.list_topics.return_value = []
    monkeypatch.setattr(check_v2_e2e, "AIOKafkaAdminClient", lambda **_: admin)
    producer = Mock()
    monkeypatch.setattr(check_v2_e2e, "AIOKafkaProducer", producer)
    async with environment() as (cfg, _, _, _):
        with pytest.raises(ValueError, match="업무 토픽"):
            await check_v2_e2e.run(cfg, ["http://localhost"], input_case())
    producer.assert_not_called()
    admin.list_topics.assert_awaited_once_with()
    admin.close.assert_awaited_once()


def test_failed_cli_keeps_execution_id_and_omits_external_error_text(tmp_path, monkeypatch):
    from scripts import check_v2_e2e

    source, output = tmp_path / "case.json", tmp_path / "result.json"
    source.write_text(json.dumps(input_case()))

    async def fail(cfg, urls, case, *, report):
        report.update(phase="process", steps={"process": {"execution_id": "test-execution"}})
        raise TimeoutError("secret-url-token")

    monkeypatch.setattr(check_v2_e2e, "JobsSettings", lambda: None)
    monkeypatch.setattr(check_v2_e2e, "run", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        ["check", str(source), "--api-url", "http://localhost", "--output", str(output)],
    )
    with pytest.raises(SystemExit) as error:
        check_v2_e2e.main()
    report = json.loads(output.read_text())
    assert error.value.code == 1 and report["status"] == "failed"
    assert report["steps"]["process"]["execution_id"] == "test-execution"
    assert report["error_type"] == "TimeoutError" and "secret" not in output.read_text()


@pytest.mark.skipif(
    not (os.environ.get("JOBS_TEST_REDIS_URL") and os.environ.get("JOBS_TEST_KAFKA_URL")),
    reason="실제 Kafka와 Redis 주소를 함께 지정해야 함",
)
@scenario
async def test_all_features_use_real_handlers_and_http_clients(tmp_path):
    upstream = Upstream()
    async with serve(upstream.app) as base, environment() as (cfg, _, _, _):
        cfg.IMAGE_SOURCE, cfg.S3_BUCKET = "s3", "test-photos"
        cfg.KAFKA_BOOTSTRAP_SERVERS = os.environ["JOBS_TEST_KAFKA_URL"]
        cfg.SEARCH_ENABLED = True
        cfg.JOBS_FAKE_PIPELINE = cfg.FAKE_PIPELINE = False
        cfg.JOBS_PROCESS_HANDLER = "app.jobs.process_handler:run"
        cfg.JOBS_STORY_HANDLER = "app.story.handler:run"
        cfg.EMBED_MODEL = MODEL_VERSION
        cfg.EMBEDDING_BASE_URL, cfg.EMBEDDING_SERVED_MODEL_NAME = base + "/v1", "test-qwen"
        cfg.VLM_BASE_URL, cfg.VLM_MODEL = base + "/v1", "test-vlm"
        cfg.QDRANT_URL = base
        cfg.JOBS_POLL_SEC, cfg.JOBS_HEARTBEAT_SEC = 0.05, 0.5
        cfg.LOG_LEVEL = "INFO"
        child_env = os.environ.copy()
        for name, value in cfg:
            if value is None:
                child_env.pop(name, None)
            else:
                child_env[name] = (
                    value.get_secret_value() if isinstance(value, SecretStr) else str(value)
                )
        child_env.update(
            AWS_ACCESS_KEY_ID="testing",
            AWS_SECRET_ACCESS_KEY="testing",
            AWS_EC2_METADATA_DISABLED="true",
            AWS_ENDPOINT_URL_S3=base,
        )
        child_env.pop("AWS_SESSION_TOKEN", None)
        topics = list(cfg.topics) + [route[1] for route in cfg.topics.values()]
        topics += [topic + "-dlt" for topic in topics.copy()]
        admin = AIOKafkaAdminClient(bootstrap_servers=cfg.KAFKA_BOOTSTRAP_SERVERS)
        processes, logs = [], []
        topics_created = False
        repo = Path(__file__).resolve().parents[1]

        def launch(name, *args):
            log = (tmp_path / f"{name}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [sys.executable, *args], cwd=repo, env=child_env, stdout=log, stderr=log
                )
            )

        await admin.start()
        try:
            await admin.create_topics([NewTopic(topic, 2, 1) for topic in topics])
            topics_created = True
            urls = []
            for i in range(2):
                port = free_port()
                urls.append(f"http://127.0.0.1:{port}")
                launch(
                    f"api-{i}",
                    "-c",
                    "import uvicorn; from app.v2_main import create_app; "
                    "from app.jobs.config import JobsSettings; "
                    "uvicorn.run(create_app(JobsSettings(_env_file=None)), "
                    f"host='127.0.0.1', port={port})",
                )
            launch("runner", "-m", "app.jobs.runner")
            async with httpx.AsyncClient(timeout=1) as http:
                for url in urls:

                    async def healthy(target=url):
                        try:
                            return (await http.get(target + "/health")).status_code == 200
                        except httpx.HTTPError:
                            return False

                    await eventually(healthy)
            report = await run(cfg, urls, input_case(), poll_sec=0.1)
            assert report["status"] == "passed"
            assert upstream.calls["image_embedding"] == 2
            assert upstream.calls["text_embedding"] == 1
            assert (
                upstream.calls["parse"] == upstream.calls["story"] == upstream.calls["answer"] == 1
            )
            assert upstream.calls["vlm_image_read"] == 4
            assert upstream.calls["upsert"] and upstream.calls["fetch_vectors"]
            report.update(
                real_redis=True,
                real_kafka=True,
                real_model=False,
                real_s3=False,
                real_qdrant_server=False,
                real_backend=False,
                upstream_http_calls=dict(upstream.calls),
            )
            (tmp_path / "e2e-report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2)
            )
            print(json.dumps(report, ensure_ascii=False))
        finally:
            for process in reversed(processes):
                await stop(process)
            for log in logs:
                log.close()
            if topics_created:
                await admin.delete_topics(topics)
            await admin.close()
            upstream.vectors.close()


@scenario
async def test_test_topics_do_not_inherit_shared_topic_names(monkeypatch):
    monkeypatch.setenv("KAFKA_TRIP_CREATE_REQUEST_TOPIC", "shared.yeodam.trip-create.request")
    async with environment() as (cfg, _, _, _):
        for request, (_, result) in cfg.topics.items():
            assert request.startswith(cfg.JOBS_ENV + ".")
            assert result.startswith(cfg.JOBS_ENV + ".")
            assert "shared" not in request and "shared" not in result
