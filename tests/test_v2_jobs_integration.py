"""실제 Kafka, Redis와 HTTP 프로세스 검증. BE와 모델은 테스트용 구현이다."""

import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from pydantic import SecretStr

from app.core.errors import AppError
from app.jobs.execution import execute
from app.jobs.kafka import RequestReceiver, consume_requests, publish_results
from app.jobs.mock import run as mock_run
from app.jobs.models import RESULT_MAX_BYTES
from app.schemas.task import ErrorBody
from tests.test_photos_ready import buffer_key, ready_request
from tests.test_v2_jobs import environment, message, process_request, scenario, story_request

pytestmark = pytest.mark.skipif(
    not (os.environ.get("JOBS_TEST_REDIS_URL") and os.environ.get("JOBS_TEST_KAFKA_URL")),
    reason="실제 Kafka와 Redis 주소를 함께 지정해야 함",
)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def eventually(check, timeout=40):
    async with asyncio.timeout(timeout):
        while not (result := await check()):
            await asyncio.sleep(0.05)
        return result


async def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=15)
        except TimeoutError:
            process.kill()
            await asyncio.to_thread(process.wait)


@pytest.mark.parametrize("work", ["trip-create", "trip-update"])
@scenario
async def test_real_kafka_stages_batches_then_process_runs_after_consumer_restart(work):
    async with environment() as (cfg, store, replica, _producer):
        address = os.environ["JOBS_TEST_KAFKA_URL"]
        batch = ready_request(ids=(101, 102))
        incoming = message(cfg, batch, work)
        topic = incoming.topic
        result_topic = cfg.topics[topic][1]
        topics = [topic, topic + "-dlt", result_topic]
        admin = AIOKafkaAdminClient(bootstrap_servers=address)
        producer = AIOKafkaProducer(
            bootstrap_servers=address, compression_type="lz4", acks="all", enable_idempotence=True
        )
        readers = [
            AIOKafkaConsumer(
                topic,
                bootstrap_servers=address,
                group_id=cfg.JOBS_ENV + "-photos",
                enable_auto_commit=False,
                auto_offset_reset="earliest",
            )
            for _ in range(2)
        ]
        results = AIOKafkaConsumer(
            result_topic,
            bootstrap_servers=address,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        consumer_task = publisher = None
        await admin.start()
        try:
            await admin.create_topics([NewTopic(name, 2, 1) for name in topics])
            await producer.start()
            await results.start()
            await readers[0].start()
            consumer_task = asyncio.create_task(
                consume_requests(readers[0], RequestReceiver(store, producer))
            )
            first = await producer.send_and_wait(topic, incoming.value, key=b"77")
            partition = TopicPartition(topic, first.partition)

            async def first_committed():
                return await readers[0].committed(partition) == first.offset + 1

            await eventually(first_committed)
            saved = await replica.redis.get(buffer_key(replica, batch))
            assert saved is not None and await replica.claim("process", "runner") is None
            assert await results.getmany(timeout_ms=200) == {}
            consumer_task.cancel()
            await asyncio.gather(consumer_task, return_exceptions=True)
            await readers[0].stop()

            # 다른 API 복제본으로 소비가 옮겨져도 Redis 묶음을 재사용한다.
            await readers[1].start()
            consumer_task = asyncio.create_task(
                consume_requests(readers[1], RequestReceiver(replica, producer))
            )
            duplicate = await producer.send_and_wait(topic, incoming.value, key=b"77")

            async def duplicate_committed():
                return await readers[1].committed(partition) == duplicate.offset + 1

            await eventually(duplicate_committed)
            assert await store.redis.get(buffer_key(store, batch)) == saved
            assert await store.claim("process", "runner") is None

            final = process_request()
            final["trigger"] = "PHOTOS_ADDED" if work == "trip-update" else "TRIP_CREATED"
            # 103의 photos_ready가 없어도 최종 목록의 세 장을 처리한다.
            final["attachments"] = ready_request(ids=(101, 102, 103))["attachments"]
            await producer.send_and_wait(topic, json.dumps(final).encode(), key=b"77")

            async def accepted():
                try:
                    return await store.latest(77, "process")
                except AppError:
                    return None

            await eventually(accepted)
            assert not await store.redis.exists(buffer_key(store, batch))
            job = await store.claim("process", "runner")
            assert job.progress.total == 3
            await execute(store, job.identity, job.owner, mock_run)
            publisher = asyncio.create_task(publish_results(replica, producer))
            async with asyncio.timeout(15):
                result = await results.getone()
            event = json.loads(result.value)
            assert result.key == b"77" and event["type"] == "process_result"
            assert event["status"] == "COMPLETED"
            assert [p["trip_attachment_id"] for p in event["result"]["unclassified"]] == [
                101,
                102,
                103,
            ]
            late = await producer.send_and_wait(topic, incoming.value, key=b"77")

            async def late_committed():
                return await readers[1].committed(partition) == late.offset + 1

            await eventually(late_committed)
            assert not await store.redis.exists(buffer_key(store, batch))
            assert await store.claim("process", "runner") is None
            assert (await replica.latest(77, "process")).status == "COMPLETED"
        finally:
            tasks = [task for task in (consumer_task, publisher) if task]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for reader in readers:
                await reader.stop()
            await results.stop()
            await producer.stop()
            await admin.delete_topics(topics)
            await admin.close()


@pytest.mark.parametrize("kind", ["process", "story"])
@scenario
async def test_real_kafka_result_limit_delivers_failed_event(kind):
    async with environment() as (cfg, store, replica, _producer):
        address = os.environ["JOBS_TEST_KAFKA_URL"]
        body = process_request() if kind == "process" else story_request()
        incoming = message(cfg, body)
        topic = cfg.topics[incoming.topic][1]
        admin = AIOKafkaAdminClient(bootstrap_servers=address)
        producer = AIOKafkaProducer(
            bootstrap_servers=address,
            compression_type="lz4",
            acks="all",
            enable_idempotence=True,
        )
        consumer = AIOKafkaConsumer(
            topic, bootstrap_servers=address, enable_auto_commit=False, auto_offset_reset="earliest"
        )
        publisher = None
        await admin.start()
        try:
            await admin.create_topics([NewTopic(topic, 2, 1)])
            await producer.start()
            await consumer.start()
            await RequestReceiver(store, producer).accept(incoming)
            job = await store.claim(kind, "worker")
            await store.finish(
                job.identity,
                job.owner,
                "FAILED",
                error=ErrorBody(
                    code="DOWNLOAD_FAILED",
                    message="실패",
                    detail={"keys": ["가" * RESULT_MAX_BYTES]},
                ),
            )
            publisher = asyncio.create_task(publish_results(store, producer))
            async with asyncio.timeout(15):
                received = await consumer.getone()
            event = json.loads(received.value)
            assert len(received.value) <= 524288
            assert received.key == b"77" and event["type"] == f"{kind}_result"
            assert event["status"] == "FAILED" and event["result"] is None
            assert event["error"]["code"] == "RESULT_TOO_LARGE"
            assert event["error"]["detail"]["max_bytes"] == 524288
            assert (await replica.latest(77, kind)).response().status == "FAILED"

            async def published():
                return (await replica.get(job.identity)).published

            await eventually(published)
        finally:
            if publisher:
                publisher.cancel()
                await asyncio.gather(publisher, return_exceptions=True)
            await consumer.stop()
            await producer.stop()
            await admin.delete_topics([topic])
            await admin.close()


@scenario
async def test_real_kafka_oversized_dlt_pauses_only_source_and_replays_after_restart(tmp_path):
    async with environment() as (cfg, store, _, _producer):
        address = os.environ["JOBS_TEST_KAFKA_URL"]
        topic = cfg.KAFKA_TRIP_CREATE_REQUEST_TOPIC
        dlt = topic + "-dlt"
        partitions = [TopicPartition(topic, i) for i in range(2)]
        admin = AIOKafkaAdminClient(bootstrap_servers=address)
        producer = AIOKafkaProducer(bootstrap_servers=address, compression_type="lz4")
        enlarged_producer = AIOKafkaProducer(
            bootstrap_servers=address, compression_type="lz4", max_request_size=2 * 1024 * 1024
        )

        def new_consumer():
            consumer = AIOKafkaConsumer(
                bootstrap_servers=address,
                group_id=cfg.JOBS_ENV + "-oversize",
                enable_auto_commit=False,
                auto_offset_reset="earliest",
            )
            consumer.assign(partitions)
            return consumer

        consumers = [new_consumer(), new_consumer()]
        dlt_reader = AIOKafkaConsumer(
            dlt, bootstrap_servers=address, enable_auto_commit=False, auto_offset_reset="earliest"
        )
        task = None
        raw = b"{" + b" " * (1024 * 1024 - 101)
        await admin.start()
        try:
            await admin.create_topics([NewTopic(t, 2, 1) for t in (topic, dlt)])
            await producer.start()
            source = await producer.send_and_wait(topic, raw, key=b"77", partition=0)
            for trip, partition in ((78, 0), (79, 1)):
                await producer.send_and_wait(
                    topic,
                    json.dumps(process_request(f"good-{trip}", trip)).encode(),
                    key=str(trip).encode(),
                    partition=partition,
                )
            await consumers[0].start()
            task = asyncio.create_task(
                consume_requests(consumers[0], RequestReceiver(store, producer))
            )

            async def paused_and_other_partition_committed():
                return (
                    partitions[0] in consumers[0].paused()
                    and (await consumers[0].committed(partitions[1])) == 1
                )

            await eventually(paused_and_other_partition_committed)
            assert not task.done()
            assert await consumers[0].committed(partitions[0]) is None
            assert await consumers[0].position(partitions[0]) == source.offset
            with pytest.raises(AppError) as missing:
                await store.latest(78, "process")
            assert missing.value.http_status == 404
            assert (await store.latest(79, "process")).status == "QUEUED"
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await consumers[0].stop()

            # 크기 설정을 조정한 발행자로 소비를 재시작하면 보존한 원문부터 재처리한다.
            await enlarged_producer.start()
            await consumers[1].start()
            await dlt_reader.start()
            task = asyncio.create_task(
                consume_requests(consumers[1], RequestReceiver(store, enlarged_producer))
            )
            async with asyncio.timeout(15):
                isolated = await dlt_reader.getone()
            assert isolated.value == raw and isolated.key == b"77"
            assert dict(isolated.headers)["attempts"] == b"3"

            async def source_partition_committed():
                return (await consumers[1].committed(partitions[0])) == 2

            await eventually(source_partition_committed)
            assert (await store.latest(78, "process")).status == "QUEUED"
            (tmp_path / "dlt-size-report.json").write_text(
                json.dumps(
                    {
                        "original_json_bytes": len(raw),
                        "initial_producer_max_request_size": 1024 * 1024,
                        "recovery_producer_max_request_size": 2 * 1024 * 1024,
                        "source_offset_preserved": source.offset,
                        "other_partition_committed_during_pause": 1,
                        "source_partition_committed_after_recovery": 2,
                        "dlt_original_preserved": True,
                    },
                    indent=2,
                )
            )
        finally:
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            for client in (*consumers, dlt_reader, producer, enlarged_producer):
                await client.stop()
            await admin.delete_topics([topic, dlt])
            await admin.close()


@scenario
async def test_real_kafka_two_api_rebalance_and_results(tmp_path):
    async with environment(JOBS_HEARTBEAT_SEC=0.1) as (cfg, store, _, _producer):
        cfg.KAFKA_BOOTSTRAP_SERVERS = os.environ["JOBS_TEST_KAFKA_URL"]
        cfg.JOBS_POLL_SEC = 0.02
        admin = AIOKafkaAdminClient(bootstrap_servers=cfg.KAFKA_BOOTSTRAP_SERVERS)
        producer = AIOKafkaProducer(
            bootstrap_servers=cfg.KAFKA_BOOTSTRAP_SERVERS,
            enable_idempotence=True,
            acks="all",
            compression_type="lz4",
        )
        request_topics = list(cfg.topics)
        result_topics = [route[1] for route in cfg.topics.values()]
        business_topics = request_topics + result_topics
        topics = business_topics + [topic + "-dlt" for topic in business_topics]
        consumer = AIOKafkaConsumer(
            *result_topics,
            *[topic + "-dlt" for topic in request_topics],
            bootstrap_servers=cfg.KAFKA_BOOTSTRAP_SERVERS,
            group_id=cfg.JOBS_ENV + "-mock-be",
            auto_offset_reset="earliest",
            enable_auto_commit=False,
        )
        processes, logs = [], []
        report = {"scope": "local Kafka + Redis + two HTTP APIs; mock BE and model handlers"}
        repo = Path(__file__).resolve().parents[1]
        child_env = os.environ.copy()
        for name, value in cfg:
            if value is not None:
                child_env[name] = (
                    value.get_secret_value() if isinstance(value, SecretStr) else str(value)
                )

        def launch(name, *args):
            log = (tmp_path / f"{name}.log").open("w")
            logs.append(log)
            process = subprocess.Popen(
                [sys.executable, *args], cwd=repo, env=child_env, stdout=log, stderr=log
            )
            processes.append(process)
            return process

        def launch_api(name, port):
            code = (
                "import uvicorn; from app.jobs.config import JobsSettings; "
                "from app.v2_main import create_app; "
                f"uvicorn.run(create_app(JobsSettings(_env_file=None)), "
                f"host='127.0.0.1', port={port}, log_level='warning')"
            )
            return launch(name, "-c", code)

        async def group_size(size):
            # aiokafka 0.14의 DescribeGroups v3 디코더는 v2 형식을 참조한다.
            # 그룹 뒤 authorized_operations가 다음 그룹으로 오인되지 않도록 한 개씩 조회한다.
            groups = []
            for group_id in cfg.consumer_groups.values():
                response = await admin.describe_consumer_groups([group_id])
                groups.extend(group for item in response for group in item.groups)
            return len(groups) == 3 and all(
                g[0] == 0 and g[2] == "Stable" and len(g[5]) == size for g in groups
            )

        received = {}
        dlt = []
        sent = []

        async def send(body, topic_index=0, partition=0):
            metadata = await producer.send_and_wait(
                request_topics[topic_index],
                json.dumps(body).encode(),
                key=str(body["trip_id"]).encode(),
                partition=partition,
            )
            sent.append(metadata)

        async def collect(executions, expect_dlt=0):
            async with asyncio.timeout(40):
                while not executions.issubset(received) or len(dlt) < expect_dlt:
                    message = await consumer.getone()
                    if message.topic.endswith("-dlt"):
                        dlt.append(message)
                        continue
                    event = json.loads(message.value)
                    assert set(event) == {
                        "type",
                        "trip_id",
                        "execution_id",
                        "status",
                        "result",
                        "error",
                    }
                    assert message.key == str(event["trip_id"]).encode()
                    if event["execution_id"] in received:
                        # 결과 중복 전달은 허용하지만 같은 실행의 본문은 같아야 한다.
                        assert received[event["execution_id"]][1] == event
                    received[event["execution_id"]] = (message.topic, event)

        await admin.start()
        try:
            await admin.create_topics(
                [
                    NewTopic(
                        name,
                        num_partitions=2,
                        replication_factor=1,
                        topic_configs={
                            "retention.ms": str((30 if name.endswith("-dlt") else 7) * 86400000),
                            "cleanup.policy": "delete",
                        },
                    )
                    for name in topics
                ]
            )
            await producer.start()
            await consumer.start()
            ports = [free_port(), free_port()]
            api1 = launch_api("api1", ports[0])
            launch_api("api2", ports[1])
            async with httpx.AsyncClient(headers={"Authorization": "Bearer test"}) as client:

                async def healthy(port):
                    try:
                        return (
                            await client.get(f"http://127.0.0.1:{port}/health")
                        ).status_code == 200
                    except httpx.HTTPError:
                        return False

                async def state(trip, kind, expected):
                    responses = [
                        await client.get(f"http://127.0.0.1:{port}/trips/{trip}/{kind}")
                        for port in ports
                    ]
                    return all(
                        r.status_code == 200 and r.json()["status"] == expected for r in responses
                    )

                for port in ports:
                    await eventually(lambda port=port: healthy(port))
                await eventually(lambda: group_size(2))
                report["consumer_group_members"] = [2]

                # 정리 실행기를 시작하기 전 대기 취소와 중복 접수를 확인한다.
                canceled = process_request("cancel", 77)
                await send(canceled)
                await send(canceled)
                await eventually(lambda: state(77, "process", "QUEUED"))
                assert await store.redis.zcard(store.key("queue:process")) == 1
                response = await client.delete(f"http://127.0.0.1:{ports[1]}/trips/77/process")
                assert response.status_code == 200 and response.json()["status"] == "CANCELED"
                await eventually(lambda: state(77, "process", "CANCELED"))
                await collect({"cancel"})
                assert received["cancel"][1]["status"] == "CANCELED"

                await send(process_request("create", 78))
                update = process_request("update", 79)
                update["trigger"] = "PHOTOS_ADDED"
                await send(update, 1, 1)
                await send(story_request("story", 80), 2)
                invalid = process_request("invalid", 81)
                invalid.pop("trip_name")
                await send(invalid, partition=1)
                await producer.send_and_wait(request_topics[0], b"not-json", key=b"82")
                launch("process-runner", "-m", "app.jobs.runner")
                await collect({"create", "update", "story", "invalid"}, expect_dlt=1)
                for execution, index in (("create", 0), ("update", 1), ("story", 2)):
                    assert received[execution][0] == result_topics[index]
                    assert received[execution][1]["status"] == "COMPLETED"
                assert received["invalid"][1]["error"]["code"] == "INVALID_REQUEST"
                assert dlt[0].topic == request_topics[0] + "-dlt"
                assert dlt[0].value == b"not-json"
                assert dict(dlt[0].headers)["ai-error"] == b"invalid_json"
                assert dict(dlt[0].headers)["attempts"] == b"3"
                report["request_result_routes"] = ["trip-create", "trip-update", "story-create"]

                # 실제 그룹을 2 -> 1 -> 2로 바꾸고 양쪽 파티션의 새 요청을 처리한다.
                api1.kill()
                await asyncio.to_thread(api1.wait)
                await eventually(lambda: group_size(1))
                report["consumer_group_members"].append(1)
                await send(process_request("after-loss-p0", 83), partition=0)
                await send(process_request("after-loss-p1", 84), partition=1)
                await collect({"after-loss-p0", "after-loss-p1"})
                launch_api("api1-restarted", ports[0])
                await eventually(lambda: healthy(ports[0]))
                await eventually(lambda: group_size(2))
                report["consumer_group_members"].append(2)
                await send(process_request("after-rejoin-p0", 85), partition=0)
                await send(process_request("after-rejoin-p1", 86), partition=1)
                await collect({"after-rejoin-p0", "after-rejoin-p1"})
                for execution in (
                    "after-loss-p0",
                    "after-loss-p1",
                    "after-rejoin-p0",
                    "after-rejoin-p1",
                ):
                    assert received[execution][1]["status"] == "COMPLETED"
                await eventually(lambda: state(78, "process", "COMPLETED"))
                responses = [
                    (await client.get(f"http://127.0.0.1:{port}/trips/78/process")).json()
                    for port in ports
                ]
                assert responses[0] == responses[1] and "result" not in responses[0]

                async def committed():
                    offsets = {}
                    for group in cfg.consumer_groups.values():
                        offsets.update(await admin.list_consumer_group_offsets(group))
                    return all(
                        TopicPartition(m.topic, m.partition) in offsets
                        and offsets[TopicPartition(m.topic, m.partition)].offset >= m.offset + 1
                        for m in sent
                    )

                await eventually(committed)
                report.update(
                    request_offsets_committed=True,
                    status_equal_after_restart=True,
                    queued_cancel=True,
                    duplicate_registration=True,
                    invalid_request_failed=True,
                    malformed_json_dlt=True,
                    terminal_results={k: v[1]["status"] for k, v in received.items()},
                )
                (tmp_path / "integration-report.json").write_text(json.dumps(report, indent=2))
        finally:
            for process in processes:
                await stop(process)
            for log in logs:
                log.close()
            await consumer.stop()
            await producer.stop()
            try:
                await admin.delete_topics(topics)
            finally:
                await admin.close()
