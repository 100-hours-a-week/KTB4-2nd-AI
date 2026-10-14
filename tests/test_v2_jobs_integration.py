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
from app.jobs.kafka import RequestReceiver, consume_requests
from tests.test_v2_jobs import environment, process_request, scenario, story_request

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
