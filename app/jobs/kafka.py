"""Kafka는 접수와 결과 전달만 담당한다. 장시간 분석은 소비 루프에서 실행하지 않는다."""

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.errors import (
    CommitFailedError,
    IllegalStateError,
    KafkaError,
    MessageSizeTooLargeError,
)
from pydantic import ValidationError
from redis.exceptions import RedisError, WatchError

from app.core.errors import ErrorCode
from app.core.logging import get_logger
from app.jobs.config import JobsSettings
from app.jobs.models import MESSAGE, JobRecord, PhotosReadyMessage
from app.jobs.redis_store import RedisJobStore
from app.schemas.task import ErrorBody, Progress

log = get_logger(__name__)


class Producer(Protocol):
    async def send_and_wait(
        self,
        topic: str,
        value: bytes,
        *,
        key: bytes | None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class Incoming:
    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes | None


def connection_options(cfg: JobsSettings) -> dict[str, Any]:
    return {
        "bootstrap_servers": cfg.KAFKA_BOOTSTRAP_SERVERS,
        "security_protocol": cfg.KAFKA_SECURITY_PROTOCOL,
    }


class InvalidMessage(ValueError):
    """DLT 격리 전 같은 원문을 다시 검증한다. 연결 오류와 구분한다."""


class RequestReceiver:
    def __init__(self, store: RedisJobStore, producer: Producer) -> None:
        self.store = store
        self.producer = producer

    async def accept(self, message: Incoming) -> str:
        route = self.store.cfg.topics.get(message.topic)
        if route is None:
            raise ValueError("허용하지 않은 요청 토픽")
        kind, result_topic = route
        try:
            data = json.loads(message.value or b"")
            # JSON escape로 들어온 단독 surrogate와 비유한 숫자도 걸러낸다.
            # 접수 뒤 Redis 직렬화나 실행 ID 인코딩에서 소비 루프를 종료시키면 안 된다.
            json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, UnicodeError, RecursionError):
            return await self.reject(message, "invalid_json")
        if not isinstance(data, dict):
            return await self.reject(message, "invalid_envelope")
        trip_id, execution_id = data.get("trip_id"), data.get("execution_id")
        allowed_types = ("process", "photos_ready") if kind == "process" else ("story",)
        if (
            type(trip_id) is not int
            or trip_id <= 0
            or not isinstance(execution_id, str)
            or not 1 <= len(execution_id) <= 128
            or data.get("type") not in allowed_types
            or message.key != str(trip_id).encode()
        ):
            return await self.reject(message, "invalid_identity_or_route")
        if data["type"] == "photos_ready":
            try:
                batch = PhotosReadyMessage.model_validate(data)
                outcome = await self.store.stage_photos(batch, message.topic)
            except (ValidationError, ValueError):
                # 보조 알림 오류로 본 작업을 FAILED 처리하면 뒤의 정상 process가 차단된다.
                return await self.reject(message, "invalid_photos_ready")
            log.info(
                "사진 묶음 수신, 최종 process 대기",
                extra={
                    "event": "ai_photos_ready_received",
                    "result": outcome,
                    "trip_id": trip_id,
                    "execution_id": execution_id,
                    "batch_no": batch.batch_no,
                    "attachment_count": len(batch.attachments),
                },
            )
            return outcome
        error = None
        try:
            request = MESSAGE.validate_python(data)
            data = request.model_dump(mode="json")
        except ValidationError:
            # 식별 가능한 잘못된 본문은 기존 INVALID_REQUEST -> FAILED 계약을 지킨다.
            error = ErrorBody(
                code=ErrorCode.INVALID_REQUEST, message="요청 형식이 올바르지 않습니다."
            )
        attachments = data.get("attachments")
        total = 4 if kind == "story" else len(attachments) if isinstance(attachments, list) else 0
        record = JobRecord(
            trip_id=trip_id,
            kind=kind,
            execution_id=execution_id,
            request=data,
            request_topic=message.topic,
            result_topic=result_topic,
            created_at=0,
            progress=Progress(done=0, total=total),
        )
        if error:
            self.store.terminate(record, "FAILED", error=error)
        outcome = await self.store.register(record)
        log.info(
            "작업 요청 처리",
            extra={
                "event": "ai_job_received",
                "result": outcome,
                "trip_id": trip_id,
                "execution_id": execution_id,
            },
        )
        return outcome

    async def reject(self, message: Incoming, reason: str) -> str:
        source = f"{message.topic}:{message.partition}:{message.offset}"
        key = self.store.key("invalid:" + hashlib.sha256(source.encode()).hexdigest())
        # 원문에 실행 ID가 없어도 복제본과 재시작 사이에서 실패 횟수를 공유한다.
        while True:
            async with self.store.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    attempts = min(3, int(await pipe.get(key) or 0) + 1)
                    pipe.multi()
                    pipe.set(key, attempts, ex=30 * 86400)
                    await pipe.execute()
                    break
                except WatchError:
                    continue
        if attempts < 3:
            raise InvalidMessage(reason)
        return await self.quarantine(message, reason)

    async def quarantine(self, message: Incoming, reason: str) -> str:
        # 전송 실패는 형식 검증 실패 횟수와 구분하며 원본 offset을 커밋하지 않는다.
        await asyncio.wait_for(
            self.producer.send_and_wait(
                message.topic + "-dlt",
                message.value or b"",
                key=message.key,
                headers=[
                    ("ai-error", reason.encode()),
                    ("original-topic", message.topic.encode()),
                    ("original-partition", str(message.partition).encode()),
                    ("original-offset", str(message.offset).encode()),
                    ("failure", b"ValueError"),
                    ("attempts", b"3"),
                ],
            ),
            timeout=30,
        )
        log.error(
            "처리할 수 없는 요청을 DLT에 보관",
            extra={
                "event": "kafka_dlt_published",
                "service": "ai",
                "dlt_topic": message.topic + "-dlt",
                "source_topic": message.topic,
                "trip_id": message.key.decode("utf-8", errors="replace") if message.key else None,
                "failure": "ValueError",
                "attempts": 3,
            },
        )
        return "quarantined"


async def consume_requests(consumer: AIOKafkaConsumer, receiver: RequestReceiver) -> None:
    while True:
        message = await consumer.getone()
        incoming = Incoming(
            message.topic, message.partition, message.offset, message.key, message.value
        )
        while True:
            try:
                await receiver.accept(incoming)
                # 등록 실패 시 offset을 진행하지 않는다. Redis와 offset은 별도 트랜잭션이다.
                await consumer.commit(
                    {TopicPartition(message.topic, message.partition): message.offset + 1}
                )
                break
            except (CommitFailedError, IllegalStateError):
                # 재배치 뒤 재전달은 Redis의 실행 ID로 중복을 제거한다.
                log.warning("재배치로 접수 offset 확인 실패", extra={"event": "ai_job_rebalance"})
                break
            except InvalidMessage:
                await asyncio.sleep(1)
            except MessageSizeTooLargeError:
                # DLT 헤더가 붙으면 원문이 들어온 제한보다 큰 공간이 필요할 수 있다.
                # 다른 파티션은 계속 받고 이 원문은 커밋하지 않는다. 크기 설정을
                # 조정한 뒤 소비자를 재시작한다. resume해도 건너뛰지 않도록 되돌린다.
                partition = TopicPartition(message.topic, message.partition)
                assigned = partition in consumer.assignment()
                if assigned:
                    consumer.seek(partition, message.offset)
                    consumer.pause(partition)
                log.error(
                    "DLT 크기 제한 초과, 원본 보존 및 해당 파티션 접수 중지",
                    extra={
                        "event": "ai_kafka_dlt_too_large",
                        "service": "ai",
                        "dlt_topic": message.topic + "-dlt",
                        "source_topic": message.topic,
                        "partition": message.partition,
                        "offset": message.offset,
                        "payload_bytes": len(message.value or b""),
                        "partition_paused": assigned,
                    },
                )
                break
            except (RedisError, KafkaError, OSError):
                log.warning(
                    "접수 저장 또는 전달 실패, 같은 메시지 재시도",
                    extra={"event": "ai_job_intake_retry"},
                )
                await asyncio.sleep(1)


async def publish_results(store: RedisJobStore, producer: Producer) -> None:
    publisher = uuid.uuid4().hex
    while True:
        try:
            for identity in await store.pending():
                record = await store.claim_result(identity, publisher)
                if record is None:
                    continue
                payload = record.event.model_dump_json().encode()
                try:
                    await asyncio.wait_for(
                        producer.send_and_wait(
                            record.result_topic,
                            payload,
                            key=str(record.trip_id).encode(),
                        ),
                        timeout=store.cfg.JOBS_PUBLISH_LEASE_SEC / 2,
                    )
                    # 브로커 확인 뒤에만 전송 완료. 여기서 장애가 나면 결과는 재전달될 수 있다.
                    await store.acknowledge_result(identity, publisher)
                    log.info(
                        "결과 발행 완료",
                        extra={
                            "event": "ai_job_result_published",
                            "execution_id": record.execution_id,
                            "payload_bytes": len(payload),
                        },
                    )
                except MessageSizeTooLargeError:
                    # 압축 전 제한도 존재한다. 결과를 버리거나 분석을 다시 실행하지 않는다.
                    log.error(
                        "Kafka 메시지 크기 제한 초과, 결과 보관 유지",
                        extra={
                            "event": "ai_job_result_too_large",
                            "execution_id": record.execution_id,
                            "payload_bytes": len(payload),
                        },
                    )
                except (KafkaError, RedisError, OSError):
                    log.warning(
                        "결과 전달 대기 유지",
                        extra={"event": "ai_job_result_retry", "execution_id": record.execution_id},
                    )
        except (RedisError, OSError):
            log.warning("작업 저장소 재연결 대기", extra={"event": "ai_job_store_retry"})
        await asyncio.sleep(store.cfg.JOBS_POLL_SEC)


async def recover_jobs(store: RedisJobStore) -> None:
    """브로커 발행이 지연되어도 실행기 사망과 시간 초과 감시는 계속한다."""
    while True:
        try:
            await store.recover()
        except (RedisError, OSError):
            log.warning("작업 감시 재연결 대기", extra={"event": "ai_job_recovery_retry"})
        await asyncio.sleep(store.cfg.WATCHDOG_INTERVAL_SEC)


async def close_client(client) -> None:
    # 브로커 단절 중 producer flush를 기다리며 HTTP 종료가 막히지 않도록 한다.
    try:
        await asyncio.wait_for(client.stop(), timeout=5)
    except (TimeoutError, KafkaError, OSError):
        log.warning("Kafka 연결 종료 시간 초과", extra={"event": "ai_kafka_close_retry"})


class KafkaRuntime:
    def __init__(self, store: RedisJobStore) -> None:
        self.store = store
        self.producer: AIOKafkaProducer | None = None
        self.ready = asyncio.Event()
        self.tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        # 네트워크 연결은 각 루프에서 시도한다. Kafka 없이도 HTTP와 작업 감시가 기동한다.
        self.tasks = [
            asyncio.create_task(self.connect_producer()),
            asyncio.create_task(publish_results(self.store, self)),
            asyncio.create_task(recover_jobs(self.store)),
            *[
                asyncio.create_task(self.connect_consumer(topic, group))
                for topic, group in self.store.cfg.consumer_groups.items()
            ],
        ]

    async def connect_producer(self) -> None:
        while True:
            producer = AIOKafkaProducer(
                **connection_options(self.store.cfg),
                acks="all",
                enable_idempotence=True,
                compression_type="lz4",
                request_timeout_ms=30000,
            )
            try:
                await producer.start()
                self.producer = producer
                self.ready.set()
                log.info("Kafka 발행 연결 준비", extra={"event": "ai_kafka_producer_ready"})
                # 시작 후 일시적인 단절은 aiokafka가 재연결한다.
                await asyncio.Future()
            except (KafkaError, OSError):
                log.warning("Kafka 발행 연결 재시도", extra={"event": "ai_kafka_connect_retry"})
            finally:
                self.ready.clear()
                self.producer = None
                await close_client(producer)
            await asyncio.sleep(1)

    async def connect_consumer(self, topic: str, group: str) -> None:
        while True:
            consumer = AIOKafkaConsumer(
                topic,
                **connection_options(self.store.cfg),
                group_id=group,
                enable_auto_commit=False,
                auto_offset_reset="earliest",
            )
            try:
                await consumer.start()
                log.info(
                    "Kafka 소비 연결 준비",
                    extra={
                        "event": "ai_kafka_consumer_ready",
                        "source_topic": topic,
                        "consumer_group": group,
                    },
                )
                # 루프 하나에서 한 메시지씩 등록과 커밋을 마친다.
                await consume_requests(consumer, RequestReceiver(self.store, self))
            except (KafkaError, OSError):
                log.warning(
                    "Kafka 소비 연결 재시도",
                    extra={
                        "event": "ai_kafka_connect_retry",
                        "consumer_group": group,
                    },
                )
            finally:
                await close_client(consumer)
            await asyncio.sleep(1)

    async def send_and_wait(self, topic, value, *, key, headers=None):
        await self.ready.wait()
        producer = self.producer
        if producer is None:
            raise OSError("Kafka 발행 연결 대기 중")
        return await producer.send_and_wait(topic, value, key=key, headers=headers)

    def healthy(self) -> bool:
        # 연결 상태가 아니라 내부 루프의 예기치 않은 종료만 검사한다.
        return bool(self.tasks) and all(not task.done() for task in self.tasks)

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
