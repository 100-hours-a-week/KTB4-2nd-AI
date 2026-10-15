"""테스트용 BE 입력으로 정리, 검색, 스토리의 실제 연결을 차례로 확인한다."""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient

from app.jobs.config import JobsSettings
from app.jobs.models import RESULT_MAX_BYTES, JobResultEvent, ProcessMessage, StoryMessage
from app.schemas.query import ParseRequest, ParseResponse, SearchRequest, SearchResponse


def validate_case(case: dict) -> None:
    process = ProcessMessage.model_validate(case["process"])
    story = StoryMessage.model_validate(case["story"])
    parsed = ParseRequest.model_validate(case["parse"])
    search = SearchRequest.model_validate(case["search"])
    photos = {item.trip_attachment_id: item.analyze_storage_key for item in process.attachments}
    if process.trip_id != story.trip_id or not search.candidates:
        raise ValueError("같은 테스트 여행과 검색 후보가 필요합니다.")
    if not {item.trip_attachment_id for item in search.candidates}.issubset(photos):
        raise ValueError("검색 후보는 테스트 정리 사진에서 선택해야 합니다.")
    if not {
        photo.trip_attachment_id for place in story.places for photo in place.attachments
    }.issubset(photos):
        raise ValueError("스토리 사진은 테스트 정리 사진에서 선택해야 합니다.")
    references = search.candidates + [
        photo for place in story.places for photo in place.attachments
    ]
    if any(photos[item.trip_attachment_id] != item.analyze_storage_key for item in references):
        raise ValueError("같은 사진 ID에는 같은 S3 사본 경로를 사용해야 합니다.")
    if parsed.query != search.query:
        raise ValueError("질의 해석과 검색에는 같은 질문을 사용해야 합니다.")


async def close_clients(clients, timeout=10):
    results = await asyncio.gather(
        *(asyncio.wait_for(client.stop(), timeout) for client in clients), return_exceptions=True
    )
    return [type(result).__name__ for result in results if isinstance(result, BaseException)]


async def check_topics(cfg):
    admin = AIOKafkaAdminClient(
        bootstrap_servers=cfg.KAFKA_BOOTSTRAP_SERVERS,
        security_protocol=cfg.KAFKA_SECURITY_PROTOCOL,
    )
    try:
        await admin.start()
        # 전체 목록 조회에는 개별 토픽명을 보내지 않는다. 잘못된 이름의 자동 생성을 피한다.
        existing = set(await admin.list_topics())
        required = set(cfg.topics) | {route[1] for route in cfg.topics.values()}
        if not required.issubset(existing):
            raise ValueError("업무 토픽 6개를 먼저 생성하고 이름을 확인해야 합니다.")
    finally:
        await asyncio.wait_for(admin.close(), 10)


async def run(
    cfg: JobsSettings, api_urls: list[str], case: dict, *, poll_sec=2.0, report=None
) -> dict:
    report = {} if report is None else report
    report.update(status="running", phase="configuration", api_replicas=len(api_urls), steps={})
    validate_case(case)
    if not api_urls:
        raise ValueError("API 주소가 필요합니다.")
    report["phase"] = "kafka_topics"
    async with asyncio.timeout(30):
        await check_topics(cfg)
    result_topics = [route[1] for route in cfg.topics.values()]
    consumer = AIOKafkaConsumer(
        *result_topics,
        bootstrap_servers=cfg.KAFKA_BOOTSTRAP_SERVERS,
        security_protocol=cfg.KAFKA_SECURITY_PROTOCOL,
        group_id="yeodam-ai-e2e-" + uuid.uuid4().hex,
        auto_offset_reset="latest",
        enable_auto_commit=False,
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=cfg.KAFKA_BOOTSTRAP_SERVERS,
        security_protocol=cfg.KAFKA_SECURITY_PROTOCOL,
        enable_idempotence=True,
        acks="all",
        compression_type="lz4",
    )
    auth = {"Authorization": "Bearer " + cfg.API_KEY}
    try:
        report["phase"] = "kafka_connect"
        async with asyncio.timeout(30):
            await producer.start()
            await consumer.start()
            while not consumer.assignment():
                await consumer.getmany(timeout_ms=100)
            # latest의 시작 위치를 요청 발행 전에 확정한다.
            for partition in consumer.assignment():
                await consumer.position(partition)
        async with httpx.AsyncClient(headers=auth, timeout=60) as http:
            for url in api_urls:
                (await http.get(url.rstrip("/") + "/health")).raise_for_status()

            async def job(kind, body, topic):
                body = dict(body, execution_id="e2e-" + uuid.uuid4().hex)
                started = time.perf_counter()
                report["phase"] = kind
                report["steps"][kind] = {
                    "trip_id": body["trip_id"],
                    "execution_id": body["execution_id"],
                    "status": "PUBLISHING",
                }
                await asyncio.wait_for(
                    producer.send_and_wait(
                        topic,
                        json.dumps(body, ensure_ascii=False).encode(),
                        key=str(body["trip_id"]).encode(),
                    ),
                    30,
                )
                states = set()
                expected_topic = cfg.topics[topic][1]
                event = None
                async with asyncio.timeout(cfg.JOBS_TIMEOUT_SEC + 60):
                    while event is None:
                        for url in api_urls:
                            response = await http.get(
                                f"{url.rstrip('/')}/trips/{body['trip_id']}/{kind}"
                            )
                            if response.status_code == 404:
                                continue
                            response.raise_for_status()
                            state = response.json()
                            if state["execution_id"] == body["execution_id"]:
                                states.add(state["status"])
                        batches = await consumer.getmany(timeout_ms=max(1, int(poll_sec * 1000)))
                        for messages in batches.values():
                            for received in messages:
                                payload = json.loads(received.value)
                                if payload.get("execution_id") != body["execution_id"]:
                                    continue
                                event = JobResultEvent.model_validate(payload)
                                if (
                                    received.topic != expected_topic
                                    or received.key != str(body["trip_id"]).encode()
                                    or event.trip_id != body["trip_id"]
                                    or event.type != f"{kind}_result"
                                    or len(received.value) > RESULT_MAX_BYTES
                                ):
                                    raise ValueError("결과 토픽 또는 메시지 계약 불일치")
                if event.status != "COMPLETED":
                    code = event.error.code if event.error else "UNKNOWN"
                    report["steps"][kind].update(status=event.status, error_code=str(code))
                    raise ValueError(f"{kind} 작업 실패: {code}")
                for url in api_urls:
                    response = await http.get(f"{url.rstrip('/')}/trips/{body['trip_id']}/{kind}")
                    response.raise_for_status()
                    state = response.json()
                    if (
                        state["execution_id"] != body["execution_id"]
                        or state["status"] != "COMPLETED"
                    ):
                        raise ValueError("API 복제본의 최종 상태 불일치")
                report["steps"][kind].update(
                    {
                        "seconds": round(time.perf_counter() - started, 3),
                        "states_observed": sorted(states),
                        "status": event.status,
                    }
                )
                return event.result

            process_topic = (
                cfg.KAFKA_TRIP_UPDATE_REQUEST_TOPIC
                if case["process"].get("trigger") == "PHOTOS_ADDED"
                else cfg.KAFKA_TRIP_CREATE_REQUEST_TOPIC
            )
            organized = await job("process", case["process"], process_topic)
            active = {
                photo["trip_attachment_id"]
                for place in organized["places"]
                for photo in place["attachments"]
            }
            candidate_ids = {photo["trip_attachment_id"] for photo in case["search"]["candidates"]}
            if not candidate_ids.issubset(active):
                raise ValueError("테스트 검색 후보에 정리되지 않은 사진이 포함되어 있습니다.")
            story_ids = {
                photo["trip_attachment_id"]
                for place in case["story"]["places"]
                for photo in place["attachments"]
            }
            if not story_ids.issubset(active):
                raise ValueError("테스트 스토리 입력에 정리되지 않은 사진이 포함되어 있습니다.")

            started = time.perf_counter()
            report["phase"] = "search"
            response = await http.post(api_urls[0].rstrip("/") + "/query/parse", json=case["parse"])
            response.raise_for_status()
            parsed = ParseResponse.model_validate(response.json())
            if not parsed.parsed:
                raise ValueError("실제 VLM 질의 해석이 대체 응답으로 끝났습니다.")
            search = dict(case["search"], intent=parsed.intent, visual_query=parsed.visual_query)
            response = await http.post(api_urls[-1].rstrip("/") + "/query/search", json=search)
            response.raise_for_status()
            found = SearchResponse.model_validate(response.json())
            if not found.attachments or not {
                hit.trip_attachment_id for hit in found.attachments
            }.issubset(candidate_ids):
                raise ValueError("후보 범위 안에서 검색 근거를 찾지 못했습니다.")
            if parsed.intent == "QUESTION" and (not found.answer or found.answer_error):
                raise ValueError("질문 답변 생성 실패")
            report["steps"]["search"] = {
                "seconds": round(time.perf_counter() - started, 3),
                "intent": parsed.intent,
                "returned": len(found.attachments),
            }
            story = await job("story", case["story"], cfg.KAFKA_STORY_CREATE_REQUEST_TOPIC)
            if not story["blocks"] or not story["verification"]["passed"]:
                raise ValueError("스토리 출력 계약 검증 실패")
            for url in api_urls:
                report["phase"] = "metrics"
                metrics = await http.get(url.rstrip("/") + "/metrics")
                metrics.raise_for_status()
                if "yeodam_ai_operation_duration_seconds_count" not in metrics.text:
                    raise ValueError("완료 작업 지표 누락")
            report["status"] = "passed"
            report["phase"] = "completed"
            return report
    finally:
        errors = await close_clients([consumer, producer])
        if errors:
            report["cleanup_errors"] = errors
            if report["status"] == "passed":
                raise RuntimeError("Kafka 클라이언트 종료 실패")


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--api-url", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "running", "phase": "configuration", "steps": {}}
    try:
        asyncio.run(
            run(JobsSettings(), args.api_url, json.loads(args.case.read_text()), report=report)
        )
    except Exception as exc:
        # 외부 예외 본문에는 URL과 자격 증명이 섞일 수 있어 종류와 실행 ID만 남긴다.
        report.update(status="failed", error_type=type(exc).__name__)
        save_report(args.output, report)
        raise SystemExit(1) from None
    save_report(args.output, report)


if __name__ == "__main__":
    main()
