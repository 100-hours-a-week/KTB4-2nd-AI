"""모델과 S3 및 Qdrant의 로컬 HTTP 대역. 실제 모델 품질을 검증하지 않는다."""

import asyncio
import base64
import io
import json
import socket
from collections import Counter
from contextlib import asynccontextmanager

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response
from qdrant_client import QdrantClient, models

from app.embedding.client import DIM


@asynccontextmanager
async def serve(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    sock.setblocking(False)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("HTTP 대역 기동 실패")
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)
        sock.close()


class Upstream:
    def __init__(self):
        from PIL import Image

        self.app, self.calls, self.photos = FastAPI(), Counter(), {}
        self.vectors = QdrantClient(":memory:")
        for i in [101, 102]:
            data = np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(data).save(buf, format="JPEG")
            self.photos[f"trips/77/{i}.jpg"] = buf.getvalue()
        self.routes()

    def routes(self):
        def ok(result):
            return {"result": result, "status": "ok", "time": 0.0}

        @self.app.get("/")
        async def version():
            return {"title": "test Qdrant", "version": "1.19.1"}

        @self.app.get("/readyz")
        async def ready():
            return Response("ready")

        @self.app.get("/collections/{name}/exists")
        async def exists(name: str):
            return ok({"exists": self.vectors.collection_exists(name)})

        @self.app.put("/collections/{name}")
        async def create(name: str, request: Request):
            body = await request.json()
            return ok(
                self.vectors.create_collection(
                    name, vectors_config=models.VectorParams(**body["vectors"])
                )
            )

        @self.app.put("/collections/{name}/points")
        async def upsert(name: str, request: Request):
            self.calls["upsert"] += 1
            body = await request.json()
            result = self.vectors.upsert(name, [models.PointStruct(**p) for p in body["points"]])
            return ok(result.model_dump(mode="json"))

        @self.app.post("/collections/{name}/points")
        async def retrieve(name: str, request: Request):
            self.calls["fetch_vectors"] += 1
            body = await request.json()
            points = self.vectors.retrieve(name, body["ids"], with_vectors=True, with_payload=False)
            return ok([point.model_dump(mode="json") for point in points])

        @self.app.post("/collections/{name}/points/query")
        async def query(name: str, request: Request):
            self.calls["vector_search"] += 1
            body = models.QueryRequest.model_validate(await request.json())
            assert body.filter.must[0].has_id == [101, 102]
            result = self.vectors.query_points(
                name,
                query=body.query,
                query_filter=body.filter,
                limit=body.limit,
                with_payload=False,
            )
            return ok(result.model_dump(mode="json"))

        @self.app.post("/v1/embeddings")
        async def embedding(request: Request):
            body = await request.json()
            assert body["model"] == "test-qwen"
            content = body["messages"][1]["content"][0]
            if content["type"] == "image_url":
                self.calls["image_embedding"] += 1
                raw = base64.b64decode(content["image_url"]["url"].split(",", 1)[1])
                assert raw in self.photos.values()  # 원본 JPEG 바이트 그대로 전달되는지 확인
            else:
                self.calls["text_embedding"] += 1
            return {"data": [{"embedding": [1.0] + [0.0] * (DIM - 1)}]}

        @self.app.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            assert body["model"] == "test-vlm"
            schema = body.get("response_format", {}).get("json_schema", {}).get("name")
            content = body["messages"][-1]["content"]
            if schema == "query_parse":
                self.calls["parse"] += 1
                output = dict(
                    intent="QUESTION",
                    date_from=None,
                    date_to=None,
                    relative_date=None,
                    region_names=None,
                    visual_query="사진",
                )
            else:
                images = [
                    item["image_url"]["url"] for item in content if item["type"] == "image_url"
                ]
                assert images and all("X-Amz-Signature=" in url for url in images)
                async with httpx.AsyncClient() as http:
                    for url in images:
                        image = await http.get(url)
                        assert image.status_code == 200 and image.content in self.photos.values()
                        self.calls["vlm_image_read"] += 1
                # 스토리는 이미지마다 라벨 텍스트가 앞에 붙으므로 사진 정보 JSON은 마지막 텍스트다.
                facts = json.loads(
                    next(item["text"] for item in reversed(content) if item["type"] == "text")
                )
                if schema == "travel_story":
                    self.calls["story"] += 1
                    output = {
                        "story_summary": "사진으로 남긴 여행",
                        "blocks": [
                            {
                                "trip_attachment_id": photo["trip_attachment_id"],
                                "detail_summary": "사진 속 풍경",
                                "memo": "사진에 여러 색이 담겨 있다.",
                            }
                            for photo in facts["photos"]
                        ],
                    }
                else:
                    assert schema == "query_answer"
                    self.calls["answer"] += 1
                    output = {"answerable": True, "answer": "사진에 여러 색이 보입니다."}
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": output
                            if isinstance(output, str)
                            else json.dumps(output, ensure_ascii=False)
                        },
                    }
                ]
            }

        @self.app.get("/{bucket}/{key:path}")
        async def photo(bucket: str, key: str):
            assert bucket == "test-photos"
            self.calls["s3_get"] += 1
            return Response(self.photos[key], media_type="image/jpeg")


def input_case():
    from tests.test_v2_jobs import process_request, story_request

    process, story = process_request(), story_request()
    process["attachments"] = [
        {
            "trip_attachment_id": i,
            "analyze_storage_key": f"trips/77/{i}.jpg",
            "taken_at": f"2026-10-12T{hour}:00:00+09:00",
            "latitude": 33.4,
            "longitude": 126.5,
            "device_model": "test",
        }
        for i, hour in [(101, "06"), (102, "07")]
    ]
    story["places"][0]["attachments"] = [
        {
            key: value
            for key, value in p.items()
            if key in {"trip_attachment_id", "analyze_storage_key", "taken_at"}
        }
        for p in process["attachments"]
    ]
    return {
        "process": process,
        "story": story,
        "parse": {"query": "사진에 무엇이 보이지?", "reference_date": "2026-10-14"},
        "search": {
            "query": "사진에 무엇이 보이지?",
            "intent": "QUESTION",
            "visual_query": "사진",
            "candidates": [
                {
                    "trip_attachment_id": p["trip_attachment_id"],
                    "analyze_storage_key": p["analyze_storage_key"],
                    "taken_at": p["taken_at"],
                    "place_name": "해변",
                    "trip_name": "제주 여행",
                }
                for p in process["attachments"]
            ],
        },
    }
