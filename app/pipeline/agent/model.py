"""장소 추정(사진 한 장), 종합, 폴더 묶기의 모델 대화와 도구. 작업의 이벤트 루프에서 돈다.

평가 도구(2026-10-28 agent.py)를 서비스 클라이언트로 옮겼다. 도구는 사진 읽기, 이 여행 사진으로
범위를 제한한 Qdrant 닮은 사진 찾기, 공공 장소 색인 검색, 카카오 키워드 검색이다(결정 130).
사진은 호출마다 분석용 사본의 URL을 새로 받아 붙인다(presigned URL 만료 대비).
"""

import asyncio
import json
import sqlite3
import threading
from collections import Counter, defaultdict
from typing import Any

from app.infra.kakao_local import KakaoBudget, KakaoUnavailable
from app.infra.place_index import normalize
from app.infra.vlm import ChatMessage, ChatTurn, ImagePart, TextPart, VLMTransientError
from app.observability.langfuse_tracing import call_name
from app.pipeline.agent import prompts
from app.pipeline.agent.resources import AgentResources
from app.pipeline.agent.trip import TripView, photo_ref, region_key
from app.pipeline.state import PhotoState

CONFIDENCES = ("high", "medium", "low")
# n번째 재시도 전에 RETRY_WAIT_SEC * n초 기다린다(평가 도구와 같은 5, 10, 15초).
RETRY_WAIT_SEC = 5.0


class Session:
    """작업 하나의 에이전트 상태: 카카오 상한, 동시 호출 수, 여행 지역, 호출 집계"""

    def __init__(
        self, resources: AgentResources, trip: TripView, qdrant, same_photo_min_sim: float
    ) -> None:
        self.resources, self.trip, self.qdrant = resources, trip, qdrant
        self.same_photo_min_sim = same_photo_min_sim
        self.budget = KakaoBudget(resources.kakao_calls)
        self.semaphore = asyncio.Semaphore(resources.concurrency)
        self.region_keys: set[tuple[str, str]] = set()
        self._regions: dict[str, tuple[str, str] | None] = {}
        # 동기 Qdrant 클라이언트를 여러 스레드에서 동시에 부르지 않는다.
        self._qdrant_lock = threading.Lock()
        self.stats: Counter = Counter()
        # 같은 작업 안에서 같은 카카오 키워드 검색은 다시 부르지 않는다(작업이 끝나면 버림).
        self.kakao_results: dict[tuple, list] = {}

    async def chat(
        self, messages: list[ChatMessage], keys: list[str], tools: list[dict], max_tokens: int
    ) -> ChatTurn:
        """전송 오류, HTTP 408, 429, 5xx와 호출 시간 초과만 간격을 늘리며 다시 보낸다(결정 159).

        평가 도구는 httpx 전송 오류(시간 초과 포함)와 408, 429, 5xx를 다시 보냈다. 여기서는 평가
        도구에 없던 호출 전체 제한(PROCESS_VLM_TIMEOUT_SEC)의 시간 초과도 다시 보낸다. 불완전
        응답과 슬롯 대기 초과는 다시 보내지 않는다. 시도마다 오류가 다르면 마지막 오류를 올린다.
        기다리는 동안 GPU 슬롯은 놓지만 작업 안의 동시 호출 자리는 잡고 있다.
        실패한 시도마다 ai_model_call이 남는다.
        """
        retries = max(0, self.resources.vlm_retries)
        async with self.semaphore:
            for attempt in range(retries + 1):
                if attempt:
                    await asyncio.sleep(RETRY_WAIT_SEC * attempt)
                    self.stats["model_retries"] += 1
                # 시도마다 사진 URL을 새로 받는다(presigned URL 만료 대비).
                urls = await asyncio.gather(*(self.resources.urls.resolve(key) for key in keys))
                self.stats["model_calls"] += 1
                try:
                    with call_name("agent"):
                        return await self.resources.vlm.chat(
                            messages, list(urls), max_tokens, tools=tools, temperature=0
                        )
                except (VLMTransientError, TimeoutError):
                    if attempt == retries:
                        raise

    async def load_regions(self, regions) -> None:
        """요청 지역 좌표를 카카오 좌표-행정구역 변환으로 시군구 이름으로 바꾼다(결정 150)."""
        labels = []
        for region in regions:
            found = await self._region(region.latitude, region.longitude)
            if found is None:
                labels.append(f"좌표 {region.latitude:.4f}, {region.longitude:.4f}")
                continue
            self.region_keys.add(region_key(*found))
            labels.append(" ".join(part for part in found if part))
        self.trip.info["regions"] = labels

    async def in_region(self, candidate: dict) -> bool:
        """새 장소 후보 좌표의 시군구가 여행 지역 가운데 하나인지. 확인하지 못하면 False"""
        key = candidate["candidate_id"]
        if key not in self._regions:
            self._regions[key] = await self._region(candidate["lat"], candidate["lon"])
        found = self._regions[key]
        return found is not None and region_key(*found) in self.region_keys

    async def _region(self, latitude: float, longitude: float) -> tuple[str, str] | None:
        try:
            region = await self.resources.kakao.region(latitude, longitude, self.budget)
        except (KakaoUnavailable, TimeoutError):
            # 호출 결과는 카카오 클라이언트가 ai_model_call로 남긴다.
            return None
        return (region.sido, region.sigungu) if region else None

    def similar(self, photo: PhotoState, k: int) -> list[tuple[PhotoState, float]]:
        ids = [p.source.trip_attachment_id for p in self.trip.photos]
        with self._qdrant_lock:
            hits = self.qdrant.search(photo.embedding, ids, k + 1)
        by_id = {p.source.trip_attachment_id: p for p in self.trip.photos}
        return [(by_id[i], s) for i, s in hits if i != photo.source.trip_attachment_id][:k]


def _text(text: str) -> TextPart:
    return {"type": "text", "text": text}


def _image() -> ImagePart:
    return {"type": "image"}


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _ref(value: Any) -> str:
    text = str(value or "").strip()
    return f"S{text}" if text.isdigit() else text


def _args(raw: str) -> dict:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class Tools:
    """한 대화 안에서 쓰는 도구와 사진 수 장부. images_used는 이 대화에 붙인 사진 수"""

    def __init__(self, session: Session, images_used: int, candidates: dict[str, dict]) -> None:
        self.session, self.trip = session, session.trip
        self.images_used, self.candidates = images_used, candidates

    async def run(self, name: str, args: dict) -> tuple[str, ChatMessage | None, list[str]]:
        """도구 응답 문자열, 사진을 붙인 user 메시지, 그 사진들의 사본 키"""
        self.session.stats[f"tool_{name}"] += 1
        if name == "read_photo":
            refs = args.get("photo_ids")
            return self._read([_ref(r) for r in refs] if isinstance(refs, list) else [])
        if name == "read_place_photos":
            place = self.trip.canon(str(args.get("place_id") or "").strip())
            members = self.trip.places().get(place, [])
            if not members:
                return "없는 장소입니다.", None, []
            n = _int(args.get("n"), 2, 1, prompts.READ_LIMIT)
            step = max(1, len(members) // n)
            return self._read([photo_ref(p) for p in members[::step][:n]])
        if name == "find_similar":
            photo = self.trip.by_ref.get(_ref(args.get("photo_id")))
            if photo is None:
                return "해당 ID의 사진이 없습니다.", None, []
            k = _int(args.get("k"), 6, 1, 10)
            hits = await asyncio.to_thread(self.session.similar, photo, k)
            rows = [
                {
                    "photo_id": photo_ref(other),
                    "place_id": self.trip.place_of(other) or "미분류",
                    "similarity": round(score, 3),
                    "taken_at": self.trip.time_text(other),
                }
                for other, score in hits
            ]
            return _dumps(rows), None, []
        if name in ("search_public_places", "kakao_keyword_search"):
            return await self._search(name, args), None, []
        return f"없는 도구입니다: {name}", None, []

    def _read(self, refs: list[str]) -> tuple[str, ChatMessage | None, list[str]]:
        photos = [self.trip.by_ref[r] for r in dict.fromkeys(refs) if r in self.trip.by_ref]
        photos = photos[: prompts.READ_LIMIT]
        if not photos:
            return "해당 ID의 사진이 없습니다.", None, []
        room = prompts.MAX_IMAGES - self.images_used
        if room <= 0:
            return (
                f"한 대화에서 볼 수 있는 사진({prompts.MAX_IMAGES}장)을 모두 썼습니다. "
                "지금까지의 근거로 답을 정하세요.",
                None,
                [],
            )
        photos = photos[:room]
        self.images_used += len(photos)
        content: list[TextPart | ImagePart] = []
        for photo in photos:
            place = self.trip.place_of(photo) or "미분류"
            label = f"{photo_ref(photo)} (장소 {place}, 촬영 시각 {self.trip.time_text(photo)})"
            content += [_text(label), _image()]
        message: ChatMessage = {"role": "user", "content": content}
        keys = [photo.source.analyze_storage_key for photo in photos]
        names = ", ".join(photo_ref(photo) for photo in photos)
        return f"요청한 사진을 다음 메시지에 첨부했습니다: {names}", message, keys

    async def _search(self, name: str, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "검색어가 비어 있습니다."
        latitude, longitude = self.trip.anchor(args.get("near"))
        if name == "search_public_places":
            radius = _int(args.get("radius_m"), 2000, 100, 30000)
            try:
                found = await asyncio.to_thread(
                    self.session.resources.places.search, query, latitude, longitude, radius
                )
            except (ValueError, sqlite3.Error):
                return "공공 장소 자료를 지금 쓸 수 없습니다. 다른 근거를 쓰세요."
            rows = [
                {
                    "candidate_id": c.candidate_id,
                    "name": c.name,
                    "category": c.category,
                    "lat": c.latitude,
                    "lon": c.longitude,
                    "distance_m": c.distance_m,
                }
                for c in found
            ]
        else:
            radius = _int(args.get("radius_m"), 2000, 100, 20000)
            key = (query, round(latitude, 4), round(longitude, 4), radius)
            found = self.session.kakao_results.get(key)
            if found is None:
                try:
                    found = await self.session.resources.kakao.keyword(
                        query, latitude, longitude, radius, self.session.budget
                    )
                except (KakaoUnavailable, TimeoutError):
                    return "카카오 검색을 지금 쓸 수 없습니다. 다른 근거를 쓰세요."
                if found is None:
                    return "카카오 검색 한도에 닿았습니다. 다른 근거를 쓰세요."
                self.session.kakao_results[key] = found
            rows = [
                {
                    "candidate_id": c.candidate_id,
                    "name": c.name,
                    "category": c.category,
                    "lat": c.latitude,
                    "lon": c.longitude,
                    "distance_m": c.distance_m,
                    "address": c.address,
                }
                for c in found
            ]
        for row in rows:
            self.candidates[row["candidate_id"]] = row
        return _dumps({"center": [round(latitude, 5), round(longitude, 5)], "results": rows})


async def episode(
    session: Session,
    system: str,
    content: list[TextPart | ImagePart],
    keys: list[str],
    tools: list[dict],
    final_name: str,
    candidates: dict[str, dict],
    *,
    max_calls: int = prompts.MAX_CALLS,
    max_tokens: int = prompts.MAX_TOKENS,
) -> dict | None:
    """모델이 최종 도구(final_name)를 부를 때까지 도구를 돌린다. 끝내 부르지 않으면 None"""
    messages: list[ChatMessage] = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    keys = list(keys)
    runner = Tools(session, len(keys), candidates)
    for turn in range(max_calls):
        reply = await session.chat(messages, keys, tools, max_tokens)
        assistant: ChatMessage = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            assistant["tool_calls"] = reply.tool_calls
        messages.append(assistant)
        if not reply.tool_calls:
            nudge = f"도구를 써서 답하세요. 최종 답은 {final_name} 도구로 냅니다."
            messages.append({"role": "user", "content": nudge})
            continue
        extra: list[ChatMessage] = []
        for call in reply.tool_calls:
            args = _args(call["arguments"])
            if call["name"] == final_name:
                session.stats[f"tool_{final_name}"] += 1
                return args
            output, image_message, image_keys = await runner.run(call["name"], args)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
            if image_message is not None:
                extra.append(image_message)
                keys.extend(image_keys)
        messages.extend(extra)
        if turn == max_calls - 2:
            last = f"응답 기회가 한 번 남았습니다. 지금 {final_name} 도구로 최종 답을 내세요."
            messages.append({"role": "user", "content": last})
    session.stats["no_final"] += 1
    return None


# ---------- 장소 추정(사진 한 장)
async def estimate(session: Session, photo: PhotoState) -> dict:
    """기본 근거(결정 126, 130, 148)를 주고 decide 답을 검증해 제안으로 바꾼다."""
    trip = session.trip
    ref = photo_ref(photo)
    content: list[TextPart | ImagePart] = [
        _text(
            "여행 정보: "
            + _dumps(trip.info)
            + "\n후보 장소 정보 목록:\n"
            + "\n".join(_dumps(row) for row in trip.summary())
        ),
        _text(f"\n대상 사진 {ref}, 촬영 시각: {trip.time_text(photo)}"),
        _image(),
    ]
    keys = [photo.source.analyze_storage_key]
    before, after = trip.neighbors(photo)
    for label, other in (("바로 앞", before), ("바로 뒤", after)):
        if other is None:
            continue
        gap = round(abs((other.taken_at - photo.taken_at).total_seconds()) / 60)
        place = trip.place_of(other) or "미분류"
        content += [
            _text(
                f"\n촬영 시각이 {label}인 사진 {photo_ref(other)}: 장소 {place}, "
                f"촬영 시각 {trip.time_text(other)}({gap}분 차이)"
            ),
            _image(),
        ]
        keys.append(other.source.analyze_storage_key)
    content.append(_text("\n장소별로 대상 사진과 임베딩 유사도가 가장 높은 사진:"))
    for place, similarity, other in trip.best_per_place(photo):
        if len(keys) >= prompts.MAX_IMAGES:
            break
        content += [
            _text(
                f"\n장소 {place}의 사진 {photo_ref(other)}(유사도 {similarity:.3f}, "
                f"촬영 시각 {trip.time_text(other)})"
            ),
            _image(),
        ]
        keys.append(other.source.analyze_storage_key)
    candidates: dict[str, dict] = {}
    final = await episode(
        session,
        prompts.ESTIMATE_SYSTEM,
        content,
        keys,
        prompts.ESTIMATE_TOOLS,
        "decide",
        candidates,
    )
    proposal = {"photo_id": ref, "decision": None, "confidence": "low", "reason": "최종 답 없음"}
    if final is None:
        return proposal
    confidence = final.get("confidence")
    proposal.update(
        decision=final.get("decision"),
        confidence=confidence if confidence in CONFIDENCES else "low",
        reason=str(final.get("reason") or ""),
    )
    if proposal["decision"] == "existing":
        place = trip.canon(str(final.get("place_id") or "").strip())
        proposal["place_id"] = place
        if place not in trip.places():
            proposal.update(confidence="low", reason="없는 장소 ID: " + proposal["reason"])
    elif proposal["decision"] == "new":
        candidate = candidates.get(str(final.get("candidate_id") or "").strip())
        if candidate is None:
            reason = "후보 ID가 검색 결과에 없음: " + proposal["reason"]
            proposal.update(confidence="low", reason=reason)
        else:
            proposal.update(
                candidate_id=candidate["candidate_id"],
                name=str(final.get("new_place_name") or candidate["name"]),
                lat=candidate["lat"],
                lon=candidate["lon"],
            )
            # high인 새 장소만 지역을 확인해 카카오 호출을 줄인다(결정 150).
            if proposal["confidence"] == "high" and not await session.in_region(candidate):
                proposal.update(confidence="low", reason="여행 지역 밖 후보: " + proposal["reason"])
    else:
        proposal.update(decision=None, confidence="low")
    return proposal


# ---------- 종합
async def aggregate(session: Session, proposals: list[dict]) -> list[dict]:
    """high 제안을 다시 확인해 받아들일 제안만 돌려준다(결정 121, 127)."""
    trip = session.trip
    high = [p for p in proposals if p["confidence"] == "high" and p["decision"]]
    if not high:
        return []
    # 연속 촬영 충돌과 새 장소 묶음은 코드로 미리 찾아 알려 준다(평가 도구와 같음).
    conflicts = []
    for i, a in enumerate(high):
        va = trip.by_ref[a["photo_id"]].embedding
        for b in high[i + 1 :]:
            if float(va @ trip.by_ref[b["photo_id"]].embedding) < session.same_photo_min_sim:
                continue
            if (a.get("place_id") or a.get("candidate_id")) != (
                b.get("place_id") or b.get("candidate_id")
            ):
                conflicts.append([a["photo_id"], b["photo_id"]])
    groups: dict[str, list[str]] = defaultdict(list)
    for p in high:
        if p["decision"] == "new":
            groups[p.get("candidate_id") or normalize(p["name"])].append(p["photo_id"])
    shown = [{k: v for k, v in p.items() if k not in ("lat", "lon")} for p in high]
    text = (
        "후보 장소 정보 목록:\n"
        + "\n".join(_dumps(row) for row in trip.summary())
        + "\n\n확신도 high 제안:\n"
        + "\n".join(_dumps(row) for row in shown)
        + "\n\n거의 같은 사진인데 다른 곳으로 제안된 쌍: "
        + _dumps(conflicts)
        + "\n같은 새 장소를 가리키는 제안 묶음: "
        + _dumps(dict(groups))
    )
    final = await episode(
        session,
        prompts.AGGREGATE_SYSTEM,
        [_text(text)],
        [],
        prompts.AGGREGATE_TOOLS,
        "finalize",
        {},
    )
    accept = (final or {}).get("accept")
    accepted = {_ref(r) for r in accept} if isinstance(accept, list) else set()
    return [p for p in high if p["photo_id"] in accepted]


# ---------- 폴더 묶기
async def merge(session: Session) -> list[list[str]]:
    """같은 장소인 폴더 묶음. high이고 지금 있는 폴더가 둘 이상인 묶음만 돌려준다(결정 132, 134)."""
    trip = session.trip
    text = (
        "여행 정보: "
        + _dumps(trip.info)
        + "\n폴더 목록:\n"
        + "\n".join(_dumps(row) for row in trip.summary())
        + "\n\n폴더 사이의 거리와 시간 관계(중심 거리 m, 촬영 시각 범위 사이 간격 분, 겹치면 0):\n"
        + "\n".join(_dumps(row) for row in trip.relations())
    )
    final = await episode(
        session,
        prompts.MERGE_SYSTEM,
        [_text(text)],
        [],
        prompts.MERGE_TOOLS,
        "merge",
        {},
        max_calls=prompts.MAX_MERGE_CALLS,
        max_tokens=prompts.MAX_MERGE_TOKENS,
    )
    raw = (final or {}).get("groups")
    out = []
    places = trip.places()
    for group in raw if isinstance(raw, list) else []:
        if not isinstance(group, dict) or group.get("confidence") != "high":
            continue
        ids = group.get("place_ids")
        if not isinstance(ids, list):
            continue
        canonical = list(dict.fromkeys(trip.canon(str(x).strip()) for x in ids))
        canonical = [x for x in canonical if x in places]
        if len(canonical) >= 2:
            out.append(canonical)
    return out
