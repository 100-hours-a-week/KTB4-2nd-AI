"""워크플로우의 모델 단계와 장소 이름 단계. 실험 코드의 함수를 이름과 순서를 유지해 옮겼다.

- agent.describe, agent.propose(PROPOSE_V2), stages.audit(AUDIT_V2)
- stages.cluster_cells, stages.summary_of
- gpsname.gps_records(4단계), gpsname.same_facility_pairs(5단계)
- naming.name_clusters5(8·10단계, prefix=True, TourAPI 상세 이미지 끔), naming.named_purge(11단계)
- run_meta.vote_cl(9단계), run_meta.one_m의 second(12단계, Qwen 유사도 순위)
실험의 결정 216번 비교 조건(hybrid)과 4단계 선택 사항(주소, 색인 검색, 지시문 제거)은
기본값(꺼짐)의 경로만 옮겼다.
CPU를 쓰는 그림 만들기는 작업 이벤트 루프를 막지 않도록 스레드에서 한다.
"""

import asyncio
import math
import random
import re
import unicodedata
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

from app.infra.place_index import normalize
from app.pipeline.workflow import prompts
from app.pipeline.workflow.client import Places, Pod, parse_json
from app.pipeline.workflow.trip import Trip, jpeg, label_box

# ---------- 값(실험 코드의 상수와 인자 기본값을 그대로 옮김)
GPS_RADIUS_M = 2000  # 4·5단계 GPS 장소 근처 후보 반경(서비스 장소 검색 기본 반경)
GPS_CATS = ["AT4", "CT1", "AD5", "FD6", "CE7"]  # 4단계 카카오 분류 검색
GPS_KEYWORD_SIZE, GPS_CATEGORY_SIZE, GPS_TOUR_K, GPS_MAX_CANDS = 15, 5, 10, 30
REGION_RADIUS_M = 20_000  # 8·10단계 여행 지역 좌표마다 카카오 반경(API 상한)
INDEX_RADIUS_M, INDEX_LIMIT, INDEX_STRONG = 30_000, 5, 4  # 8·10단계 공공 장소 색인
KEYWORD_PER_QUERY, NAME_MAX_CANDS, CHOOSE_ROUNDS = 10, 80, 2
ANCHOR_MIN_KM, ANCHOR_MAX, ANCHOR_PER_QUERY = 3, 4, 5
SHORTLIST_CAP = 10
ALL_TYPES = ["12", "14", "28", "32"]  # TourAPI 관광지·문화시설·레포츠·숙박
CAT_CODE = {"관광명소": "AT4", "문화시설": "CT1", "숙박": "AD5"}
SAME_RECORD_M = 300
VOTE_SEED, SECOND_SEED, SECOND_TOP = 2601, 2401, 8


def chunk(seq, k):
    return [seq[i : i + k] for i in range(0, len(seq), k)]


def mdist(lat, lon, pts):
    R = 6371008.8
    best = None
    for a, b in pts:
        p1, p2 = math.radians(lat), math.radians(a)
        h = (
            math.sin((p2 - p1) / 2) ** 2
            + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b - lon) / 2) ** 2
        )
        d = 2 * R * math.asin(math.sqrt(h))
        best = d if best is None or d < best else best
    return best


def km(a, b):
    R = 6371.0088
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(h))


def _norm(t):
    """실험 named_score.norm. 서비스 place_index.normalize와 달리 밑줄 외 기호만 지운다."""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", t or "").lower())


def same_place_records(a, b, dist_m=SAME_RECORD_M):
    """이름(괄호 안 글자, 띄어쓰기·기호 제거)이 같거나 한쪽이 다른 쪽을 포함하고, 좌표가 dist_m
    안이면 같은 장소.
    """
    na = _norm(re.sub(r"\(.*?\)", "", a["name"]))
    nb = _norm(re.sub(r"\(.*?\)", "", b["name"]))
    if not na or not nb:
        return False
    if not (na == nb or na in nb or nb in na):
        return False
    return km((a["lat"], a["lon"]), (b["lat"], b["lon"])) * 1000 <= dist_m


def same_record(a, b):
    if not a or not b:
        return False
    return a["id"] == b["id"] or same_place_records(a, b)


def parse_card(ch, letters, names):
    ch = str(ch or "").strip()
    if ch.upper()[:1] in letters and (len(ch) == 1 or not ch[1:2].isalpha() or ch[1:2] in " :.)"):
        return letters.index(ch.upper()[:1])
    for k, n in enumerate(names):
        if n and (ch == n or n in ch):
            return k
    return None


def pick(Q, ph, k):
    """select_merge.pick: 대표 사진과, 고른 사진과 가장 덜 닮은 사진을 차례로 최대 k장"""
    if len(ph) == 1:
        return list(ph)
    sub = Q[np.ix_(ph, ph)].copy()
    np.fill_diagonal(sub, 0)
    ch = [ph[int(np.argmax(sub.sum(1)))]]
    while len(ch) < min(k, len(ph)):
        rest = [i for i in ph if i not in ch]
        ch.append(min(rest, key=lambda i: max(Q[i, c] for c in ch)))
    return ch


def zs(M):
    """사진 쌍 유사도의 표준화. 순위에만 쓰므로 사진이 한 장이거나 값이 모두 같으면 그대로 둔다."""
    iu = np.triu_indices(len(M), 1)
    v = M[iu]
    if v.size == 0 or not v.std() > 0:
        return M
    return (M - v.mean()) / v.std()


def cluster_cells(trip: Trip, pl, k=9, label=""):
    """묶음의 사진 k장. 큰 그룹부터 대표 사진을 하나씩, 모자라면 다시 돌며 다른 사진을 더한다."""
    order = sorted(pl, key=lambda g: -len(trip.groups[g]))
    picks, depth = [], 1
    while len(picks) < k and depth <= 4:
        for g in order:
            ds = trip.diverse(g, depth)
            if len(ds) >= depth and ds[depth - 1] not in picks:
                picks.append(ds[depth - 1])
            if len(picks) >= k:
                break
        depth += 1
    return [(i, label if j == 0 else "") for j, i in enumerate(picks)]


def summary_of(desc, pl, n_type=3, n_text=6):
    types, texts = [], []
    for g in pl:
        d = desc[g] if g < len(desc) else None
        if not isinstance(d, dict):
            continue
        if d.get("place_type") and d["place_type"] not in types:
            types.append(d["place_type"])
        for x in d.get("text") or []:
            if isinstance(x, str) and x.strip() and x not in texts:
                texts.append(x.strip()[:30])
    return (
        "장소 종류: "
        + ", ".join(types[:n_type])
        + (" / 글자: " + ", ".join(texts[:n_text]) if texts else "")
    )


def desc_line(k, d):
    if not isinstance(d, dict) or "error" in d:
        return f"[{k}] (설명 없음)"
    text = ", ".join(t for t in (d.get("text") or []) if isinstance(t, str))[:80]
    return (
        f"[{k}] {d.get('subject', '')} | {d.get('place_type', '')} | {d.get('setting', '')} | "
        f"{d.get('distinctive', '')}"
        + (f" | 글자: {text}" if text else "")
        + (f" | 사람: {d.get('people')}" if d.get("people") else "")
    )


def text(t):
    return {"type": "text", "text": t}


def image(url):
    return {"type": "image_url", "image_url": {"url": url}}


async def grid(trip: Trip, cells, cols, cell) -> str:
    return await asyncio.to_thread(lambda: jpeg(trip.montage(cells, cols=cols, cell=cell)))


async def photos(trip: Trip, idx, side=896) -> list[str]:
    return await asyncio.to_thread(lambda: [trip.photo_uri(i, side) for i in idx])


# ---------- 3단계 그룹 설명
async def describe(pod: Pod, trip: Trip, k=2):
    async def one(g):
        chosen = trip.diverse(g, k)
        content = [text(prompts.DESCRIBE.format(k=len(chosen), trip=trip.trip_text()))]
        content += [image(u) for u in await photos(trip, chosen)]
        r = await pod.chat(content, 400, budget=280, stage="describe")
        return (
            r.get("text")
            and parse_json(r["text"])
            or {"error": r.get("error"), "raw": r.get("text")}
        )

    return list(await asyncio.gather(*(one(g) for g in range(trip.G))))


# ---------- 6단계 묶음 제안
async def propose(pod: Pod, trip: Trip, desc):
    lines = "\n".join(desc_line(g + 1, d) for g, d in enumerate(desc))
    content = [text(prompts.PROPOSE_V2.format(n=trip.G, trip=trip.trip_text(), desc=lines))]
    for s in range(0, trip.G, 16):
        cells = [(trip.medoid(g), str(g + 1)) for g in range(s, min(s + 16, trip.G))]
        content.append(image(await grid(trip, cells, 4, 256)))
    r = await pod.chat(content, 4000, budget=1120, stage="propose")
    p = parse_json(r.get("text")) if r.get("text") else None
    places = (p or {}).get("places") or []
    assign, dup, bad = {}, 0, 0
    for c, pl in enumerate(places):
        for num in pl.get("groups") or []:
            if not isinstance(num, int) or not 1 <= num <= trip.G:
                bad += 1
            elif num - 1 in assign:
                dup += 1
            else:
                assign[num - 1] = c
    return {"places": places, "assign": assign, "dup": dup, "bad": bad}


def places_of(prop, G):
    by = {}
    for g, c in prop["assign"].items():
        by.setdefault(c, []).append(g)
    places = list(by.values())
    places += [[g] for g in range(G) if g not in prop["assign"]]
    return places


# ---------- 7단계 묶음 검토
async def audit(pod: Pod, trip: Trip, places, per_group=2):
    async def one(gs):
        if len(gs) < 2:
            return [gs]
        k = per_group if len(gs) * per_group <= 36 else 1
        cells = [(i, str(g + 1)) for g in gs for i in trip.diverse(g, k)][:64]
        cols = 3 if len(cells) <= 36 else 4
        nums = ", ".join(str(g + 1) for g in gs)
        content = [text(prompts.AUDIT_V2.format(trip=trip.trip_text(), nums=nums))]
        for part in chunk(cells, cols * cols):
            content.append(image(await grid(trip, part, cols, 1024 // cols)))
        r = await pod.chat(content, 1500, budget=1120, stage="audit")
        p = parse_json(r.get("text")) if r.get("text") else None
        parts, seen = [], set()
        for part in (p or {}).get("parts") or []:
            sub = [
                n - 1
                for n in part.get("groups") or []
                if isinstance(n, int) and n - 1 in gs and n - 1 not in seen
            ]
            seen |= set(sub)
            if sub:
                parts.append(sub)
        rest = [g for g in gs if g not in seen]
        if rest:  # 응답에서 빠진 그룹은 원래 묶음에 남긴다(첫 부분이 있으면 그쪽, 없으면 그대로)
            if parts:
                parts[0] += rest
            else:
                parts = [gs]
        return parts

    res = await asyncio.gather(*(one(gs) for gs in places))
    return [part for parts in res for part in parts]


# ---------- 4단계 GPS 장소 레코드 확인
def tour_near(raw, pts, k=GPS_TOUR_K):
    out = []
    for x in raw:
        if not x.get("mapy"):
            continue
        lat, lon = float(x["mapy"]), float(x["mapx"])
        if abs(lat - pts[0][0]) > 0.05 or abs(lon - pts[0][1]) > 0.06:
            continue
        d = mdist(lat, lon, pts)
        if d <= GPS_RADIUS_M:
            out.append(
                {
                    "id": f"kb:tourapi:{x['contentid']}",
                    "name": x.get("title", ""),
                    "category": f"TourAPI 종류 {x.get('contenttypeid')}",
                    "address": x.get("addr1", ""),
                    "lat": lat,
                    "lon": lon,
                    "dist": round(d),
                }
            )
    return sorted(out, key=lambda r: r["dist"])[:k]


def _near_rec(p, pts):
    return {
        "id": p.candidate_id,
        "name": p.name,
        "category": p.category,
        "address": p.address,
        "lat": p.latitude,
        "lon": p.longitude,
        "dist": round(mdist(p.latitude, p.longitude, pts)),
    }


async def gps_records(pod: Pod, places: Places, view: Trip, supers, grec, desc):
    """supers: GPS 그룹 번호 목록. 돌려주는 값: {그룹: 레코드 또는 None}"""

    async def one(g):
        pts = [tuple(p) for p in grec[g]["points"]]
        c = (grec[g]["lat"], grec[g]["lon"])
        img = await grid(view, cluster_cells(view, [g], 9), 3, 341)
        summ = summary_of(desc, [g])
        qtext = prompts.QUERY2.format(trip=view.trip_text(), summ=summ)
        r = await pod.chat([text(qtext), image(img)], 300, budget=560, stage="gpsname-query")
        p = parse_json(r.get("text")) if r.get("text") else None
        qs = [q for q in ((p or {}).get("queries") or []) if isinstance(q, str) and q.strip()][:3]
        vis = (p or {}).get("visible_name")
        if isinstance(vis, str) and vis.strip() and vis.strip() not in qs:
            qs = [vis.strip()] + qs
        cands = []
        for q in qs:
            found = await places.keyword(
                q, center=c, radius_m=GPS_RADIUS_M, size=GPS_KEYWORD_SIZE, sort="accuracy"
            )
            cands += [_near_rec(x, pts) for x in found]
        for code in GPS_CATS:
            found, _ = await places.category(
                code, c[0], c[1], GPS_RADIUS_M, size=GPS_CATEGORY_SIZE, sort="distance"
            )
            cands += [_near_rec(x, pts) for x in found]
        cands += await asyncio.to_thread(tour_near, places.tour, pts)
        seen, uniq = set(), []
        for x in cands:
            if x["id"] not in seen:
                seen.add(x["id"])
                uniq.append(x)
        uniq = sorted(uniq, key=lambda x: x["dist"])[:GPS_MAX_CANDS]
        if not uniq:
            return g, None
        lines = "\n".join(
            f"{k + 1}. {x['name']} | {x['category']} | {x['address'] or x.get('addr_note', '')} "
            f"| 사진 좌표에서 {x['dist']} m"
            for k, x in enumerate(uniq)
        )
        ctext = prompts.GPS_CHOOSE.format(trip=view.trip_text(), summ=summ, cands=lines)
        r2 = await pod.chat([text(ctext), image(img)], 300, budget=560, stage="gpsname-choose")
        p2 = parse_json(r2.get("text")) if r2.get("text") else None
        ch = str((p2 or {}).get("choice", "")).strip()
        if ch.isdecimal() and 1 <= int(ch) <= len(uniq):
            return g, uniq[int(ch) - 1]
        return g, None

    res = await asyncio.gather(*(one(g) for g in supers))
    return {g: r for g, r in res}


# ---------- 5단계 같은 시설 판정
async def same_facility_pairs(pod: Pod, view: Trip, pairs, found, sdist):
    async def one(a, b):
        ra, rb = found[a], found[b]
        txt = prompts.FACILITY.format(
            trip=view.trip_text(),
            a=ra["name"],
            ca=ra.get("category", ""),
            aa=ra.get("address", ""),
            b=rb["name"],
            cb=rb.get("category", ""),
            ab=rb.get("address", ""),
            d=round(sdist(a, b)),
        )
        r = await pod.chat([text(txt)], 150, stage="facility")
        p = parse_json(r.get("text")) if r.get("text") else None
        v = (p or {}).get("same_facility")
        return a, b, v is True or (isinstance(v, str) and v.strip().lower() == "true")

    res = await asyncio.gather(*(one(a, b) for a, b in pairs))
    return [(a, b) for a, b, same in res if same]


# ---------- 8·10단계 이름 붙이기
def sigungu_tokens(regions):
    """'강원특별자치도 정선군' -> '정선군', '경기도 안산시 단원구' -> '안산시'(시·군 단위)."""
    return [r.split()[1] for r in regions if len(r.split()) >= 2]


def raw_catalog(raw, centers, types):
    """여행 지역 좌표마다 가장 가까운 원자료 레코드의 법정동 시군구 코드를 얻고, 그 코드의 종류별
    레코드를 모은다.
    """
    codes = []
    for lat, lon in centers:
        best = min(
            (x for x in raw if x.get("mapy") and x.get("lDongSignguCd")),
            key=lambda x: (
                (float(x["mapy"]) - lat) ** 2
                + ((float(x["mapx"]) - lon) * math.cos(math.radians(lat))) ** 2
            ),
        )
        c = (best["lDongRegnCd"], best["lDongSignguCd"])
        if c not in codes:
            codes.append(c)
    return [
        x
        for x in raw
        if (x.get("lDongRegnCd"), x.get("lDongSignguCd")) in codes
        and x.get("contenttypeid") in types
    ]


def _kakao_rec(p):
    return {
        "id": p.candidate_id,
        "name": p.name,
        "category": p.category,
        "address": p.address,
        "lat": p.latitude,
        "lon": p.longitude,
    }


class Searcher:
    """실험 naming.Searcher(regional=True)와 kakao_category. 여행 지역 좌표(centers)와 시·군
    이름으로 거른다.
    """

    def __init__(self, places: Places, centers, regions) -> None:
        self.places, self.centers = places, centers
        self.tokens = sigungu_tokens(regions)
        self._cat: dict = {}
        self._catalog: dict = {}

    def _in_region(self, p) -> bool:
        return any(t in p.address_name + p.road_address_name for t in self.tokens)

    async def kakao(self, q):
        """전국 검색과 여행 지역 좌표마다 반경 20 km 검색을 합치고, 주소에 시·군 이름이 든 것만
        남긴다.
        """
        docs = await self.places.keyword(q, size=15, sort="accuracy")
        for lat, lon in self.centers:
            docs += await self.places.keyword(
                q, center=(lat, lon), radius_m=REGION_RADIUS_M, size=15, sort="accuracy"
            )
        seen = set()
        docs = [d for d in docs if not (d.candidate_id in seen or seen.add(d.candidate_id))]
        return [_kakao_rec(d) for d in docs if self._in_region(d)]

    async def anchor(self, q, a):
        docs = await self.places.keyword(
            q, center=a, radius_m=REGION_RADIUS_M, size=15, sort="distance"
        )
        return [_kakao_rec(d) for d in docs[:ANCHOR_PER_QUERY]]

    def _index(self, q):
        out, seen = [], set()
        for lat, lon in self.centers:
            for c in self.places.index.search(q, lat, lon, INDEX_RADIUS_M, limit=INDEX_LIMIT):
                if c.candidate_id not in seen:
                    seen.add(c.candidate_id)
                    out.append(
                        {
                            "id": c.candidate_id,
                            "name": c.name,
                            "category": c.category,
                            "address": f"지역 중심에서 {c.distance_m} m",
                            "lat": c.latitude,
                            "lon": c.longitude,
                        }
                    )
        return out

    async def index(self, q):
        return await asyncio.to_thread(self._index, q)

    async def category(self, code):
        """여행 지역 좌표마다 반경 20 km 안의 분류 장소를 정확도순 3쪽까지 모으고 시·군 이름으로
        거른다(캐시).
        """
        if code in self._cat:
            return self._cat[code]
        docs = []
        for lat, lon in self.centers:
            for page in (1, 2, 3):
                found, is_end = await self.places.category(
                    code, lat, lon, REGION_RADIUS_M, page=page, size=15, sort="accuracy"
                )
                docs += found
                if is_end:
                    break
        seen, out = set(), []
        for d in docs:
            if d.candidate_id in seen or not self._in_region(d):
                continue
            seen.add(d.candidate_id)
            out.append(_kakao_rec(d))
        self._cat[code] = out
        return out

    async def catalog(self, types):
        if not self.places.tour:
            return []  # 실험은 원자료가 늘 있었다. 비어 있으면 TourAPI 경로를 건너뛴다
        key = tuple(types)
        if key not in self._catalog:
            self._catalog[key] = await asyncio.to_thread(
                raw_catalog, self.places.tour, self.centers, types
            )
        return self._catalog[key]


def strong_index(q, items):
    """색인의 약한 일치(레코드 이름이 검색어보다 짧고 검색어 안에 들어 있음)는 뺀다."""
    nq = normalize(q)
    return [
        x for x in items if not (len(normalize(x["name"])) < len(nq) and normalize(x["name"]) in nq)
    ]


async def tour_pick2(pod: Pod, trip: Trip, srch: Searcher, img_uri, summ, tag, cap=SHORTLIST_CAP):
    """TourAPI 지역 목록 → 짧은 목록(최대 cap개) → 대표 이미지 대조. 상세 이미지는 쓰지
    않는다(no_detail).
    """
    items = await srch.catalog(ALL_TYPES)
    if not items:
        return None
    lines = "\n".join(f"{k + 1}. {x['title']} | {x.get('addr1', '')}" for k, x in enumerate(items))
    stext = prompts.SHORTLIST10_PREFIX.format(trip=trip.trip_text(), summ=summ, lines=lines)
    r = await pod.chat([text(stext), image(img_uri)], 200, budget=1120, stage=f"{tag}-shortlist0")
    p = parse_json(r.get("text")) if r.get("text") else None
    got = []
    for n in (p or {}).get("candidates") or []:
        n = int(n) if isinstance(n, str) and n.strip().isdecimal() else n
        if isinstance(n, int) and 1 <= n <= len(items) and n - 1 not in got:
            got.append(n - 1)
    # 실험은 짧은 목록을 runs번 받아 자주 나온 순으로 합쳤다. runs=1이면 받은 순서 그대로다.
    cnt = Counter(got)
    first = {j: pos for pos, j in reversed(list(enumerate(got)))}
    ranked = sorted(cnt, key=lambda j: (-cnt[j], first[j]))
    cands = [items[j] for j in ranked[:cap]]
    if not cands:
        return None
    flat = [(k, x["firstimage"]) for k, x in enumerate(cands) if x.get("firstimage")]
    ims = await asyncio.gather(*(srch.places.image(u) for _, u in flat))
    cells = [(k, im) for (k, _), im in zip(flat, ims, strict=True) if im is not None]
    if not cells:
        return None

    def canvas_uri():
        cell, cols = 256, 4
        rows = (len(cells) + cols - 1) // cols
        canvas = Image.new("RGB", (cell * cols, cell * rows), (128, 128, 128))
        draw = ImageDraw.Draw(canvas)
        for pos, (k, im) in enumerate(cells):
            im.thumbnail((cell - 4, cell - 4))
            xx, yy = (pos % cols) * cell, (pos // cols) * cell
            canvas.paste(im, (xx + (cell - im.width) // 2, yy + (cell - im.height) // 2))
            label_box(draw, xx, yy, str(k + 1), trip.font)
        return jpeg(canvas)

    canvas = await asyncio.to_thread(canvas_uri)
    shown = sorted({k for k, _ in cells})
    names = ", ".join(f"{k + 1}. {cands[k]['title']}" for k in shown)
    ptext = prompts.IMAGE_PICK2.format(trip=trip.trip_text(), names=names)
    r = await pod.chat(
        [text(ptext), image(img_uri), image(canvas)], 300, budget=1120, stage=f"{tag}-imagepick"
    )
    p = parse_json(r.get("text")) if r.get("text") else None
    ch = str((p or {}).get("choice", "none")).strip()
    if ch.isdecimal() and int(ch) - 1 in shown:
        x = cands[int(ch) - 1]
        return {
            "id": f"kb:tourapi:{x['contentid']}",
            "name": x["title"],
            "category": f"tourapi:{x['contenttypeid']}",
            "address": x.get("addr1", ""),
            "lat": float(x["mapy"]),
            "lon": float(x["mapx"]),
        }
    return None


async def name_clusters5(pod: Pod, trip: Trip, srch: Searcher, places, desc, tag, min_photos=2):
    """사진 min_photos장 이상인 묶음마다 TourAPI 경로 → 키워드 경로 → 기준점 다시 찾기. 같은
    레코드끼리 합친다.
    """
    state = {}

    async def search(queries, anchors=None):
        cands, seen = [], set()
        for q in queries:
            got = (await srch.kakao(q))[:KEYWORD_PER_QUERY]
            for a in anchors or []:
                got += await srch.anchor(q, a)
            for x in got + strong_index(q, await srch.index(q))[:INDEX_STRONG]:
                if x["id"] not in seen:
                    seen.add(x["id"])
                    cands.append(x)
        return cands

    async def choose(img, summ, queries, listed, anchors, anchor_names):
        for rnd in range(CHOOSE_ROUNDS):
            cands = await search(queries, anchors)
            ids = {x["id"] for x in cands}
            allc = (cands + [x for x in listed if x["id"] not in ids])[:NAME_MAX_CANDS]
            if not allc:
                break

            def dist(x):
                if not anchors:
                    return ""
                return (
                    f" | 확정 장소까지 {min(km((x['lat'], x['lon']), a) for a in anchors):.1f} km"
                )

            lines = "\n".join(
                f"{k + 1}. {x['name']} | {x['category']} | "
                f"{x['address'] or x.get('addr_note', '')}{dist(x)}"
                for k, x in enumerate(allc)
            )
            prompt = (
                prompts.CHOOSE4.format(
                    trip=trip.trip_text(), summ=summ, cands=lines, anchors=anchor_names
                )
                if anchors
                else prompts.CHOOSE3.format(trip=trip.trip_text(), summ=summ, cands=lines)
            )
            stage = f"{tag}-choose{'A' if anchors else ''}{rnd}"
            r = await pod.chat([text(prompt), image(img)], 400, budget=1120, stage=stage)
            p2 = parse_json(r.get("text")) if r.get("text") else None
            ch = str((p2 or {}).get("choice", "none")).strip()
            idx = int(ch) - 1 if ch.isdecimal() and 1 <= int(ch) <= len(allc) else None
            if idx is not None:
                return allc[idx]
            queries = [
                q.strip()
                for q in (p2 or {}).get("retry_queries") or []
                if isinstance(q, str) and q.strip()
            ][:3]
            if not queries:
                break
        return None

    async def first(c):
        pl = places[c]
        if sum(len(trip.groups[g]) for g in pl) < min_photos:
            return None
        img = await grid(trip, cluster_cells(trip, pl, 9, ""), 3, 341)
        summ = summary_of(desc, pl)
        qtext = prompts.QUERY2.format(trip=trip.trip_text(), summ=summ)
        r = await pod.chat([text(qtext), image(img)], 300, budget=1120, stage=f"{tag}-query")
        p = parse_json(r.get("text")) if r.get("text") else None
        queries = [
            q.strip() for q in (p or {}).get("queries") or [] if isinstance(q, str) and q.strip()
        ][:3]
        cat = str((p or {}).get("category") or "").strip()
        state[c] = (img, summ, queries, cat)
        rec = await tour_pick2(pod, trip, srch, img, summ, tag)
        if rec:
            return {"choice": rec, "via": "tourapi"}
        listed = await srch.category(CAT_CODE[cat]) if cat in CAT_CODE else []
        rec = await choose(img, summ, queries, listed, None, None)
        return {"choice": rec, "via": "keyword" if rec else None}

    res = list(await asyncio.gather(*(first(c) for c in range(len(places)))))
    anchors, anchor_names = [], []
    for r in res:
        if r and r.get("choice"):
            pt = (r["choice"]["lat"], r["choice"]["lon"])
            if all(km(pt, a) > ANCHOR_MIN_KM for a in anchors) and len(anchors) < ANCHOR_MAX:
                anchors.append(pt)
                anchor_names.append(r["choice"]["name"])
    anchor_names = ", ".join(anchor_names)

    async def second(c):
        r = res[c]
        if r is None or r.get("choice") or c not in state or not anchors:
            return r
        img, summ, queries, cat = state[c]
        listed = await srch.category(CAT_CODE[cat]) if cat in CAT_CODE else []
        rec = await choose(img, summ, queries, listed, anchors, anchor_names)
        if rec:
            r["choice"], r["via"] = rec, "anchor"
        return r

    res = list(await asyncio.gather(*(second(c) for c in range(len(places)))))
    by: dict = {}
    for c, r in enumerate(res):
        key = r["choice"]["id"] if r and r.get("choice") else f"_none{c}"
        by.setdefault(key, []).append(c)
    folders = [
        {
            "record": None if key.startswith("_none") else next(res[c]["choice"] for c in cs),
            "groups": [g for c in cs for g in places[c]],
        }
        for key, cs in by.items()
    ]
    vias = Counter((r or {}).get("via") for r in res)
    return folders, dict(vias)


# ---------- 9단계 다수결 배정
async def vote_cl(pod: Pod, trip: Trip, folders, far, tveto, votes=3, seed=VOTE_SEED):
    """이름 붙은 폴더를 카드 A~Z로 보이고 모든 그룹을 카드 순서를 바꿔 votes번 묻는다.

    far(옮길 그룹, 앵커 그룹들)가 참이거나 tveto(그룹, 앵커 그룹들)가 참이면 다수가 가리켜도
    옮기지 않는다.
    """
    named = [f for f in folders if f["record"]]
    if not named:
        return folders, {"blocked": 0, "time_blocked": 0}
    cur = {g: k for k, f in enumerate(folders) for g in f["groups"]}
    named_idx = {k: j for j, k in enumerate([k for k, f in enumerate(folders) if f["record"]])}
    rounds = []
    for v in range(votes):
        order = list(range(len(named)))
        random.Random(seed + v).shuffle(order)
        letters = [chr(ord("A") + k) for k in range(len(order))]
        imgs = [
            await grid(trip, cluster_cells(trip, named[j]["groups"], 9, lt), 3, 341)
            for j, lt in zip(order, letters, strict=True)
        ]
        names = ", ".join(
            f"{lt}: {named[j]['record']['name']}" for j, lt in zip(order, letters, strict=True)
        )
        head = [text(prompts.REASSIGN2.format(trip=trip.trip_text(), names=names))]
        head += [image(u) for u in imgs]
        rounds.append((order, letters, head))

    async def one(g, v):
        order, letters, head = rounds[v]
        content = head + [image(u) for u in await photos(trip, trip.diverse(g, 2), side=640)]
        r = await pod.chat(content, 200, budget=560, stage=f"vote{v}")
        p = parse_json(r.get("text")) if r.get("text") else None
        k = parse_card(
            (p or {}).get("choice"), letters, [named[j]["record"]["name"] for j in order]
        )
        return None if k is None else order[k]

    G = trip.G
    res = await asyncio.gather(*(one(g, v) for g in range(G) for v in range(votes)))
    target, blocked, tblocked = {}, 0, 0
    for g in range(G):
        vs = [x for x in res[g * votes : (g + 1) * votes] if x is not None]
        c = Counter(vs).most_common(1)
        maj = c[0][0] if c and c[0][1] >= 2 else None
        home = named_idx.get(cur[g])
        if maj is not None and maj != home and far([g], named[maj]["groups"]):
            blocked += 1
            maj = None
        if maj is not None and maj != home and tveto is not None and tveto(g, named[maj]["groups"]):
            tblocked += 1
            maj = None
        if maj is not None and maj != home:
            target[g] = ("anchor", maj)
        elif home is not None:
            target[g] = ("anchor", home)
        else:
            target[g] = ("rest", cur[g])
    new = [{"record": f["record"], "groups": []} for f in named]
    rest: dict = {}
    for g in range(G):
        kind, k = target[g]
        if kind == "anchor":
            new[k]["groups"].append(g)
        else:
            rest.setdefault(k, []).append(g)
    new = [f for f in new if f["groups"]] + [{"record": None, "groups": gs} for gs in rest.values()]
    return new, {"blocked": blocked, "time_blocked": tblocked}


# ---------- 11단계 정리
async def named_purge(pod: Pod, trip: Trip, folders):
    """폴더마다 그 장소가 아닌 것이 분명한 그룹 목록을 돌려준다(폴더 순서와 같음)."""

    async def one(f):
        if not f["record"] or len(f["groups"]) < 2:
            return []
        gs = f["groups"]
        per = 2 if len(gs) * 2 <= 36 else 1
        cells = [(i, str(g + 1)) for g in gs for i in trip.diverse(g, per)][:64]
        cols = 3 if len(cells) <= 36 else 4
        r = f["record"]
        content = [
            text(
                prompts.PURGE.format(
                    name=r["name"],
                    category=r.get("category", ""),
                    address=r.get("address", ""),
                    trip=trip.trip_text(),
                    nums=", ".join(str(g + 1) for g in gs),
                )
            )
        ]
        for part in chunk(cells, cols * cols):
            content.append(image(await grid(trip, part, cols, 1024 // cols)))
        res = await pod.chat(content, 600, budget=1120, stage="purge")
        p = parse_json(res.get("text")) if res.get("text") else None
        out_ = []
        for n in (p or {}).get("not_here") or []:
            n = int(n) if isinstance(n, str) and n.strip().isdecimal() else n
            if isinstance(n, int) and n - 1 in gs and n - 1 not in out_:
                out_.append(n - 1)
        return out_

    return list(await asyncio.gather(*(one(f) for f in folders)))


# ---------- 12단계 미분류 재배정
async def second_select(pod: Pod, trip: Trip, folders):
    """레코드 없는 폴더의 그룹마다 Qwen 유사도 상위 8개 이름 폴더 가운데 하나 또는 none을 고르게
    한다.

    돌려주는 값: [(그룹, 고른 폴더 번호 또는 None)]
    """
    srank = zs(trip.S)
    named_k = [k for k, f in enumerate(folders) if f["record"]]
    orphans = [g for f in folders if not f["record"] for g in f["groups"]]
    fph = {k: [i for g in folders[k]["groups"] for i in trip.groups[g]] for k in named_k}

    async def ask(g):
        gp = trip.groups[g]
        cand = sorted(named_k, key=lambda k: -srank[np.ix_(gp, fph[k])].max())[:SECOND_TOP]
        if not cand:
            return g, None
        order = cand[:]
        random.Random(SECOND_SEED * 1000 + g).shuffle(order)
        letters = [chr(ord("A") + k) for k in range(len(order))]
        qx = pick(trip.S, gp, 4)
        c = [text(prompts.SELECT.format(trip=trip.trip_text(), last=letters[-1]))]
        c.append(
            image(await grid(trip, [(i, "X" if j == 0 else "") for j, i in enumerate(qx)], 2, 512))
        )
        for L, k in zip(letters, order, strict=True):
            cx = pick(trip.S, fph[k], 2)
            cells = [(i, L if j == 0 else "") for j, i in enumerate(cx)]
            c.append(image(await grid(trip, cells, 2, 384)))
        r = await pod.chat(c, 200, budget=560, stage="second")
        p_ = parse_json(r.get("text")) if r.get("text") else None
        ch = str((p_ or {}).get("choice", "")).strip().upper()[:1]
        return g, (order[letters.index(ch)] if ch in letters else None)

    return list(await asyncio.gather(*(ask(g) for g in orphans)))
