"""워크플로우 정리 파이프라인의 진입점과 단계 순서. 실험 run_meta.one_m의 다음 인자 조합만 옮겼다.

    one_m(key, s, rec_mode=True, facility=True, tveto_min=30, second=True, rank_qwen=True,
          no_detail=True)  # big=5, cl_m=None, nr_m=None, gml=False

순서: 1단계 그룹 → 2단계 메타데이터 잇기 → 3 그룹 설명 → 4 GPS 장소 레코드 → 5 같은 시설 → 6 묶음
제안
→ 7 묶음 검토 → 8 큰 묶음 이름 → 9 다수결 배정 → 10 남은 묶음 이름 → 11 정리 → 12 미분류 재배정 →
13 최종 판정.
13단계 뒤 결과를 사진 상태(place_id, 좌표, 출처, 미분류 사유)로 옮기는 부분(apply_folders)은
실험에 없고
서비스 결과 계약에 맞추려고 더했다.
"""

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, wait
from datetime import timedelta
from typing import Any

from app.core.errors import PipelineCancelled, PipelineError
from app.pipeline.agent.trip import region_key
from app.pipeline.state import PhotoState
from app.pipeline.workflow import steps
from app.pipeline.workflow.client import Places, Pod
from app.pipeline.workflow.resources import WorkflowResources, WorkflowTimeout
from app.pipeline.workflow.trip import Trip, View, build_views, gps_record, haversine, load_font
from app.schemas.process import KST, Issue, ProcessRequest, RegionOrigin

BIG = 5  # 8단계 큰 묶음 기준 사진 수
MAX_CARDS = 26  # 9단계 카드 A~Z
FACILITY_M = 2000  # 5단계 같은 시설을 묻는 GPS 사진 사이 최대 거리
ATTACH_M = 100.0  # 검색 레코드를 GPS 묶음에 흡수하는 거리
POLL_SEC = 1.0
CANCEL_CHECK_SEC = 5.0
REQUIRED = (
    "workflow_group_min_sim",
    "workflow_gps_link_m",
    "workflow_insert_gap_min",
    "workflow_clock_pair_min_sim",
    "workflow_time_veto_min",
)


def size(view, pl):
    return sum(len(view.groups[g]) for g in pl)


async def organize(
    trip: Trip, pod: Pod, places: Places, centers, th, log: dict, region_keys=None
) -> tuple:
    """실험 one_m의 본문. 돌려주는 값: (full_view, 폴더 목록, SUPER 그룹 집합)

    region_keys(요청 지역의 (시도, 시·군) 집합)가 있으면 실험에 없던 지역 밖 규칙을 더한다
    (결정 221). GPS 장소는 대표 좌표의 시·군이 지역 밖이면 2단계 뒤 빼고, GPS 없는 묶음은
    8·10단계에서 고른 레코드 좌표의 시·군이 지역 밖이면 이름을 붙이지 않은 것으로 다룬다.
    시·군을 알아내지 못하면 남긴다. None이면 실험과 같다(정합성 비교에 쓴다).
    """
    secs: dict[str, float] = {}
    t0 = time.monotonic()
    s, fv, gfold = await asyncio.to_thread(
        build_views,
        trip,
        eps=float(th["workflow_gps_link_m"]),
        gap_min=float(th["workflow_insert_gap_min"]),
        pair_min_sim=float(th["workflow_clock_pair_min_sim"]),
    )
    tveto_min = float(th["workflow_time_veto_min"])
    secs["stage2"] = time.monotonic() - t0
    R, K = len(s["rest"]), len(gfold)
    if region_keys:
        # 지역 밖 GPS 장소: 그 장소의 사진(끼워 넣은 GPS 없는 사진 포함)을 어느 그룹에도 두지 않아
        # 위치 미분류로 보낸다. 남은 GPS 장소의 번호와 자리표시 이름은 0부터 다시 매긴다.
        keep = []
        for k in range(K):
            r = gfold[k]["record"]
            if await _in_region(places, r["lat"], r["lon"], region_keys):
                keep.append(k)
        log["region_excluded_gps_places"] = K - len(keep)
        if len(keep) < K:
            fv = View(trip, [fv.groups[g] for g in range(R)] + [fv.groups[R + k] for k in keep])
            K = len(keep)
            gfold = [
                {"record": gps_record(fv, R + j, s["rows"], j), "groups": [R + j]} for j in range(K)
            ]
    SUPER = set(range(R, R + K))
    grec = {R + k: gfold[k]["record"] for k in range(K)}
    log["groups"] = trip.G
    log["stage2"] = {
        "places": K,
        "rest_groups": R,
        "inserted_groups": len(s["inserted"]),
        "clock_days": {str(k): v for k, v in s["offs"].items()},
    }

    def gps_of(groups):
        sup = [g for g in groups if g in SUPER]
        return grec[max(sup, key=lambda g: (len(fv.groups[g]), -g))] if sup else None

    M = await asyncio.to_thread(fv.group_sim)
    _dm: dict = {}

    def sdist(a, b):
        if (a, b) not in _dm:
            _dm[(a, b)] = _dm[(b, a)] = min(
                float(haversine(x[0], x[1], y[0], y[1]))
                for x in grec[a]["points"]
                for y in grec[b]["points"]
            )
        return _dm[(a, b)]

    cls: dict = {}  # GPS 그룹 -> 같은 레코드 묶음 번호(레코드 없으면 없음)

    def link(a, b):
        """두 GPS 그룹이 한 폴더에 들 수 있는가(rec_mode, nr_m 없음): 같은 레코드 묶음일 때만"""
        ca, cb = cls.get(a), cls.get(b)
        if ca is not None and cb is not None:
            return ca == cb
        return False

    def far(ga, gb):
        """두 그룹 목록을 합칠 수 없는가. 어느 한쪽에 GPS가 없으면 False."""
        sa, sb = [g for g in ga if g in SUPER], [g for g in gb if g in SUPER]
        if not (sa and sb):
            return False
        return not any(link(a, b) for a in sa for b in sb)

    T = s["T"]

    def tveto(g, anchor_groups):
        """그룹의 보정 시각이 하나도 앵커 방문 구간 ± tveto_min분 안에 들지 않으면 옮기지 않는다.

        앵커에 GPS 그룹이 있으면 방문 구간을 GPS 그룹 사진의 시각으로만 만든다. 시각이 없으면
        막지 않는다.
        """
        tg = [T[i] for i in fv.groups[g] if T.get(i) is not None]
        base = [h for h in anchor_groups if h in SUPER] or list(anchor_groups)
        ta = sorted(T[i] for h in base for i in fv.groups[h] if T.get(i) is not None)
        if not tg or not ta:
            return False
        gap = timedelta(minutes=tveto_min)
        spans, st, en = [], ta[0], ta[0]
        for t_ in ta[1:]:
            if t_ - en <= gap:
                en = t_
            else:
                spans.append((st, en))
                st = en = t_
        spans.append((st, en))
        return not any(a - gap <= t_ <= b + gap for t_ in tg for a, b in spans)

    def time_split(places_, tag):
        if not tveto_min:  # 실험처럼 0이면 시각으로 떼지 않는다
            return places_
        new, moved = [], []
        for pl in places_:
            sup = [g for g in pl if g in SUPER]
            if not sup:
                new.append(pl)
                continue
            keep = []
            for g in pl:
                if g not in SUPER and tveto(g, sup):
                    new.append([g])
                    moved.append(g)
                else:
                    keep.append(g)
            new.append(keep)
        log.setdefault("time_split", {})[tag] = len(moved)
        return new

    def time_split_f(folders, tag):
        if not tveto_min:
            return folders
        new = []
        for f in folders:
            parts = time_split([f["groups"]], tag)
            new.append({"record": f["record"], "groups": parts[-1]})
            new += [{"record": None, "groups": pt} for pt in parts[:-1]]
        return [f for f in new if f["groups"]]

    def must_link(folders, tag):
        """같은 레코드로 확인된 GPS 그룹이 다른 폴더에 있으면 그 폴더들을 합친다(앞 폴더의 레코드를
        남김).
        """
        out_, where, merged = [], {}, 0
        for f in folders:
            c = next((cls[g] for g in f["groups"] if g in SUPER and cls.get(g) is not None), None)
            if c is not None and c in where:
                out_[where[c]]["groups"] = out_[where[c]]["groups"] + f["groups"]
                merged += 1
            else:
                if c is not None:
                    where[c] = len(out_)
                out_.append({"record": f["record"], "groups": list(f["groups"])})
        log.setdefault("must_link", {})[tag] = merged
        return out_

    def split_cl(places_, tag):
        """한 묶음 안의 GPS 그룹을 link로 단일 연결해 나누고, GPS 없는 그룹은 임베딩 유사도가 가장
        높은 쪽에 둔다.
        """
        new, splits = [], []
        for pl in places_:
            sup = [g for g in pl if g in SUPER]
            if len(sup) < 2:
                new.append(pl)
                continue
            par = {g: g for g in sup}

            def f(x, par=par):
                while par[x] != x:
                    x = par[x]
                return x

            for a in sup:
                for b in sup:
                    if a < b and link(a, b):
                        par[f(a)] = f(b)
            comps: dict = {}
            for g in sup:
                comps.setdefault(f(g), []).append(g)
            parts = list(comps.values())
            if len(parts) == 1:
                new.append(pl)
                continue
            for g in pl:
                if g not in SUPER:
                    max(parts, key=lambda part: max(M[g, x] for x in part)).append(g)
            splits.append([len(p_) for p_ in parts])
            new += parts
        log[tag] = len(splits)
        return new

    def fix(folders):
        """GPS 그룹이 든 폴더의 레코드는 가장 큰 GPS 그룹의 레코드(BE 이름)로 둔다."""
        return [
            {
                "record": gps_of(f["groups"])
                or (f["record"] if not (f["record"] or {}).get("gps") else None),
                "groups": f["groups"],
            }
            for f in folders
        ]

    def attach_near_gps(folders, tag):
        """검색 레코드 좌표가 GPS 묶음의 GPS 사진과 100 m 안이면(또는 같은 레코드면) 가장 가까운
        묶음에 합친다.
        """
        gps_idx = [k for k, f in enumerate(folders) if gps_of(f["groups"])]
        moved, keep = 0, []
        for k, f in enumerate(folders):
            r = f["record"]
            if not r or r.get("gps") or r.get("lat") is None or not gps_idx:
                keep.append(k)
                continue
            best = None
            for j in gps_idx:
                pts = [p for g in folders[j]["groups"] if g in SUPER for p in grec[g]["points"]]
                d = min(float(haversine(r["lat"], r["lon"], a, b)) for a, b in pts)
                if any(
                    steps.same_record(r, found.get(g)) for g in folders[j]["groups"] if g in SUPER
                ):
                    d = 0.0
                if d <= ATTACH_M and (best is None or d < best[0]):
                    best = (d, j)
            if best:
                folders[best[1]]["groups"] = folders[best[1]]["groups"] + f["groups"]
                moved += 1
            else:
                keep.append(k)
        log[tag] = moved
        return [folders[k] for k in keep]

    def merge_ids(folders):
        by: dict = {}
        for f in folders:
            k2 = f["record"]["id"] if f["record"] else f"_n{id(f)}"
            by.setdefault(k2, {"record": f["record"], "groups": []})["groups"] += f["groups"]
        return list(by.values())

    region_cache: dict = {}

    async def drop_out_of_region(found_folders, tag):
        """이름 단계에서 고른 레코드의 시·군이 요청 지역 밖이면 이름을 붙이지 않은 것으로 다룬다."""
        if not region_keys:
            return found_folders
        dropped = 0
        for f in found_folders:
            r = f["record"]
            if not r:
                continue
            if r["id"] not in region_cache:
                region_cache[r["id"]] = await _in_region(places, r["lat"], r["lon"], region_keys)
            if not region_cache[r["id"]]:
                f["record"] = None
                dropped += 1
        log.setdefault("region_dropped_records", {})[tag] = dropped
        return found_folders

    async def timed(name, coro):
        t = time.monotonic()
        value = await coro
        secs[name] = time.monotonic() - t
        return value

    # 3단계 그룹 설명
    desc = await timed("describe", steps.describe(pod, fv, k=2))
    # 4단계 GPS 장소 레코드 확인
    sup = sorted(SUPER)
    found = await timed("gpsname", steps.gps_records(pod, places, fv, sup, grec, desc))
    par = {g: g for g in sup}

    def fr(x):
        while par[x] != x:
            x = par[x]
        return x

    for a in sup:
        for b in sup:
            if a < b and steps.same_record(found.get(a), found.get(b)):
                par[fr(a)] = fr(b)
    # 5단계 같은 시설 판정: 레코드가 있고 아직 다른 묶음이며 GPS 사진 사이 거리가 2 km 안인 짝만
    # 묻는다
    cand = [
        (a, b)
        for a in sup
        for b in sup
        if a < b and found.get(a) and found.get(b) and fr(a) != fr(b) and sdist(a, b) <= FACILITY_M
    ]
    yes = await timed("facility", steps.same_facility_pairs(pod, fv, cand, found, sdist))
    for a, b in yes:
        par[fr(a)] = fr(b)
    log["facility"] = {"asked": len(cand), "same": len(yes)}
    for g in sup:
        if found.get(g):
            cls[g] = fr(g)
            grec[g] = dict(
                grec[g],
                name=found[g]["name"],
                category=found[g].get("category", ""),
                address=found[g].get("address", ""),
            )
    log["gpsname"] = {
        "places": len(sup),
        "with_record": sum(1 for g in sup if found.get(g)),
        "classes": len(set(cls.values())),
    }
    # 6단계 묶음 제안
    prop = await timed("propose", steps.propose(pod, fv, desc))
    log["propose"] = {"dup": prop["dup"], "bad": prop["bad"]}
    places_ = split_cl(steps.places_of(prop, fv.G), "cl_split_propose")
    places_ = [
        f["groups"]
        for f in must_link([{"record": None, "groups": pl} for pl in places_], "propose")
    ]
    places_ = time_split(places_, "propose")
    # 7단계 묶음 검토
    places_ = await timed("audit", steps.audit(pod, fv, places_))
    places_ = [
        f["groups"] for f in must_link([{"record": None, "groups": pl} for pl in places_], "audit")
    ]
    places_ = time_split(places_, "audit")
    gps_f = [{"record": gps_of(pl), "groups": pl} for pl in places_ if gps_of(pl)]
    nogps = [pl for pl in places_ if not gps_of(pl)]
    bigs = [pl for pl in nogps if size(fv, pl) >= BIG]
    smalls = [pl for pl in nogps if size(fv, pl) < BIG]
    # 8단계 큰 묶음 이름 붙이기. 실험처럼 단계마다 검색 도구(분류 검색 캐시)를 새로 만든다
    srch = steps.Searcher(places, centers, trip.regions)
    fb, log["name_big"] = (
        await timed("name_big", steps.name_clusters5(pod, fv, srch, bigs, desc, "name_big"))
        if bigs
        else ([], {})
    )
    fb = await drop_out_of_region(fb, "name_big")
    folders = gps_f + merge_ids(fb) + [{"record": None, "groups": pl} for pl in smalls]
    folders = time_split_f(attach_near_gps(folders, "near_gps_1"), "near_gps_1")
    # 9단계 다수결 배정. 앵커 카드는 26장까지, 넘으면 사진이 적은 앵커를 잠시 빼고 배정 뒤 되돌린다.
    named_k = sorted(
        [k for k, f in enumerate(folders) if f["record"]],
        key=lambda k: -size(fv, folders[k]["groups"]),
    )
    hidden = {k: folders[k]["record"] for k in named_k[MAX_CARDS:]}
    log["vote_anchors"] = len(named_k)
    vin = [
        {"record": None if k in hidden else f["record"], "groups": f["groups"]}
        for k, f in enumerate(folders)
    ]
    veto = tveto if tveto_min else None
    voted, log["vote"] = await timed("vote", steps.vote_cl(pod, fv, vin, far, veto))
    for f in voted:
        if f["record"] is None:
            home = next((k for k, of in enumerate(folders) if f["groups"][0] in of["groups"]), None)
            if home in hidden:
                f["record"] = hidden[home]
    folders = fix(must_link(voted, "vote"))
    # 10단계 남은 묶음 이름 붙이기
    named = [f for f in folders if f["record"]]
    unnamed = [f["groups"] for f in folders if not f["record"]]
    f2, log["name_rest"] = (
        await timed(
            "name_rest",
            steps.name_clusters5(
                pod, fv, steps.Searcher(places, centers, trip.regions), unnamed, desc, "name_rest"
            ),
        )
        if unnamed
        else ([], {})
    )
    f2 = await drop_out_of_region(f2, "name_rest")
    folders = merge_ids(named + f2)
    folders = fix(
        time_split_f(must_link(attach_near_gps(folders, "near_gps_2"), "name_rest"), "name_rest")
    )
    # 11단계 정리
    removed = await timed("purge", steps.named_purge(pod, fv, folders))
    new, ignored = [], 0
    for k, f in enumerate(folders):
        rm = set(removed[k])
        sup_k = [g for g in f["groups"] if g in SUPER]
        keep, solo = [], []
        for g in f["groups"]:
            if g in rm and g in SUPER and len(sup_k) < 2:
                ignored += 1
                keep.append(g)
            elif g in rm:
                solo.append(g)
            else:
                keep.append(g)
        if keep and not any(g in SUPER for g in keep) and sup_k:
            # GPS 그룹이 모두 빠진 경우: 가장 큰 GPS 그룹 하나는 남긴다
            g0 = max(sup_k, key=lambda g: len(fv.groups[g]))
            keep.append(g0)
            solo.remove(g0)
        if keep:
            new.append({"record": f["record"], "groups": keep})
        new += [{"record": None, "groups": [g]} for g in solo]
    folders = fix(must_link(new, "purge"))
    log["purge_ignored_super"] = ignored
    # 12단계 미분류 재배정(Qwen 유사도 순위)
    res2 = await timed("second", steps.second_select(pod, fv, folders))
    moved2, tblk2 = [], 0
    for g, k in res2:
        if k is None:
            continue
        if tveto_min and tveto(g, folders[k]["groups"]):
            tblk2 += 1
            continue
        moved2.append([g, k])
    mv = {g: k for g, k in moved2}
    new2 = []
    for k, f in enumerate(folders):
        if f["record"]:
            extra = [g for g, kk in mv.items() if kk == k]
            new2.append({"record": f["record"], "groups": f["groups"] + extra})
        else:
            rest = [g for g in f["groups"] if g not in mv]
            if rest:
                new2.append({"record": None, "groups": rest})
    folders = fix(new2)
    log["second"] = {
        "asked": len(res2),
        "picked": sum(1 for _, k in res2 if k is not None),
        "moved": len(moved2),
        "time_blocked": tblk2,
    }
    log["seconds"] = {k: round(v, 1) for k, v in secs.items()}
    return fv, folders, SUPER


# ---------- 13단계와 서비스 사진 상태 반영(실험에 없음)
def apply_folders(fv, folders, photos: list[PhotoState]) -> dict:
    """좌표가 있는 폴더(GPS 사진이 있거나 장소 레코드가 있음)만 장소로 두고 나머지는 위치 미분류로
    보낸다.

    - GPS 사진은 원래 좌표(EXIF)를 쓴다.
    - GPS 폴더의 GPS 없는 사진은 그 폴더 GPS 사진 가운데 가장 닮은 사진의 좌표를 받는다(결정 150,
    기존 서비스 규칙).
    - 레코드 폴더의 사진은 레코드 좌표를 받는다.
    장소 ID는 폴더의 가장 작은 사진 ID 순서로 p1부터 매긴다.
    """
    kept = []
    for f in folders:
        members = [fv.photos[i] for g in f["groups"] for i in fv.groups[g]]
        if not members:
            continue
        has_gps = any(p.source.has_gps for p in members)
        if has_gps or f["record"]:
            kept.append((members, has_gps, f["record"]))
    kept.sort(key=lambda item: min(p.source.trip_attachment_id for p in item[0]))
    placed: set[int] = set()
    record_folders = 0
    for index, (members, has_gps, record) in enumerate(kept):
        place_id = f"p{index + 1}"
        anchors = [p for p in members if p.source.has_gps]
        anchors.sort(key=lambda p: p.source.trip_attachment_id)
        for photo in members:
            photo.place_id = place_id
            photo.issue = None
            if photo.source.has_gps:
                photo.latitude, photo.longitude = photo.source.latitude, photo.source.longitude
                photo.region_origin = RegionOrigin.EXIF
            elif has_gps:
                nearest = max(anchors, key=lambda other: float(other.embedding @ photo.embedding))
                photo.latitude, photo.longitude = nearest.latitude, nearest.longitude
                photo.region_origin = RegionOrigin.INFERRED
            else:
                photo.latitude, photo.longitude = float(record["lat"]), float(record["lon"])
                photo.region_origin = RegionOrigin.INFERRED
            placed.add(photo.source.trip_attachment_id)
        record_folders += not has_gps
    for photo in photos:
        if photo.source.trip_attachment_id not in placed:
            photo.place_id = None
            photo.issue = Issue.UNCLEAR_LOCATION
            if photo.source.has_gps:
                # 지역 밖으로 뺀 GPS 사진은 LOCATING의 기간 밖 GPS 사진처럼 원래 좌표를 달고 나간다.
                photo.latitude, photo.longitude = photo.source.latitude, photo.source.longitude
                photo.region_origin = RegionOrigin.EXIF
            else:
                photo.latitude = photo.longitude = None
                photo.region_origin = RegionOrigin.UNKNOWN
    return {
        "places": len(kept),
        "record_places": record_folders,
        "unclassified": len(photos) - len(placed),
    }


async def _load_regions(places: Places, request: ProcessRequest):
    """요청 지역 좌표를 카카오 좌표→행정구역(법정동)으로 '시도 시군구' 이름으로 바꾼다.

    실험은 앱이 시청·군청 좌표로 만든 같은 형식의 이름을 썼다. 찾지 못한 지역은 이름에서 뺀다.
    돌려주는 값: (이름 목록, (시도, 시·군) 집합). 한 지역이라도 찾지 못하면 지역 밖 판정을 할 수
    없으므로 집합 대신 None을 돌려준다(빼지 않고 남기는 규칙).
    """
    names, keys, complete = [], set(), True
    for region in request.regions:
        found = await places.region(region.latitude, region.longitude)
        if found is None:
            complete = False
            continue
        names.append(" ".join(part for part in (found.sido, found.sigungu) if part))
        keys.add(region_key(found.sido, found.sigungu))
    return names, (keys if complete and keys else None)


async def _in_region(places: Places, lat: float, lon: float, keys) -> bool:
    """좌표의 시·군이 요청 지역 가운데 하나인지. 카카오 변환이 실패하거나 결과가 없으면 남긴다"""
    found = await places.region(lat, lon)
    return found is None or region_key(found.sido, found.sigungu) in keys


def exclude_out_of_period(photos: list[PhotoState], period) -> int:
    """LOCATING과 같은 기준(CLOCK 처리 시각이 있고 여행 기간 밖)으로 사진을 위치 미분류로 뺀다.

    좌표는 건드리지 않으므로 GPS 사진은 원래 좌표를 단 채 미분류로 나간다(LOCATING과 같음).
    """
    count = 0
    for photo in photos:
        if photo.issue is None and photo.taken_at is not None:
            day = photo.taken_at.astimezone(KST).date()
            if not period.start_date <= day <= period.end_date:
                photo.issue = Issue.UNCLEAR_LOCATION
                count += 1
    return count


def run_workflow(
    photos: list[PhotoState],
    request: ProcessRequest,
    ctx,
    images: dict[int, bytes],
    report: Callable[[int], None],
    check_cancel: Callable[[], None],
) -> dict:
    """사진 상태에 워크플로우 결과(place_id, 좌표, 출처, 위치 미분류)를 반영하고 집계를 돌려준다.

    images는 모든 사진의 분석용 사본 bytes. 모델 호출이 끝날 때마다 report(끝난 호출 수)를 부른다.
    취소와 시간 예산은 결과를 기다리는 동안 이 스레드에서 확인한다.
    """
    resources: WorkflowResources = ctx.workflow
    th = ctx.thresholds
    if missing := [k for k in REQUIRED if k not in th]:
        raise PipelineError(f"파이프라인 임계값이 누락됨: {missing}")
    started = time.monotonic()
    # 기간 밖으로 뺀 사진(LOCATING)은 대상이 아니다.
    targets = [photo for photo in photos if photo.issue is None]
    if not targets:
        ctx.log.info(
            "워크플로우 정리 생략", extra={"event": "ai_workflow_summary", "result": "skipped"}
        )
        return {"result": "skipped"}
    pod = Pod(resources.vlm, resources.concurrency, resources.vlm_retries)
    places = Places(resources)
    font = load_font(resources.font_path)
    centers = [(r.latitude, r.longitude) for r in request.regions]
    log: dict[str, Any] = {}

    async def main():
        regions, keys = await _load_regions(places, request)
        log["region_rule"] = keys is not None
        trip = await asyncio.to_thread(
            Trip,
            targets,
            images,
            name=request.trip_name,
            regions=regions,
            period=request.period,
            photo_seed=resources.photo_seed,
            group_min_sim=float(th["workflow_group_min_sim"]),
            font=font,
        )
        log["regions"] = regions
        return await organize(trip, pod, places, centers, th, log, region_keys=keys)

    future = resources.submit(main())
    fv, folders, _ = _wait(future, pod, resources.deadline, report, check_cancel)
    summary = apply_folders(fv, folders, targets)
    summary.update(
        log,
        model_calls=pod.stats["calls"],
        model_requests=pod.stats["requests"],
        model_errors=pod.stats["errors"],
        model_errors_by_stage={
            k.split(":", 1)[1]: v for k, v in pod.stats.items() if k.startswith("errors:")
        },
        model_truncated=pod.stats["truncated"],
        model_truncated_by_stage={
            k.split(":", 1)[1]: v for k, v in pod.stats.items() if k.startswith("truncated:")
        },
        kakao_calls=places.budget.used,
        kakao_errors=places.stats["kakao_errors"],
        kakao_over_budget=places.stats["kakao_over_budget"],
        duration_ms=round((time.monotonic() - started) * 1000, 2),
    )
    ctx.log.info("워크플로우 정리 완료", extra={"event": "ai_workflow_summary", **summary})
    return summary


def _wait(future: Future, pod: Pod, deadline: float, report, check_cancel):
    """코루틴이 끝날 때까지 기다리며 진행률, 취소, 시간 예산을 확인한다. 빠져나갈 때 남은 작업을
    멈춘다.
    """
    checked, reported = time.monotonic(), 0
    try:
        check_cancel()
        while True:
            done, _ = wait([future], timeout=POLL_SEC)
            if pod.done != reported:
                reported = pod.done
                report(reported)
            if done:
                try:
                    return future.result()
                except CancelledError:
                    raise PipelineCancelled() from None
            if time.monotonic() >= deadline:
                raise WorkflowTimeout("워크플로우 정리 단계의 시간 예산을 넘겼습니다.")
            if time.monotonic() - checked >= CANCEL_CHECK_SEC:
                checked = time.monotonic()
                check_cancel()
    finally:
        future.cancel()
