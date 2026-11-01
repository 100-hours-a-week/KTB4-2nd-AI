"""전국 공공 장소 원자료로 정리 에이전트용 SQLite 색인을 만든다(#151).

사용: uv run python -m scripts.build_place_index <원자료 폴더> <출력.sqlite>

원자료 폴더 구성(수집 스크립트는 팀 저장소 밖 프로젝트 기록에 둔다)
    tourapi/tourapi_all.json      TourAPI KorService2 areaBasedList2 전국
    std_tourist/*.csv             전국관광지정보표준데이터 제공기관별 파일
    semas/semas_all.csv.gz        소상공인시장진흥공단 상가(상권)정보 전국
    wikidata/wikidata_items.json  Wikidata 시도별 P131 계층, 좌표와 한국어 라벨
    overture/overture_kr.csv      Overture Places 대한민국 범위
    osm/osm_pois.csv              Geofabrik 대한민국 추출본의 이름 있는 POI

출처별 이름과 분류 필드는 항목 2 공공 자료 측정(2026-10-26)과 평가 도구 색인(2026-10-28)을 따른다.
좌표가 없거나 0이거나 대한민국 범위(위도 33~38.7, 경도 124.5~131.9) 밖인 행, 이름이 없는 행은 뺀다.
"""

import csv
import gzip
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from app.infra.place_index import SCHEMA_VERSION, file_sha256, normalize

LAT, LON = (33.0, 38.7), (124.5, 131.9)
OSM_NAME_FIELDS = [
    "name", "name_ko", "name_en", "name_ko-Latn", "alt_name", "alt_name_ko", "old_name",
    "official_name", "short_name", "loc_name", "brand", "brand_ko",
]  # fmt: skip


def tourapi(raw: Path):
    for x in json.loads((raw / "tourapi/tourapi_all.json").read_text(encoding="utf-8"))["items"]:
        yield (
            x["contentid"],
            x.get("mapy"),
            x.get("mapx"),
            [x.get("title")],
            f"type={x['contenttypeid']}",
        )


def std_tourist(raw: Path):
    for path in sorted((raw / "std_tourist").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for i, x in enumerate(csv.DictReader(handle)):
                yield (
                    f"{path.stem}#{i}",
                    x.get("위도"),
                    x.get("경도"),
                    [x.get("관광지명")],
                    x.get("관광지구분"),
                )


def semas(raw: Path):
    with gzip.open(raw / "semas/semas_all.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for x in csv.DictReader(handle):
            names = [x["상호명"]]
            if x["지점명"]:
                names.append(f"{x['상호명']} {x['지점명']}")
            yield x["상가업소번호"], x["위도"], x["경도"], names, x["상권업종소분류명"]


def wikidata(raw: Path):
    data = json.loads((raw / "wikidata/wikidata_items.json").read_text(encoding="utf-8"))
    for x in data["items"]:
        names = [x["label_ko"], x.get("label_en")] + x.get("alias_ko", "").split("|")
        yield x["qid"], x["lat"], x["lon"], names, x.get("p31_ko")


def overture(raw: Path):
    with (raw / "overture/overture_kr.csv").open(encoding="utf-8", newline="") as handle:
        for x in csv.DictReader(handle):
            names = [x["name_primary"]]
            if x["names_common"]:
                names += list(json.loads(x["names_common"]).values())
            names += x["names_rules"].split("|") + [x["brand_primary"]]
            yield x["id"], x["lat"], x["lon"], names, x["basic_category"]


def osm(raw: Path):
    with (raw / "osm/osm_pois.csv").open(encoding="utf-8", newline="") as handle:
        for x in csv.DictReader(handle):
            names = [x.get(k) for k in OSM_NAME_FIELDS]
            if x.get("branch"):
                for base in ("name", "brand"):
                    if x.get(base) and x["branch"] not in x[base]:
                        names.append(f"{x[base]} {x['branch']}")
            if x["other_names"]:
                for value in json.loads(x["other_names"]).values():
                    names += value.split(";")
            yield f"{x['osm_type']}/{x['osm_id']}", x["lat"], x["lon"], names, x["main_tag"]


SOURCES = {
    "tourapi": tourapi,
    "std_tourist": std_tourist,
    "semas": semas,
    "wikidata": wikidata,
    "overture": overture,
    "osm": osm,
}


def coordinate(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def build(raw: Path, out: Path) -> dict:
    partial = out.with_name(out.name + ".building")
    partial.unlink(missing_ok=True)
    db = sqlite3.connect(partial)
    db.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE place(
            id INTEGER PRIMARY KEY, source TEXT NOT NULL, source_id TEXT NOT NULL,
            name TEXT NOT NULL, category TEXT, lat REAL NOT NULL, lon REAL NOT NULL,
            norms TEXT NOT NULL);
        CREATE TABLE place_name(norm TEXT NOT NULL, place_id INTEGER NOT NULL,
            PRIMARY KEY(norm, place_id)) WITHOUT ROWID;
        CREATE VIRTUAL TABLE place_rtree USING rtree(id, min_lat, max_lat, min_lon, max_lon);
        CREATE VIRTUAL TABLE place_name_fts
            USING fts5(norm, place_id UNINDEXED, tokenize='trigram');
        """
    )
    kept, dropped, pid = Counter(), Counter(), 0
    for source, read in SOURCES.items():
        seen: set[str] = set()
        for source_id, lat, lon, names, category in read(raw):
            lat, lon = coordinate(lat), coordinate(lon)
            if (
                lat is None
                or lon is None
                or not (LAT[0] <= lat <= LAT[1] and LON[0] <= lon <= LON[1])
            ):
                dropped[f"{source}:coordinate"] += 1
                continue
            labels = list(dict.fromkeys(n.strip() for n in names if n and n.strip()))
            norms = [n for n in dict.fromkeys(normalize(label) for label in labels) if n]
            if not norms:
                dropped[f"{source}:name"] += 1
                continue
            if source_id in seen:
                dropped[f"{source}:duplicate"] += 1
                continue
            seen.add(source_id)
            pid += 1
            db.execute(
                "INSERT INTO place VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                # 정규화한 이름에는 |가 없으므로, 이어 붙여도 이름 경계를 넘는 일치가 생기지 않는다.
                (pid, source, str(source_id), labels[0], category or "", lat, lon, "|".join(norms)),
            )
            db.execute("INSERT INTO place_rtree VALUES (?, ?, ?, ?, ?)", (pid, lat, lat, lon, lon))
            db.executemany("INSERT INTO place_name VALUES (?, ?)", [(n, pid) for n in norms])
            db.executemany(
                "INSERT INTO place_name_fts(norm, place_id) VALUES (?, ?)",
                [(n, pid) for n in norms if len(n) >= 3],
            )
            kept[source] += 1
        db.commit()
        print(source, kept[source], flush=True)
    db.execute("INSERT INTO place_name_fts(place_name_fts) VALUES ('optimize')")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "counts": json.dumps(dict(kept), ensure_ascii=False),
        "dropped": json.dumps(dict(dropped), ensure_ascii=False),
    }
    db.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
    db.commit()
    db.execute("VACUUM")
    db.close()
    partial.replace(out)
    return {**meta, "sha256": file_sha256(out), "bytes": out.stat().st_size}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(build(Path(sys.argv[1]), Path(sys.argv[2])), ensure_ascii=False, indent=1))
