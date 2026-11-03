"""공공 장소 색인 검색. 정리 에이전트가 사진 속 장소 후보를 찾을 때 쓴다.

색인은 scripts/build_place_index.py가 만든 읽기 전용 SQLite 파일이다.
프로세스마다 파일을 열기만 하고 메모리에 올리지 않으므로,
정리 작업 자식과 대기 자식이 늘어도 색인을 다시 적재하지 않는다(#151).

이름 일치는 평가 도구(2026-10-28 agent.py의 kb_search)와 같은 세 단계다.
3: 정규화한 이름이 질의와 같음, 2: 이름이 질의를 포함(질의 2자 이상),
1: 질의가 이름을 포함(이름 3자 이상).
반경 안의 일치 후보를 점수가 높고 가까운 순으로 돌려준다.
"""

import fcntl
import hashlib
import math
import os
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "2"
MAX_RADIUS_M = 30_000
MAX_TWO_CHAR_RADIUS_M = 5_000
# 질의 안의 이름(1단계)은 부분 문자열마다 찾으므로 길이를 제한한다.
MAX_QUERY_CHARS = 64
EARTH_RADIUS_M = 6_371_000


def normalize(name: str | None) -> str:
    """NFKC, 소문자로 바꾸고 글자와 숫자만 남긴다. 띄어쓰기와 기호 차이를 무시하려는 것이다."""
    text = unicodedata.normalize("NFKC", name or "").lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class PlaceCandidate:
    candidate_id: str
    name: str
    category: str
    source: str
    latitude: float
    longitude: float
    distance_m: int


class PlaceIndex:
    """읽기 전용 색인. 한 연결을 잠금으로 보호해 여러 스레드에서 불러도 된다."""

    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"공공 장소 색인이 없음: {path}")
        # immutable이면 SQLite가 잠금과 변경 확인을 생략한다. 색인 파일은 교체할 뿐 고치지 않는다.
        self._db = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True, check_same_thread=False
        )
        self._lock = threading.Lock()
        version = self._meta("schema_version")
        if version != SCHEMA_VERSION:
            self._db.close()
            raise ValueError(f"공공 장소 색인 스키마가 다름: {version}")

    def _meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def search(
        self, query: str, latitude: float, longitude: float, radius_m: int, limit: int = 10
    ) -> list[PlaceCandidate]:
        q = normalize(query)[:MAX_QUERY_CHARS]
        if not q or limit < 1:
            return []
        if not all(map(math.isfinite, (latitude, longitude, radius_m))):
            raise ValueError("좌표와 반경은 유한한 수여야 합니다.")
        # 두 글자는 trigram 색인을 못 써 반경 안 이름을 훑으므로 반경을 더 좁힌다(결정 153).
        cap = MAX_TWO_CHAR_RADIUS_M if len(q) == 2 else MAX_RADIUS_M
        radius_m = max(1, min(int(radius_m), cap))
        # 거리 계산과 같은 지구 반지름으로 상자를 잡고 1% 여유를 둔다. 정확한 거리는 아래에서 거름
        dlat = math.degrees(radius_m / EARTH_RADIUS_M) * 1.01
        dlon = dlat / max(math.cos(math.radians(min(abs(latitude) + dlat, 89.0))), 0.01)
        box = (latitude - dlat, latitude + dlat, longitude - dlon, longitude + dlon)
        best: dict[int, tuple] = {}

        def collect(score: int, rows) -> None:
            for pid, source, source_id, name, category, lat, lon in rows:
                if pid in best and best[pid][0] <= -score:
                    continue
                distance = haversine_m(latitude, longitude, lat, lon)
                if distance <= radius_m:
                    best[pid] = (-score, distance, source, source_id, name, category, lat, lon)

        with self._lock:
            # 3: 이름이 질의와 같음
            collect(3, self._query("JOIN place_name n ON n.place_id = p.id", "n.norm = ?", q, box))
            # 2: 이름이 질의를 포함. 3자 이상은 trigram 색인, 2자는 반경 상자 안의 이름을 훑는다.
            if len(q) >= 3:
                join = "JOIN place_name_fts ON place_name_fts.place_id = p.id"
                collect(2, self._query(join, "place_name_fts MATCH ?", _phrase(q), box))
            elif len(q) == 2:
                collect(2, self._query("", "instr(p.norms, ?) > 0", q, box))
            # 1: 질의가 이름을 포함. 질의의 3자 이상 부분 문자열과 같은 이름을 찾는다.
            parts = {q[i:j] for i in range(len(q)) for j in range(i + 3, len(q) + 1)} - {q}
            for part in sorted(parts):
                collect(
                    1,
                    self._query("JOIN place_name n ON n.place_id = p.id", "n.norm = ?", part, box),
                )
        hits = sorted(best.values(), key=lambda hit: (hit[0], hit[1], hit[2], hit[3]))
        return [
            PlaceCandidate(
                candidate_id=f"kb:{source}:{source_id}",
                name=name,
                category=category or "",
                source=source,
                latitude=lat,
                longitude=lon,
                distance_m=round(distance),
            )
            for _, distance, source, source_id, name, category, lat, lon in hits[:limit]
        ]

    def _query(self, join: str, condition: str, value: str, box: tuple[float, ...]):
        # 반경의 외접 상자로 먼저 거르고, 정확한 거리는 호출자가 계산한다.
        return self._db.execute(
            "SELECT p.id, p.source, p.source_id, p.name, p.category, p.lat, p.lon "
            f"FROM place_rtree r JOIN place p ON p.id = r.id {join} "
            "WHERE r.max_lat >= ? AND r.min_lat <= ? AND r.max_lon >= ? AND r.min_lon <= ? "
            f"AND {condition}",
            (*box, value),
        )

    def close(self) -> None:
        self._db.close()


def _phrase(text: str) -> str:
    # FTS5 구문으로 해석되지 않도록 큰따옴표 구문으로 감싼다. 정규화 후에는 따옴표가 남지 않는다.
    return '"' + text.replace('"', '""') + '"'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_local_index(path: Path, sha256: str, download) -> Path:
    """path에 sha256이 같은 색인이 없으면 download(임시 경로)로 받아 바꿔 넣는다.

    여러 프로세스가 동시에 불러도 파일 잠금으로 한 번만 받는다. 확인한 해시와 파일 크기, 수정 시각을
    옆 파일에 남겨 다음 호출부터는 큰 파일을 다시 읽지 않는다.
    파일이 바뀌면 표지가 맞지 않아 다시 받는다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = path.with_name(path.name + ".sha256")
    with open(path.with_name(path.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file() and stamp.is_file() and stamp.read_text() == _stamp(path, sha256):
            return path
        partial = path.with_name(path.name + ".partial")
        download(partial)
        if file_sha256(partial) != sha256:
            partial.unlink(missing_ok=True)
            raise ValueError("공공 장소 색인의 해시가 설정값과 다름")
        # 바꿔 넣는 도중 중단되어도 이전 표지가 새 파일을 보증하지 않도록 먼저 지운다.
        stamp.unlink(missing_ok=True)
        os.replace(partial, path)
        stamp.write_text(_stamp(path, sha256))
    return path


def _stamp(path: Path, sha256: str) -> str:
    info = path.stat()
    return f"{sha256} {info.st_size} {info.st_mtime_ns}"


def s3_downloader(uri: str, region: str):
    """s3://버킷/키를 받는 함수를 만든다. 자격 증명은 실행기의 인스턴스 역할을 쓴다."""
    import boto3

    bucket, _, key = uri.removeprefix("s3://").partition("/")

    def download(target: Path) -> None:
        boto3.client("s3", region_name=region).download_file(bucket, key, str(target))

    return download


def prepare_place_index(cfg) -> Path | None:
    """실행기 시작 때 한 번 부른다. 설정이 없으면 None, S3 URI가 있으면 받아서 경로를 돌려준다."""
    if not cfg.PLACE_INDEX_PATH:
        return None
    path = Path(cfg.PLACE_INDEX_PATH)
    if cfg.PLACE_INDEX_S3_URI:
        ensure_local_index(
            path, cfg.PLACE_INDEX_SHA256, s3_downloader(cfg.PLACE_INDEX_S3_URI, cfg.AWS_REGION)
        )
    # 시작할 때 한 번 열어 파일 손상이나 스키마 차이를 작업 전에 드러낸다.
    PlaceIndex(path).close()
    return path
