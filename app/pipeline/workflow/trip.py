"""워크플로우가 보는 여행: 섞은 사진 순서, 임베딩 유사도, 1단계 그룹, 2단계 메타데이터 잇기, 격자
그림.

실험 코드(2026-11-05 step3-agent)의 agent_common.Trip, meta.py, 격자 함수(montage, photo_uri,
jpeg)를
서비스 사진 상태(PhotoState)에 맞춰 옮겼다. 계산 순서와 동점 처리는 실험과 같게 두었다.
- 사진 순서를 시드로 섞는다. 이후 모든 번호(0..n-1)는 섞은 순서의 번호이며, 사진 ID와 저장 키는
  모델에 보내지 않는다(실험의 누출 방지 규칙). 시드는 실험 기본값 61106이다.
- 사진 파일 대신 분석용 사본 bytes를 연다. 실험은 같은 사본(analyze_copy)을 파일로 열었다.
"""

import base64
import bisect
import io
import json
import random
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.pipeline.state import PhotoState
from app.schemas.process import Period

GROUP_SEED = 1106  # 실험의 그룹 번호 섞기 시드(agent_common.GROUP_SEED)
FONT_SIZE = 34
# 실험은 macOS의 Arial Bold를 썼다. 컨테이너에는 없으므로 없으면 다음 후보를 쓴다.
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def load_font(path: str | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """격자 칸 번호용 글꼴. 지정 경로, 실험 글꼴, DejaVu 순으로 찾고 없으면 Pillow 기본 글꼴"""
    for candidate in ([path] if path else []) + list(FONT_CANDIDATES):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, FONT_SIZE)
    return ImageFont.load_default(size=FONT_SIZE)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


class Trip:
    """파이프라인이 볼 수 있는 것: 섞은 순서의 사진, 임베딩, 여행 이름·지역·기간. 사진 ID는 쓰지
    않는다.
    """

    def __init__(
        self,
        photos: list[PhotoState],
        images: dict[int, bytes],
        *,
        name: str,
        regions: list[str],
        period: Period,
        photo_seed: int,
        group_min_sim: float,
        font=None,
    ) -> None:
        perm = list(range(len(photos)))
        random.Random(photo_seed).shuffle(perm)
        self.photos = [photos[i] for i in perm]
        self._images = images
        self.font = font if font is not None else load_font()
        X = np.array([p.embedding for p in self.photos], float)
        self.X = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.S = self.X @ self.X.T
        self.n = len(self.photos)
        self.groups = self._step1(group_min_sim)
        self.G = len(self.groups)
        self.name, self.regions = name, regions
        self.period = period
        self.days = (period.end_date - period.start_date).days + 1

    def _step1(self, thr: float) -> list[list[int]]:
        """코사인 유사도 thr 이상인 사진 쌍을 단일 연결로 잇는다(결정 186). 그룹 번호는 시드로
        섞는다.
        """
        n = self.n
        parent = list(range(n))

        def f(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if self.S[i, j] >= thr:
                    parent[f(i)] = f(j)
        comp: dict[int, list[int]] = {}
        for i in range(n):
            comp.setdefault(f(i), []).append(i)
        groups = list(comp.values())
        random.Random(GROUP_SEED).shuffle(groups)
        return groups

    # ---------- 실험의 Trip 메서드
    def medoid(self, g: int) -> int:
        m = self.groups[g]
        if len(m) == 1:
            return m[0]
        sub = self.S[np.ix_(m, m)].copy()
        np.fill_diagonal(sub, 0.0)
        return m[int(np.argmax(sub.sum(1)))]

    def diverse(self, g: int, k: int) -> list[int]:
        """대표 사진부터 시작해 이미 고른 사진과 가장 덜 닮은 사진을 차례로 고른다(최대 k장)."""
        m = self.groups[g]
        chosen = [self.medoid(g)]
        while len(chosen) < min(k, len(m)):
            rest = [i for i in m if i not in chosen]
            chosen.append(min(rest, key=lambda i: max(self.S[i, c] for c in chosen)))
        return chosen

    def group_sim(self) -> np.ndarray:
        G = self.G
        M = np.zeros((G, G))
        for a in range(G):
            for b in range(a + 1, G):
                M[a, b] = M[b, a] = self.S[np.ix_(self.groups[a], self.groups[b])].max()
        return M

    def trip_text(self) -> str:
        """실험의 trip_text(with_period=True). 여행 이름, 지역 이름, 기간을 JSON 한 줄로"""
        d = {
            "name": self.name,
            "regions": self.regions,
            "period": f"{self.period.start_date} ~ {self.period.end_date} ({self.days}일)",
        }
        return json.dumps(d, ensure_ascii=False)

    # ---------- 그림
    def open(self, i: int) -> Image.Image:
        photo_id = self.photos[i].source.trip_attachment_id
        return Image.open(io.BytesIO(self._images[photo_id])).convert("RGB")

    def photo_uri(self, i: int, side: int = 896) -> str:
        im = self.open(i)
        im.thumbnail((side, side))
        return jpeg(im)

    def montage(self, cells: list[tuple[int, str]], cols: int = 4, cell: int = 256) -> Image.Image:
        """(사진 번호, 표시 글자) 목록을 cols×rows 격자 한 장으로. 칸 왼쪽 위에 글자를 노란 바탕으로
        쓴다.
        """
        rows = (len(cells) + cols - 1) // cols
        canvas = Image.new("RGB", (cell * cols, cell * rows), (128, 128, 128))
        draw = ImageDraw.Draw(canvas)
        for pos, (i, label) in enumerate(cells):
            im = self.open(i)
            im.thumbnail((cell - 4, cell - 4))
            x, y = (pos % cols) * cell, (pos // cols) * cell
            canvas.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
            if label:
                label_box(draw, x, y, label, self.font)
        return canvas


def label_box(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, font) -> None:
    box = draw.textbbox((x + 6, y + 4), label, font=font)
    draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 4), fill=(255, 255, 0))
    draw.text((x + 6, y + 4), label, fill=(0, 0, 0), font=font)


def jpeg(im: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class View(Trip):
    """Trip과 같은 사진·임베딩을 쓰되 그룹 목록만 바꾼 보기. 그룹 번호 0..len(groups)-1."""

    def __init__(self, trip: Trip, groups: list[list[int]]) -> None:  # noqa: D107
        self.__dict__.update({k: v for k, v in trip.__dict__.items() if k not in ("groups", "G")})
        self.groups = [list(m) for m in groups]
        self.G = len(self.groups)


# ---------- 2단계: 메타데이터로 잇기(meta.py)
def rows_of(trip: Trip) -> list[dict]:
    """섞은 사진 순서대로 BE 메타데이터. 원본 값(PhotoSource)을 쓰며 CLOCK 단계의 보정값은 쓰지
    않는다.
    """
    rows = []
    for p in trip.photos:
        s = p.source
        rows.append(
            {
                "gps": s.latitude is not None and s.longitude is not None,
                "lat": s.latitude,
                "lon": s.longitude,
                "t": s.taken_at,
                "dev": s.device_model,
            }
        )
    return rows


class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def f(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a: int, b: int) -> None:
        a, b = self.f(a), self.f(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def clock(rows: list[dict], X: np.ndarray, pair_min_sim: float):
    """실험의 시각 보정(meta.clock). 서비스 CLOCK과 달리 최소 쌍 수와 허용 오차가 없다(쌍 1개로도
    정함).

    GPS와 시각을 함께 가진 기기를 기준으로, 다른 기기마다 서로 가장 닮은 쌍(유사도 pair_min_sim
    이상)의
    시각 차이를 일 단위로 반올림한 최빈값을 그 기기의 일수 차이로 둔다. 기준 기기가 없으면 모든
    시각을
    버리고, 맞는 쌍이 없는 기기의 시각도 버린다.
    """
    n = len(rows)
    ref_dev = {r["dev"] for r in rows if r["gps"] and r["t"] is not None and r["dev"] is not None}
    offs: dict = {d: 0 for d in ref_dev}
    if ref_dev:
        ref = [i for i in range(n) if rows[i]["dev"] in ref_dev and rows[i]["t"] is not None]
        for dev in sorted({r["dev"] for r in rows if r["t"] is not None} - ref_dev, key=str):
            tg = [i for i in range(n) if rows[i]["dev"] == dev and rows[i]["t"] is not None]
            S = X[tg] @ X[ref].T
            o = []
            for a in range(len(tg)):
                b = int(np.argmax(S[a]))
                if S[a, b] >= pair_min_sim and int(np.argmax(S[:, b])) == a:
                    o.append((rows[ref[b]]["t"] - rows[tg[a]]["t"]).total_seconds())
            if not o:
                offs[dev] = None
                continue
            offs[dev] = Counter(round(x / 86400) for x in o).most_common(1)[0][0]
    T = {}
    for i, r in enumerate(rows):
        d = offs.get(r["dev"]) if r["t"] is not None else None
        T[i] = r["t"] + timedelta(days=d) if d is not None else None
    return T, offs


def stage2(trip: Trip, *, eps: float, gap_min: float, pair_min_sim: float) -> dict:
    """GPS 그룹 잇기(eps m), 시각 보정, 끼워 넣기(gap_min분, 바뀌지 않을 때까지)"""
    rows = rows_of(trip)
    groups, G = trip.groups, trip.G
    gps_groups = [g for g in range(G) if any(rows[i]["gps"] for i in groups[g])]
    dsu = DSU(G)
    for x in range(len(gps_groups)):
        for y in range(x + 1, len(gps_groups)):
            a = [i for i in groups[gps_groups[x]] if rows[i]["gps"]]
            b = [i for i in groups[gps_groups[y]] if rows[i]["gps"]]
            D = haversine(
                np.array([[rows[i]["lat"]] for i in a]),
                np.array([[rows[i]["lon"]] for i in a]),
                np.array([[rows[j]["lat"] for j in b]]),
                np.array([[rows[j]["lon"] for j in b]]),
            )
            if D.min() <= eps:
                dsu.u(gps_groups[x], gps_groups[y])
    pog = {g: dsu.f(g) for g in gps_groups}
    T, offs = clock(rows, trip.X, pair_min_sim)
    place = {i: pog[g] for g in pog for i in groups[g]}
    gap = timedelta(minutes=gap_min)
    inserted: dict[int, int] = {}
    rounds = 0
    while True:
        rounds += 1
        tl = sorted([i for i in place if T[i] is not None], key=lambda i: (T[i], i))
        tl_t = [T[i] for i in tl]
        added = 0
        for g in range(G):
            if g in pog or g in inserted:
                continue
            votes = []
            for i in groups[g]:
                if T[i] is None:
                    continue
                kb, ka = bisect.bisect_right(tl_t, T[i]) - 1, bisect.bisect_left(tl_t, T[i])
                if kb < 0 or ka >= len(tl):
                    continue
                prev, nxt = tl[kb], tl[ka]
                if place[prev] == place[nxt] and T[i] - T[prev] <= gap and T[nxt] - T[i] <= gap:
                    votes.append(place[prev])
            if votes and len(set(votes)) == 1:
                inserted[g] = votes[0]
                for i in groups[g]:
                    place[i] = votes[0]
                added += 1
        if not added:
            break
    final = {**pog, **inserted}
    roots = sorted(set(final.values()))
    places = [[g for g in range(G) if final.get(g) == r] for r in roots]
    return {
        "rows": rows,
        "T": T,
        "offs": offs,
        "places": places,
        "inserted": sorted(inserted),
        "rounds": rounds,
        "rest": [g for g in range(G) if g not in final],
    }


def gps_record(view: View, g: int, rows: list[dict], k: int) -> dict:
    """BE가 GPS 좌표를 역지오코딩해 이름을 붙인다는 가정의 레코드. 이름은 카드 글자 구분용
    자리표시자다.
    """
    pts = [(rows[i]["lat"], rows[i]["lon"]) for i in view.groups[g] if rows[i]["gps"]]
    lat, lon = float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts]))
    return {
        "id": f"gps:{k}",
        "name": f"GPS 장소 {k + 1}",
        "lat": lat,
        "lon": lon,
        "category": "사진의 GPS 좌표로 정한 장소",
        "address": "좌표로 정함",
        "gps": True,
        "points": pts,
    }


def build_views(trip: Trip, **stage2_args):
    """full_view: 2단계 장소에 들지 못한 R개 그룹 + 장소마다 하나로 합친 그룹(R..R+K-1)."""
    s = stage2(trip, **stage2_args)
    rest = s["rest"]
    R = len(rest)
    rest_groups = [trip.groups[g] for g in rest]
    place_groups = [[i for g in pl for i in trip.groups[g]] for pl in s["places"]]
    full_view = View(trip, rest_groups + place_groups)
    gps_folders = [
        {"record": gps_record(full_view, R + k, s["rows"], k), "groups": [R + k]}
        for k in range(len(place_groups))
    ]
    return s, full_view, gps_folders
