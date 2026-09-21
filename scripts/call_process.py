"""백엔드 대신 사진 정리 요청을 보내는 호출 스크립트. EC2 단독 검증과 백엔드 호출 예시용

사본 키 목록으로 명세대로 요청 JSON을 만들어 POST /trips/{trip_id}/process에 보내고
응답을 그대로 출력. v1은 동기라 완료까지 기다림(타임아웃 35분).
--cancel-after N이면 N초 뒤 별도 스레드에서 DELETE를 보내 취소 경로 확인.
--print-only면 보내지 않고 요청 JSON만 출력, 백엔드에 줄 호출 예시

사용
    uv run scripts/call_process.py --api http://127.0.0.1:8000 --key <API_KEY> --trip 77 \\
        --keys trips/77/analyze/a.jpg trips/77/analyze/b.jpg
    uv run scripts/call_process.py ... --no-gps            좌표 없는 사진, 전부 미분류
    uv run scripts/call_process.py ... --cancel-after 3    3초 뒤 DELETE
    uv run scripts/call_process.py ... --print-only        요청 JSON만
"""

import argparse
import json
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

KST = ZoneInfo("Asia/Seoul")
WAIT_TIMEOUT = 35 * 60


def build_request(keys: list[str], gps: bool) -> dict:
    start = datetime(2026, 10, 12, 6, 0, tzinfo=KST)
    attachments = []
    for i, key in enumerate(keys):
        attachments.append(
            {
                "trip_attachment_id": 100 + i,
                "analyze_storage_key": key,
                "taken_at": (start + timedelta(minutes=10 * i)).isoformat(),
                "latitude": 33.4580 + i * 0.0002 if gps else None,
                "longitude": 126.9423 if gps else None,
                "device_model": "Apple iPhone 15",
            }
        )
    return {
        "trip_name": "연동 시험",
        "period": {"start_date": "2026-10-12", "end_date": "2026-10-14"},
        "regions": [{"latitude": 33.4996, "longitude": 126.5312}],
        "attachments": attachments,
        "existing_places": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--key", required=True, help="API_KEY")
    parser.add_argument("--trip", type=int, default=77)
    parser.add_argument("--keys", nargs="+", required=True, help="analyze_storage_key 목록")
    parser.add_argument("--no-gps", action="store_true", help="좌표 없이, 전부 미분류")
    parser.add_argument("--cancel-after", type=float, help="N초 뒤 DELETE")
    parser.add_argument("--print-only", action="store_true", help="요청 JSON만 출력")
    args = parser.parse_args(argv)

    body = build_request(args.keys, gps=not args.no_gps)
    if args.print_only:
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return 0

    headers = {"Authorization": f"Bearer {args.key}"}
    url = f"{args.api}/trips/{args.trip}/process"

    if args.cancel_after:

        def cancel() -> None:
            time.sleep(args.cancel_after)
            res = httpx.delete(url, headers=headers, timeout=10)
            print(f"DELETE {res.status_code} {res.text}")

        threading.Thread(target=cancel, daemon=True).start()

    started = datetime.now(UTC)
    print(f"POST {url} 사진 {len(body['attachments'])}장")
    res = httpx.post(url, json=body, headers=headers, timeout=WAIT_TIMEOUT)
    elapsed = (datetime.now(UTC) - started).total_seconds()
    print(f"{res.status_code} {elapsed:.1f}s")
    try:
        print(json.dumps(res.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(res.text)
    return 0 if res.status_code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
