"""정답 기반 평가 JSON을 계산하고 선택적으로 Redis 대시보드 지표에 반영한다."""

import argparse
import asyncio
import json
from pathlib import Path

from redis.asyncio import Redis

from app.jobs.config import JobsSettings
from app.observability.quality import EvaluationInput, evaluate, publish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    report = evaluate(EvaluationInput.model_validate_json(args.input.read_text()))
    output = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(output + "\n")
    print(json.dumps(report, ensure_ascii=False))
    if args.publish:

        async def send():
            cfg = JobsSettings()
            redis = Redis.from_url(
                cfg.REDIS_URL.get_secret_value(),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            try:
                await publish(redis, cfg.redis_prefix, report)
            finally:
                await redis.aclose()

        asyncio.run(send())


if __name__ == "__main__":
    main()
