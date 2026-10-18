"""관측 테스트의 공용 도구. 로그 이벤트 조회와 민감 정보 노출 확인"""

import logging

import pytest

from app.core.logging import JsonFormatter

# 로그에 남으면 안 되는 값. 사진 키와 presigned URL 서명 형태를 함께 담는다.
SECRET = "trips/77/secret-key.jpg?X-Amz-Signature=abc"


def events(caplog, name):
    return [record for record in caplog.records if getattr(record, "event", None) == name]


def rendered(caplog) -> str:
    """운영 형식(JSON 한 줄)으로 직렬화한 전체 로그"""
    formatter = JsonFormatter()
    return "\n".join(formatter.format(record) for record in caplog.records)


def completion(content="응답", finish_reason="stop", usage=None):
    body = {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ]
    }
    if usage is not None:
        body["usage"] = usage
    return body


@pytest.fixture(autouse=True)
def info_logs(caplog):
    caplog.set_level(logging.INFO)
