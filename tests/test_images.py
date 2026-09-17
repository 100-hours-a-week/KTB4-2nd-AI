"""이미지 소스. 로컬 폴더는 실제 파일로, S3는 가짜 클라이언트로 재시도와 실패 수집을 검증"""

import pytest

from app.core.errors import DownloadFailed
from app.infra.images import LocalDirImageSource, S3ImageSource


@pytest.fixture
def photo_dir(tmp_path):
    (tmp_path / "trips/77/analyze").mkdir(parents=True)
    (tmp_path / "trips/77/analyze/a.jpg").write_bytes(b"AAA")
    (tmp_path / "trips/77/analyze/b.jpg").write_bytes(b"BBBB")
    return tmp_path


def test_local_reads_files(photo_dir):
    src = LocalDirImageSource(photo_dir)
    got = src.get_many(["trips/77/analyze/a.jpg", "trips/77/analyze/b.jpg"])
    assert got == {"trips/77/analyze/a.jpg": b"AAA", "trips/77/analyze/b.jpg": b"BBBB"}


def test_local_collects_all_missing_keys(photo_dir):
    src = LocalDirImageSource(photo_dir)
    with pytest.raises(DownloadFailed) as info:
        src.get_many(["trips/77/analyze/a.jpg", "trips/77/analyze/x.jpg", "y.jpg"])
    assert info.value.keys == ["trips/77/analyze/x.jpg", "y.jpg"]
    assert info.value.detail == {"keys": ["trips/77/analyze/x.jpg", "y.jpg"]}


def test_local_empty_keys(photo_dir):
    assert LocalDirImageSource(photo_dir).get_many([]) == {}


class _FakeS3Client:
    """bad로 시작하는 키는 항상 실패, 호출 횟수를 셈"""

    def __init__(self):
        self.calls: list[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803, boto3 시그니처와 동일하게
        self.calls.append(Key)
        if Key.startswith("bad"):
            raise RuntimeError("SlowDown")

        class Body:
            def read(self_inner):
                return b"OK-" + Key.encode()

        return {"Body": Body()}


@pytest.fixture
def s3():
    src = S3ImageSource(bucket="b", region="ap-northeast-2", retries=3, backoff_seconds=0.0)
    src._client = _FakeS3Client()
    return src


def test_s3_parallel_keeps_order(s3):
    got = s3.get_many(["k1", "k2", "k3"])
    assert list(got) == ["k1", "k2", "k3"]
    assert got["k2"] == b"OK-k2"


def test_s3_retries_then_collects_failures(s3):
    with pytest.raises(DownloadFailed) as info:
        s3.get_many(["k1", "bad1", "k2", "bad2"])
    assert info.value.keys == ["bad1", "bad2"]
    assert s3._client.calls.count("bad1") == 3
    assert s3._client.calls.count("k1") == 1
