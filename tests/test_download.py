import pytest
from PIL import Image

from app.core.errors import DecodeFailed, DownloadFailed
from app.infra.images import LocalDirImageSource
from app.pipeline.steps.download import load_images
from app.schemas.process import ProcessAttachment


@pytest.fixture # 반복되는 준비를 한 번에 처리
def photo_dir(tmp_path):
    """정상, 흑백, 깨진 사진을 담은 임시 폴더"""
    Image.new("RGB", (8, 8), "red").save(tmp_path / "ok.jpg", "JPEG") # 정상 사진
    Image.new("RGB", (16, 8), "blue").save(tmp_path / "ok2.jpg", "JPEG") # 정상 사진2
    Image.new("L", (8, 8), 128).save(tmp_path / "gray.jpg", "JPEG") # 흑백 사진
    (tmp_path / "broken.jpg").write_bytes(b"not a jpeg") # 깨진 사진
    (tmp_path / "broken2.jpg").write_bytes(b"not a jpeg") # 깨진 사진2
    return tmp_path

def attachment(attachment_id: int, key: str) -> ProcessAttachment:
    return ProcessAttachment(trip_attachment_id=attachment_id, analyze_storage_key=key)

def test_decodes_to_rgb(photo_dir):
    src = LocalDirImageSource(photo_dir)
    got = load_images([attachment(101, "ok.jpg")], src)

    assert list(got) == [101]
    assert got[101].mode == "RGB"
    assert got[101].size == (8, 8) # 정상 디코딩 확인

def test_maps_each_id_to_own_image(photo_dir):
    src = LocalDirImageSource(photo_dir)
    got = load_images([attachment(101, "ok.jpg"), attachment(102, "ok2.jpg")], src)

    assert sorted(got) == [101, 102]
    assert got[101].size == (8, 8)
    assert got[102].size == (16, 8)

def test_converts_grayscale_to_rgb(photo_dir):
    src = LocalDirImageSource(photo_dir)
    assert Image.open(photo_dir / "gray.jpg").mode == "L"

    got = load_images([attachment(103, "gray.jpg")], src)

    assert got[103].mode == "RGB"

def test_collects_all_decode_failures(photo_dir):
    src = LocalDirImageSource(photo_dir)
    attachments = [
        attachment(101, "ok.jpg"),
        attachment(102, "broken.jpg"),
        attachment(103, "broken2.jpg"),
    ]

    with pytest.raises(DecodeFailed) as info:
        load_images(attachments, src)

    assert info.value.ids == [102, 103]
    assert info.value.detail == {"ids": [102, 103]}

def test_download_failure_passes_through(photo_dir):
    src = LocalDirImageSource(photo_dir)

    with pytest.raises(DownloadFailed) as info:
        load_images([attachment(101, "ok.jpg"), attachment(102, "missing.jpg")], src)

    assert info.value.keys == ["missing.jpg"]
    