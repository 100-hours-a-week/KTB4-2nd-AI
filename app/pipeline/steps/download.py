import io

from PIL import Image

from app.core.errors import DecodeFailed
from app.infra.images import ImageSource
from app.schemas.process import ProcessAttachment


def load_images(
    attachments: list[ProcessAttachment],
    images: ImageSource,
) -> dict[int, Image.Image]:
    """백엔드 서버로부터 전달 받은 attachments를 통해
    이미지 id와 데이터를 key, value로 갖는 dict 반환하는 함수.
    
    디코딩에 실패한 이미지들의 id는 따로 모아 DecodedFailed로 처리."""
    keys = [a.analyze_storage_key for a in attachments]
    blobs = images.get_many(keys)

    failed_ids: list[int] = []
    result: dict[int, Image.Image] = {}
    for a in attachments:
        data = blobs[a.analyze_storage_key]
        try:
            img = Image.open(io.BytesIO(data))
            img = img.convert("RGB")
            result[a.trip_attachment_id] = img
        except OSError:
            failed_ids.append(a.trip_attachment_id)

    if failed_ids:
        raise DecodeFailed(failed_ids)

    return result