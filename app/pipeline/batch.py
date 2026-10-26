"""업로드 묶음 임베딩. photos_ready 묶음의 사본을 정리 전에 임베딩해 Qdrant에 저장

정리 EMBEDDING이 ctx.reuse_vectors로 저장 벡터를 재사용. 결과의 기준은 정리이므로 여기서 빠진
사진은 정리가 다시 임베딩하거나 DecodeFailed로 처리
api와 jobs를 import하지 않음
"""

import io
from collections.abc import Mapping

from PIL import Image

from app.pipeline.context import PipelineContext
from app.pipeline.steps.embed import encode_checked


def embed_batch(photos: Mapping[int, str], ctx: PipelineContext) -> list[int]:
    """{trip_attachment_id: analyze_storage_key} 묶음을 임베딩해 저장하고 저장한 ID를 반환

    다운로드 실패는 ctx.images의 DownloadFailed를 그대로 올림. 디코딩 실패 사진은 로그만 남기고
    제외, 판정은 정리 DOWNLOADING의 DecodeFailed가 담당. 엔진 출력 계약 위반은 PipelineError
    """
    ids = list(photos)
    blobs = ctx.images.get_many([photos[photo_id] for photo_id in ids])
    decodable: list[int] = []
    failed: list[int] = []
    for photo_id in ids:
        # 정리의 load_images와 같은 기준으로 열리는지만 확인하고 이미지는 바로 해제
        try:
            with Image.open(io.BytesIO(blobs[photos[photo_id]])) as img:
                img.convert("RGB").close()
        except (OSError, ValueError):
            failed.append(photo_id)
        else:
            decodable.append(photo_id)
    if failed:
        ctx.log.warning(
            "업로드 묶음의 디코딩 실패 사진 제외",
            extra={
                "event": "ai_photo_batch_decode_failed",
                "failed_count": len(failed),
                "failed_ids": failed,
            },
        )
    if decodable:
        vectors = encode_checked(ctx.engine, [blobs[photos[photo_id]] for photo_id in decodable])
        ctx.qdrant.upsert(decodable, vectors)
    return decodable
