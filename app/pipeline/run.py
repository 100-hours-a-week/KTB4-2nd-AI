"""정리 파이프라인 진입점, 인터페이스 1과 5

이 파일의 시그니처와 규칙은 api와 worker가 의존하는 계약. 본문과 steps/는 파이프라인 담당이 채움.
바꾸려면 상대를 리뷰어로 지정
"""

from collections.abc import Callable

from app.pipeline.context import PipelineContext
from app.schemas.process import ProcessRequest, ProcessResult
from app.schemas.task import ProcessStep

ProgressCallback = Callable[[ProcessStep, int, int], None]
CancelCheck = Callable[[], bool]


def run(
    request: ProcessRequest,
    ctx: PipelineContext,
    on_progress: ProgressCallback,
    is_cancelled: CancelCheck,
) -> ProcessResult:
    """사진 정리 파이프라인. 요청 하나를 받아 결과 하나를 반환

    인자
        request       이번 여행의 사진 목록, 기간, 지역.
                      사본 키는 request.attachments[].analyze_storage_key
        ctx           외부 자원. ctx.images.get_many(keys), ctx.engine.encode_images(images),
                      ctx.qdrant.upsert(ids, vectors), ctx.thresholds[키], ctx.settings, ctx.log
        on_progress   진행률 보고 함수 on_progress(step, done, total). 워커가 api로 콜백
        is_cancelled  취소 확인 함수. True면 즉시 PipelineCancelled를 raise

    단계, 이 순서로 ProcessStep 값을 on_progress에 넘김
        DOWNLOADING  ctx.images로 사본 bytes 확보, PIL로 디코딩. 안 열리는 사진은 DecodeFailed(ids)
        EMBEDDING    16장 배치로 ctx.engine.encode_images, 배치마다 ctx.qdrant.upsert
        CLOCK        device_model로 기기별 묶음, 유사도 쌍으로 기기별 단일 오프셋 산출과 적용
        LOCATING     기간 밖 분리, GPS 앵커, 유사도 상속
        CLUSTERING   haversine DBSCAN, 재방문은 한 폴더
        FILTERING    라플라시안 BLURRY, pHash DUPLICATED, 원래 폴더 place_id 연결
        FINALIZING   대표컷은 폴더 평균 임베딩 최근접, first/last_taken_at, ProcessResult 조립

    진행률 규칙
        total은 항상 사진 수(len(request.attachments)), 단계가 바뀌어도 같음
        DOWNLOADING과 EMBEDDING은 사진 묶음 또는 배치마다 done을 올림
        나머지 다섯 단계는 시작에 (step, 0, total), 끝에 (step, total, total) 한 번씩
        어떤 단계도 120초 넘게 on_progress 없이 조용하면 안 됨. api가 WORKER_DEAD로 판정.
        넘길 수 있는 단계는 중간에 done을 갱신해 호출

    취소 규칙
        단계 사이와 EMBEDDING 배치 사이에서 is_cancelled()를 확인
        True면 raise PipelineCancelled(). 결과를 반환하지 않음

    실패 규칙
        실패는 반환값이 아니라 예외로만. 러너가 예외의 code로 실패 콜백을 보냄
        DownloadFailed(keys)   ctx.images가 던짐, 그대로 통과시킴
        DecodeFailed(ids)      디코딩 실패한 trip_attachment_id 목록
        PipelineError(message) 그 외, INTERNAL_ERROR로 나감
        부분 실패 없음. 결과의 failed[]는 정상 경로에서 빈 배열

    결과 규칙
        ProcessResult의 검증기가 명세 규칙을 확인함. 시각은 오프셋 있는 datetime만,
        BLURRY와 DUPLICATED는 place_id 필수, DUPLICATED는 duplicate_of_attachment_id 필수,
        대표컷은 폴더 안 사진. 검증 실패는 INTERNAL_ERROR

    금지
        api와 worker를 import하지 않음. 서버 없이 scripts/run_pipeline.py로 실행 가능해야 함
        전역 상태를 두지 않음. 작업 사이에 남는 것은 ctx.qdrant의 벡터뿐
        임계값을 코드에 두지 않음, 전부 ctx.thresholds
    """
    raise NotImplementedError("파이프라인 담당이 채움")
