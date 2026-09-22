"""작업 상태 저장소, trip_id → Task

api 프로세스 메모리의 dict 하나. 접근 주체가 이벤트 루프 스레드 하나뿐이라 락 없음.
get, put, count, all 넷으로 격리해 두어 조건이 생기면 이 클래스만 Redis 구현으로 교체.
시각은 전부 time.monotonic() 값이며 경과 시간 비교(사망, 타임아웃, 유휴)에만 씀
"""

from dataclasses import dataclass

from app.schemas.process import ProcessRequest, ProcessResult
from app.schemas.task import ErrorBody, ProcessStep, Progress, TaskStatus, TaskStatusResponse


@dataclass
class Task:
    """여행 하나의 작업 기록. 필드를 바꾸는 코드는 task_service뿐"""

    trip_id: int
    request: ProcessRequest  # 대기열에서 꺼낼 때 워커에 넘길 본문
    total: int
    created_at: float
    status: TaskStatus = TaskStatus.QUEUED
    done: int = 0
    current_step: ProcessStep | None = None
    result: ProcessResult | None = None
    error: ErrorBody | None = None
    started_at: float | None = None  # PROCESSING이 된 시각, 타임아웃 기준
    last_callback: float | None = None  # 마지막 워커 콜백 시각, 사망 판정 기준
    finished_at: float | None = None

    def to_response(self) -> TaskStatusResponse:
        """POST와 GET 응답. COMPLETED면 done을 total로, 마지막 진행률 콜백이 total 미만이어도"""
        done = self.total if self.status is TaskStatus.COMPLETED else self.done
        return TaskStatusResponse(
            trip_id=self.trip_id,
            execution_id=self.request.execution_id,
            status=self.status,
            progress=Progress(done=done, total=self.total),
            current_step=self.current_step,
            result=self.result,
            error=self.error,
        )


class TaskStore:
    """trip_id로 찾는 메모리 저장소

    끝난 작업도 다음 POST가 덮어쓸 때까지 보관. 연결이 끊긴 백엔드가 GET으로 회수
    """

    def __init__(self, boot_time: float) -> None:
        self._tasks: dict[int, Task] = {}
        self.boot_time = boot_time
        self.last_activity: float | None = None  # 마지막 작업 종료 시각, 종료된 적 없으면 None

    def get(self, trip_id: int) -> Task | None:
        return self._tasks.get(trip_id)

    def put(self, task: Task) -> None:
        """같은 trip_id가 있으면 덮어씀. 완료 뒤 재POST가 새 시도가 되는 이유"""
        self._tasks[task.trip_id] = task

    def count(self, status: TaskStatus) -> int:
        return sum(1 for task in self._tasks.values() if task.status is status)

    def all(self) -> list[Task]:
        """순회 중 상태가 바뀌어도 안전하도록 복사본"""
        return list(self._tasks.values())
