"""
Роутер для управления сессиями и запуска обработки.
"""

from typing import Annotated
from uuid import uuid4

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, Path, status

from edu_doc_agent.api.dependencies import get_arq_pool
from edu_doc_agent.api.schemas import ProcessResponse

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.post(
    "/{session_id}/process",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ProcessResponse,
)
async def process_session(
    session_id: Annotated[str, Path(..., description="ID сессии пользователя")],
    redis_pool: Annotated[ArqRedis, Depends(get_arq_pool)],
) -> ProcessResponse:
    """
    Запускает асинхронную проверку и обработку образовательной программы.
    Не блокирует HTTP-запрос, сразу возвращает статус 'queued'.
    """
    run_id = uuid4()

    # Отправляем задачу в очередь Redis
    await redis_pool.enqueue_job(
        "process_session_task",
        session_id=session_id,
        run_id=str(run_id),
        _job_id=str(run_id),
    )

    # Возвращаем ответ клиенту, не дожидаясь завершения задачи
    return ProcessResponse(run_id=run_id, session_id=session_id, status="queued")
