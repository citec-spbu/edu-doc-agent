"""
Роутер для управления сессиями и запуска обработки.
"""

import asyncio
import logging
from collections.abc import Coroutine
from contextlib import suppress
from typing import Annotated, Any
from uuid import uuid4

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Path, status
from redis.exceptions import RedisError

from edu_doc_agent.api.dependencies import get_arq_pool
from edu_doc_agent.api.schemas import ProcessResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])

ENQUEUE_DEADLINE_SECONDS = 2.0


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    """Забирает исключение брошенной задачи, чтобы не было 'exception never retrieved'."""
    with suppress(asyncio.CancelledError, Exception):
        task.exception()


async def run_with_deadline[T](coro: Coroutine[Any, Any, T], timeout: float) -> T:
    """
    Ждёт результат не дольше timeout секунд.

    В отличие от asyncio.timeout, не дожидается завершения отменённой задачи:
    очистка соединения (например, Pipeline.reset() с UNWATCH) может зависнуть
    вместе с Redis и продлить HTTP-запрос сверх дедлайна. Отменённая задача
    дочищается в фоне.
    """
    task = asyncio.create_task(coro)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        # Клиент отключился: тоже не ждём очистку.
        task.cancel()
        task.add_done_callback(_consume_task_result)
        raise
    if task in done:
        return task.result()
    task.cancel()
    task.add_done_callback(_consume_task_result)
    raise TimeoutError


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
    Возвращает статус 'queued' после подтверждения постановки, не ожидая worker.
    """
    run_id = uuid4()
    run_id_str = str(run_id)

    # Отправляем задачу в очередь Redis
    try:
        job = await run_with_deadline(
            redis_pool.enqueue_job(
                "process_session_task",
                session_id=session_id,
                run_id=run_id_str,
                _job_id=run_id_str,
            ),
            ENQUEUE_DEADLINE_SECONDS,
        )

    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Очередь задач временно недоступна. Повторите попытку позже.",
        ) from e

    except RedisError as e:
        logger.error("Redis connection error for session %s: %s", session_id, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис очередей недоступен. Повторите попытку позже.",
        ) from e

    except Exception as e:
        logger.error("Failed to enqueue job for session %s: %s", session_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка при постановке задачи в очередь.",
        ) from e

    if job is None:
        # ARQ возвращает None и при WatchError после обрыва соединения.
        # Подтверждения нет; по этому значению нельзя установить исход записи.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Постановка задания не подтверждена. Результат записи неизвестен.",
        )

    # Возвращаем ответ клиенту, не дожидаясь завершения задачи
    return ProcessResponse(run_id=run_id, session_id=session_id, status="queued")
