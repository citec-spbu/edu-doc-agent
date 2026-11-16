"""
Роутер для управления сессиями и запуска обработки.
"""

import asyncio
import logging
from collections.abc import Coroutine
from contextlib import suppress
from typing import Annotated, Any
from uuid import UUID, uuid4

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Path, status
from redis.exceptions import RedisError

from edu_doc_agent.api.dependencies import get_arq_pool, get_session_run_dao
from edu_doc_agent.api.schemas import ProcessResponse, RunStatusResponse
from edu_doc_agent.dao import ActiveRunConflictError, SessionRunDAO

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


async def _compensate_failed_enqueue(
    dao: SessionRunDAO,
    session_id: str,
    run_id: UUID,
) -> None:
    """Переводит Run в 'failed' после неподтверждённой постановки в Redis, если он ещё 'queued'"""
    try:
        await dao.mark_failed_if_queued(run_id)
    except Exception:
        logger.exception(
            "Failed to compensate queued run after enqueue error. session_id=%s, run_id=%s",
            session_id,
            run_id,
        )


@router.post(
    "/{session_id}/process",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ProcessResponse,
)
async def process_session(
    session_id: Annotated[str, Path(..., description="ID сессии пользователя")],
    redis_pool: Annotated[ArqRedis, Depends(get_arq_pool)],
    dao: Annotated[SessionRunDAO | None, Depends(get_session_run_dao)] = None,
) -> ProcessResponse:
    """
    Запускает асинхронную проверку и обработку образовательной программы.
    Фиксирует Session и Run ('queued') в PostgreSQL до постановки в очередь Redis.
    """
    if not isinstance(dao, SessionRunDAO):
        dao = SessionRunDAO()

    run_id = uuid4()
    run_id_str = str(run_id)

    try:
        await dao.create_queued_run(session_id=session_id, run_id=run_id)
    except ActiveRunConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Для данной сессии уже выполняется активный запуск.",
        ) from e

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

    except asyncio.CancelledError:
        await asyncio.shield(_compensate_failed_enqueue(dao, session_id, run_id))
        raise

    except TimeoutError as e:
        await _compensate_failed_enqueue(dao, session_id, run_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Очередь задач временно недоступна. Повторите попытку позже.",
        ) from e

    except RedisError as e:
        logger.error("Redis connection error for session %s: %s", session_id, e)
        await _compensate_failed_enqueue(dao, session_id, run_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис очередей недоступен. Повторите попытку позже.",
        ) from e

    except Exception as e:
        logger.error("Failed to enqueue job for session %s: %s", session_id, e)
        await _compensate_failed_enqueue(dao, session_id, run_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка при постановке задачи в очередь.",
        ) from e

    if job is None:
        # ARQ возвращает None и при WatchError после обрыва соединения.
        # Подтверждения нет; переводим только ещё не взятый воркером run ('queued') в 'failed'.
        await _compensate_failed_enqueue(dao, session_id, run_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Постановка задания не подтверждена. Результат записи неизвестен.",
        )

    # Возвращаем ответ клиенту, не дожидаясь завершения задачи
    return ProcessResponse(run_id=run_id, session_id=session_id, status="queued")


@router.get(
    "/{session_id}/runs/{run_id}",
    status_code=status.HTTP_200_OK,
    response_model=RunStatusResponse,
)
async def get_run_status(
    session_id: Annotated[str, Path(..., description="ID сессии пользователя")],
    run_id: Annotated[UUID, Path(..., description="ID запуска (run_id)")],
    dao: Annotated[SessionRunDAO | None, Depends(get_session_run_dao)] = None,
) -> RunStatusResponse:
    """
    Возвращает канонический статус запуска и временные метки из PostgreSQL.
    Не обращается к Redis.
    """
    if not isinstance(dao, SessionRunDAO):
        dao = SessionRunDAO()

    run = await dao.get_run(session_id=session_id, run_id=run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Запуск не найден в рамках указанной сессии.",
        )

    return RunStatusResponse(
        run_id=run.id,
        session_id=run.session_id,
        status=run.status,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )
