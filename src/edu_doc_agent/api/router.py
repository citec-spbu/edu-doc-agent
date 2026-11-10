"""
Роутер для управления сессиями и запуска обработки.
"""

import asyncio
import inspect
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

SESSION_LOCK_TTL_SECONDS = 900
ENQUEUE_DEADLINE_SECONDS = 2.0

RELEASE_SESSION_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


def get_session_lock_key(redis_pool: Any, session_id: str) -> str:
    """Формирует ключ блокировки активного запуска с учётом имени очереди."""
    queue_name = getattr(redis_pool, "default_queue_name", None)
    if not isinstance(queue_name, str) or not queue_name:
        queue_name = "arq:queue"
    return f"{queue_name}:session_lock:{session_id}"


async def acquire_session_lock(redis_pool: Any, lock_key: str, run_id: str) -> bool:
    """Атомарно резервирует активный запуск за сессией в Redis."""
    result = redis_pool.set(
        lock_key,
        run_id,
        nx=True,
        ex=SESSION_LOCK_TTL_SECONDS,
    )
    if inspect.isawaitable(result):
        result = await result
    return bool(result)


async def release_session_lock(redis_pool: Any, session_id: str, run_id: str) -> None:
    """Атомарно освобождает резерв сессии, если он принадлежит указанному run_id."""
    if redis_pool is None:
        return
    lock_key = get_session_lock_key(redis_pool, session_id)
    with suppress(Exception):
        result = redis_pool.eval(RELEASE_SESSION_LOCK_SCRIPT, 1, lock_key, run_id)
        if inspect.isawaitable(result):
            await result


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


async def reserve_and_enqueue(
    redis_pool: Any, session_id: str, lock_key: str, run_id: str
) -> Any | None:
    """
    Резервирует сессию и ставит задачу в очередь.

    Возвращает job или None, если сессия уже занята / задача уже существует.
    При ошибке постановки освобождает резерв. При отмене (дедлайн) резерв
    намеренно не освобождается. Резерв истечёт по TTL.
    """
    if not await acquire_session_lock(redis_pool, lock_key, run_id):
        return None
    try:
        job = await redis_pool.enqueue_job(
            "process_session_task",
            session_id=session_id,
            run_id=run_id,
            _job_id=run_id,
        )
    except Exception:
        await release_session_lock(redis_pool, session_id, run_id)
        raise
    if job is None:
        await release_session_lock(redis_pool, session_id, run_id)
    return job


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
    run_id_str = str(run_id)
    lock_key = get_session_lock_key(redis_pool, session_id)

    # Отправляем задачу в очередь Redis
    try:
        job = await run_with_deadline(
            reserve_and_enqueue(redis_pool, session_id, lock_key, run_id_str),
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
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Задача для данной сессии уже находится в очереди.",
        )

    # Возвращаем ответ клиенту, не дожидаясь завершения задачи
    return ProcessResponse(run_id=run_id, session_id=session_id, status="queued")
