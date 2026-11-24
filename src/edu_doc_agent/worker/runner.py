"""Инфраструктурная обёртка (Runner) для запуска бизнес-логики/графа с фиксацией статусов в БД."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from typing import Any
from uuid import UUID

from edu_doc_agent.dao import SessionRunDAO

logger = logging.getLogger(__name__)

# Бизнес-функция принимает только session_id (thread_id) и не знает про run_id и ARQ.
BusinessCallable = Callable[[str], Awaitable[Any]]


async def _finish_status_update[T](operation: Coroutine[Any, Any, T]) -> T:
    """Дожидается операции БД даже при повторной отмене, затем распространяет отмену."""
    task = asyncio.create_task(operation)
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc
        except Exception as exc:
            if cancellation is None:
                raise
            logger.exception("Status update failed while worker cancellation was pending")
            raise cancellation from exc
        else:
            if cancellation is not None:
                raise cancellation
            return result


async def _stop_heartbeat(task: asyncio.Task[None] | None) -> None:
    """Отменяет heartbeat_task и дожидается её завершения даже при повторном CancelledError."""
    if task is None:
        return
    task.cancel()
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.done():
                break
            cancellation = exc
    with suppress(asyncio.CancelledError):
        task.result()
    if cancellation is not None:
        raise cancellation


class SessionRunner:
    """Управляет жизненным циклом Run вокруг вызова бизнес-логики."""

    def __init__(self, dao: SessionRunDAO, heartbeat_interval: float | None = 30.0) -> None:
        self._dao = dao
        self._heartbeat_interval = heartbeat_interval

    async def _heartbeat_loop(
        self,
        run_id: UUID,
        interval: float,
        expected_reconcile_attempts: int | None,
    ) -> None:
        """Периодически обновляет last_heartbeat_at, пока задача выполняется."""
        while True:
            await asyncio.sleep(interval)
            try:
                updated = await self._dao.touch_heartbeat(
                    run_id,
                    expected_reconcile_attempts=expected_reconcile_attempts,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Не удалось обновить heartbeat для run_id=%s",
                    run_id,
                    exc_info=True,
                )
                continue

            if not updated:
                logger.warning(
                    "Остановка heartbeat: lease потерян или run_id=%s больше не в 'running'",
                    run_id,
                )
                return

    async def execute(
        self,
        session_id: str,
        run_id: UUID | str,
        logic_callable: BusinessCallable,
        job_try: int = 1,
        max_tries: int = 3,
        execution_timeout: float | None = None,
        job_deadline: float | None = None,
    ) -> dict[str, str]:
        """Атомарно переводит Run в 'running', вызывает бизнес-логику и фиксирует исход."""
        run_uuid = UUID(str(run_id)) if not isinstance(run_id, UUID) else run_id

        lease_token: int | None = None
        is_last_try = job_try >= max_tries

        async def claim_run() -> None:
            nonlocal lease_token
            lease_token = await self._dao.claim_run_for_execution(run_uuid)

        try:
            await _finish_status_update(claim_run())
            if lease_token is None:
                logger.warning(
                    "Пропуск выполнения: run_id=%s (session_id=%s) не находится в статусе 'queued'",
                    run_uuid,
                    session_id,
                )
                return {
                    "status": "skipped",
                    "session_id": session_id,
                    "run_id": str(run_uuid),
                }

            logger.info(
                "Запущено выполнение сессии: session_id=%s, run_id=%s (попытка %d/%d)",
                session_id,
                run_uuid,
                job_try,
                max_tries,
            )

            heartbeat_task: asyncio.Task[None] | None = None
            if self._heartbeat_interval is not None and self._heartbeat_interval > 0:
                heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(run_uuid, self._heartbeat_interval, lease_token)
                )

            try:
                try:
                    # По умолчанию длительность job ограничивает только ARQ.
                    async with asyncio.timeout(execution_timeout):
                        await logic_callable(session_id)
                finally:
                    await _stop_heartbeat(heartbeat_task)
            except Exception:
                try:
                    await _finish_status_update(
                        self._dao.mark_run_failed(
                            run_uuid,
                            expected_reconcile_attempts=lease_token,
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed to mark run as failed after business logic error. "
                        "session_id=%s, run_id=%s",
                        session_id,
                        run_uuid,
                    )
                logger.exception(
                    "Ошибка при выполнении сессии. session_id=%s, run_id=%s",
                    session_id,
                    run_uuid,
                )
                raise

            completed = await _finish_status_update(
                self._dao.mark_run_completed(
                    run_uuid,
                    expected_reconcile_attempts=lease_token,
                )
            )
            if not completed:
                logger.warning(
                    "mark_run_completed returned False "
                    "(run was not in 'running' state or lease lost). "
                    "session_id=%s, run_id=%s",
                    session_id,
                    run_uuid,
                )

        except asyncio.CancelledError:
            if lease_token is None:
                raise
            deadline_expired = (
                job_deadline is not None and asyncio.get_running_loop().time() >= job_deadline
            )
            try:
                if is_last_try or deadline_expired:
                    await _finish_status_update(
                        self._dao.mark_run_failed(
                            run_uuid,
                            expected_reconcile_attempts=lease_token,
                        )
                    )
                else:
                    await _finish_status_update(
                        self._dao.requeue_running_run(
                            run_uuid,
                            expected_reconcile_attempts=lease_token,
                        )
                    )
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "Failed to update run status during CancelledError handling. "
                    "session_id=%s, run_id=%s",
                    session_id,
                    run_uuid,
                )
            raise

        logger.info("Выполнение сессии завершено: session_id=%s, run_id=%s", session_id, run_uuid)
        return {
            "status": "completed",
            "session_id": session_id,
            "run_id": str(run_uuid),
        }
