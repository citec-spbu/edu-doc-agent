"""Инфраструктурная обёртка (Runner) для запуска бизнес-логики/графа с фиксацией статусов в БД."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
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


class SessionRunner:
    """Управляет жизненным циклом Run вокруг вызова бизнес-логики.

    Известные ограничения:
    - Если бизнес-логика завершилась успешно, но последующий вызов `mark_run_completed`
      упал по ошибке БД, статус в таблице runs не обновится (останется в 'running'),
      а задача завершится с исключением БД. В этом случае корректное состояние
      восстанавливается силами фонового Reconciler'а.
    """

    def __init__(self, dao: SessionRunDAO) -> None:
        self._dao = dao

    async def execute(
        self,
        session_id: str,
        run_id: UUID | str,
        logic_callable: BusinessCallable,
        job_try: int = 1,
        max_tries: int = 3,
        execution_timeout: float | None = None,
    ) -> dict[str, str]:
        """Атомарно переводит Run в 'running', вызывает бизнес-логику и фиксирует исход."""
        run_uuid = UUID(str(run_id)) if not isinstance(run_id, UUID) else run_id

        claimed = False
        is_last_try = job_try >= max_tries

        async def claim_run() -> None:
            nonlocal claimed
            # Флаг присваивается внутри защищённой операции: отмена после commit
            # не должна потерять сведения о том, кто захватил run.
            claimed = await self._dao.claim_run_for_execution(run_uuid)

        try:
            await _finish_status_update(claim_run())
            if not claimed:
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

            try:
                # По умолчанию длительность job ограничивает только ARQ.
                async with asyncio.timeout(execution_timeout):
                    await logic_callable(session_id)
            except Exception:
                try:
                    await _finish_status_update(self._dao.mark_run_failed(run_uuid))
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

            completed = await _finish_status_update(self._dao.mark_run_completed(run_uuid))
            if not completed:
                logger.warning(
                    "mark_run_completed returned False "
                    "(run was not in 'running' state or not found). "
                    "session_id=%s, run_id=%s",
                    session_id,
                    run_uuid,
                )

        except asyncio.CancelledError:
            if not claimed:
                # Не возвращаем в очередь run, который выполняет другой worker.
                raise
            try:
                if is_last_try:
                    await _finish_status_update(self._dao.mark_run_failed(run_uuid))
                else:
                    await _finish_status_update(self._dao.requeue_running_run(run_uuid))
            except asyncio.CancelledError:
                # Повторная отмена распространяется только после завершения записи.
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
