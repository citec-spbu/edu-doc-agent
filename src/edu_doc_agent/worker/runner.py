"""Инфраструктурная обёртка (Runner) для запуска бизнес-логики/графа с фиксацией статусов в БД."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from edu_doc_agent.dao import SessionRunDAO

logger = logging.getLogger(__name__)

# Бизнес-функция принимает только session_id (thread_id) и не знает про run_id и ARQ.
BusinessCallable = Callable[[str], Awaitable[Any]]


class SessionRunner:
    """Управляет жизненным циклом Run вокруг вызова бизнес-логики"""

    def __init__(self, dao: SessionRunDAO) -> None:
        self._dao = dao

    async def execute(
        self,
        session_id: str,
        run_id: UUID | str,
        logic_callable: BusinessCallable,
    ) -> dict[str, str]:
        """Атомарно переводит Run в 'running', вызывает бизнес-логику и фиксирует исход.

        - Если Run уже не в статусе 'queued' (например, скомпенсирован API в 'failed' или уже
          завершён), бизнес-функция не вызывается.
        - При необработанном исключении фиксирует 'failed' в БД и пробрасывает ошибку в ARQ.
        - При штатной отмене корутины (asyncio.CancelledError при остановке воркера) возвращает
          статус 'running' -> 'queued', чтобы ARQ мог повторно выполнить тот же run_id.
        """
        run_uuid = UUID(str(run_id)) if not isinstance(run_id, UUID) else run_id

        claimed = await self._dao.claim_run_for_execution(run_uuid)
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

        try:
            await logic_callable(session_id)
        except asyncio.CancelledError:
            # Защищаем возврат статуса в 'queued' от повторной отмены, чтобы ARQ мог сделать retry
            await asyncio.shield(self._dao.requeue_running_run(run_uuid))
            logger.info(
                "Выполнение прервано (CancelledError), run возвращён в 'queued'. "
                "session_id=%s, run_id=%s",
                session_id,
                run_uuid,
            )
            raise
        except Exception:
            await self._dao.mark_run_failed(run_uuid)
            logger.exception(
                "Ошибка при выполнении сессии. session_id=%s, run_id=%s переведён в 'failed'",
                session_id,
                run_uuid,
            )
            raise

        await self._dao.mark_run_completed(run_uuid)
        return {
            "status": "completed",
            "session_id": session_id,
            "run_id": str(run_uuid),
        }
