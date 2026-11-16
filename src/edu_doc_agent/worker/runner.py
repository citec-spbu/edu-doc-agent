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
    def __init__(self, dao: SessionRunDAO) -> None:
        self._dao = dao

    async def execute(
        self,
        session_id: str,
        run_id: UUID | str,
        logic_callable: BusinessCallable,
        job_try: int = 1,
        max_tries: int = 3,
        execution_timeout: float = 570.0,  # Чуть меньше стандартного job_timeout (600с)
    ) -> dict[str, str]:
        run_uuid = UUID(str(run_id)) if not isinstance(run_id, UUID) else run_id

        claimed = await self._dao.claim_run_for_execution(run_uuid)
        if not claimed:
            logger.warning("Пропуск выполнения: run_id=%s не в статусе 'queued'", run_uuid)
            return {"status": "skipped", "session_id": session_id, "run_id": str(run_uuid)}

        logger.info(
            "Запущено выполнение сессии: session_id=%s, run_id=%s (попытка %d/%d)",
            session_id,
            run_uuid,
            job_try,
            max_tries,
        )

        is_last_try = job_try >= max_tries

        try:
            # Защита внутренним таймаутом, чтобы отличить таймаут от SIGTERM воркера
            async with asyncio.timeout(execution_timeout):
                await logic_callable(session_id)

        except TimeoutError:
            # Наш собственный таймаут — это всегда фатальный сбой для текущего рана
            await self._dao.mark_run_failed(run_uuid)
            logger.error(
                "Превышен внутренний таймаут выполнения. "
                "session_id=%s, run_id=%s переведён в 'failed'",
                session_id,
                run_uuid,
            )
            raise

        except asyncio.CancelledError:
            # Внешняя отмена (например, graceful shutdown воркера SIGTERM)
            if is_last_try:
                await self._dao.mark_run_failed(run_uuid)
                logger.warning(
                    "Выполнение прервано на последней попытке (%d/%d). Run переведён в 'failed'",
                    job_try,
                    max_tries,
                )
            else:
                await asyncio.shield(self._dao.requeue_running_run(run_uuid))
                logger.info(
                    "Выполнение прервано, попыток осталось. "
                    "Run возвращён в 'queued'. session_id=%s",
                    session_id,
                )
            raise

        except Exception:
            await self._dao.mark_run_failed(run_uuid)
            logger.exception(
                "Ошибка при выполнении сессии. Переведён в 'failed'. session_id=%s", session_id
            )
            raise

        await self._dao.mark_run_completed(run_uuid)
        logger.info("Выполнение сессии завершено: session_id=%s, run_id=%s", session_id, run_uuid)
        return {
            "status": "completed",
            "session_id": session_id,
            "run_id": str(run_uuid),
        }
