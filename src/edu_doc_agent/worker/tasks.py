"""
Фоновые задачи для ARQ-воркера.
"""

import asyncio
import logging
from typing import Any

from edu_doc_agent.dao import SessionRunDAO
from edu_doc_agent.worker.runner import SessionRunner

logger = logging.getLogger(__name__)


async def _stub_session_logic(session_id: str) -> None:
    """Временная заглушка, эмулирующая работу графа LangGraph."""
    logger.info("Запущена обработка бизнес-логики для session_id=%s", session_id)
    await asyncio.sleep(3)
    logger.info("Бизнес-логика успешно завершена для session_id=%s", session_id)


async def process_session_task(ctx: dict[str, Any], session_id: str, run_id: str) -> dict[str, str]:
    dao = ctx.get("dao")
    if not isinstance(dao, SessionRunDAO):
        dao = SessionRunDAO()

    # Достаем метаданные попыток из ARQ контекста
    job_try = ctx.get("job_try", 1)
    max_tries = ctx.get("max_tries", 3)

    runner = SessionRunner(dao=dao)
    return await runner.execute(
        session_id=session_id,
        run_id=run_id,
        job_try=job_try,
        max_tries=max_tries,
        logic_callable=_stub_session_logic,
    )
