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
    """Тонкая точка входа ARQ: достаёт DAO из контекста и делегирует управление в SessionRunner."""
    dao = ctx.get("dao")
    if not isinstance(dao, SessionRunDAO):
        dao = SessionRunDAO()

    runner = SessionRunner(dao=dao)
    return await runner.execute(
        session_id=session_id,
        run_id=run_id,
        logic_callable=_stub_session_logic,
    )
