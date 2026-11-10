"""
Фоновые задачи для ARQ-воркера.
"""

import asyncio
import logging
from typing import Any

from edu_doc_agent.api.router import release_session_lock

logger = logging.getLogger(__name__)


async def process_session_task(ctx: dict[str, Any], session_id: str, run_id: str) -> dict[str, str]:
    """
    Фоновая задача-заглушка для обработки документов.
    Эмулирует долгую работу графа.
    """

    logger.info(f"Начата обработка сессии. session_id={session_id}, run_id={run_id}")
    try:
        await asyncio.sleep(3)

        logger.info(f"Обработка сессии успешно завершена. session_id={session_id}, run_id={run_id}")

        return {
            "status": "completed",
            "session_id": session_id,
            "run_id": run_id,
        }
    finally:
        await release_session_lock(ctx.get("redis"), session_id, run_id)
