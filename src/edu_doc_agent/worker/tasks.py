"""
Фоновые задачи для ARQ-воркера.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def process_session_task(ctx: dict[str, Any], session_id: str, run_id: str) -> dict[str, str]:
    """
    Фоновая задача-заглушка для обработки документов.
    Эмулирует долгую работу графа.
    """

    logger.info(f"Начата обработка сессии. session_id={session_id}, run_id={run_id}")
    await asyncio.sleep(3)

    logger.info(f"Обработка сессии успешно завершена. session_id={session_id}, run_id={run_id}")

    # Терминальный исход фиксирует ARQ после возврата из задачи. Отмена этой
    # попытки (например, SIGTERM) может означать retry, а не завершение run.
    return {
        "status": "completed",
        "session_id": session_id,
        "run_id": run_id,
    }
