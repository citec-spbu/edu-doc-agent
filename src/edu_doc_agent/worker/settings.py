"""
Конфигурация среды выполнения для ARQ-воркера.

Этот файл является точкой входа для CLI-утилиты arq. Здесь регистрируются
все фоновые задачи, настраивается подключение к Redis и определяются
глобальные параметры выполнения.
"""

import logging
from typing import Any

from arq.connections import RedisSettings

from edu_doc_agent.config import get_settings
from edu_doc_agent.worker.tasks import process_session_task

logger = logging.getLogger(__name__)
settings = get_settings()


async def startup(ctx: dict[str, Any]) -> None:
    """Хук, который выполняется один раз при старте процесса воркера."""

    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.info("Worker is starting up. Connected to Redis.")


async def shutdown(ctx: dict[str, Any]) -> None:
    """Хук, который выполняется при корректном завершении процесса воркера."""
    logger.info("Worker is shutting down...")


class WorkerSettings:
    """
    Класс конфигурации, который ищет библиотека arq при запуске.
    """

    redis_settings: RedisSettings = settings.arq_redis_settings

    # Функции, которые разрешено выполнять
    functions = [process_session_task]

    on_startup = startup
    on_shutdown = shutdown

    # Глобальные архитектурные ограничения
    job_timeout = 600  # Максимальное время выполнения одной задачи в секундах (10 минут)
    max_tries = 3  # Максимальное количество ретраев при сбоях
