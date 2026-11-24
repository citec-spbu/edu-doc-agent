"""
Конфигурация среды выполнения для ARQ-воркера.

Этот файл является точкой входа для CLI-утилиты arq. Здесь регистрируются
все фоновые задачи, настраивается подключение к Redis и определяются
глобальные параметры выполнения.
"""

import asyncio
import logging
from typing import Any

from arq import cron
from arq.connections import RedisSettings

from edu_doc_agent.config import get_settings
from edu_doc_agent.dao import LocalFileStorage, SessionRunDAO
from edu_doc_agent.worker.tasks import process_session_task, reconcile_stale_runs_task

JOB_TIMEOUT = 600
MAX_TRIES = 3

logger = logging.getLogger(__name__)
settings = get_settings()


async def startup(ctx: dict[str, Any]) -> None:
    """Хук, который выполняется один раз при старте процесса воркера."""
    current_settings = get_settings()

    logging.basicConfig(
        level=current_settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    ctx["dao"] = SessionRunDAO()
    ctx["storage"] = LocalFileStorage(current_settings.storage_dir)
    ctx["max_tries"] = MAX_TRIES
    ctx["heartbeat_interval"] = settings.run_heartbeat_interval_seconds
    ctx["stale_running_seconds"] = settings.run_stale_running_seconds
    ctx["stale_queued_seconds"] = settings.run_stale_queued_seconds
    ctx["max_reconcile_attempts"] = settings.run_max_reconcile_attempts
    logger.info("Worker is starting up. Connected to Redis.")


async def shutdown(ctx: dict[str, Any]) -> None:
    """Хук, который выполняется при корректном завершении процесса воркера."""
    logger.info("Worker is shutting down...")


async def job_start(ctx: dict[str, Any]) -> None:
    """Запоминает срок job до запуска ARQ wait_for, не создавая отдельного таймера."""
    ctx["job_deadline"] = asyncio.get_running_loop().time() + WorkerSettings.job_timeout


class WorkerSettings:
    """
    Класс конфигурации, который ищет библиотека arq при запуске.
    """

    redis_settings: RedisSettings = settings.arq_redis_settings

    # Функции, которые разрешено выполнять
    functions = [process_session_task]

    on_startup = startup
    on_shutdown = shutdown
    on_job_start = job_start

    # Глобальные архитектурные ограничения
    job_timeout = JOB_TIMEOUT
    max_tries = MAX_TRIES

    cron_jobs = [
        cron(reconcile_stale_runs_task, second=0, run_at_startup=True),
    ]
