"""
Фоновые задачи для ARQ-воркера.
"""

import asyncio
import logging
from typing import Any

from edu_doc_agent.worker.reconciler import RunReconciler
from edu_doc_agent.worker.runner import SessionRunner

logger = logging.getLogger(__name__)


async def _stub_session_logic(session_id: str) -> None:
    """Временная заглушка, эмулирующая работу графа LangGraph."""
    logger.info("Запущена обработка бизнес-логики для session_id=%s", session_id)
    await asyncio.sleep(3)
    logger.info("Бизнес-логика успешно завершена для session_id=%s", session_id)


async def process_session_task(ctx: dict[str, Any], session_id: str, run_id: str) -> dict[str, str]:
    """Тонкая точка входа ARQ: делегирует управление статусами в SessionRunner."""

    job_try = ctx.get("job_try", 1)
    max_tries = ctx.get("max_tries", 3)

    runner = SessionRunner(
        dao=ctx["dao"],
        heartbeat_interval=ctx.get("heartbeat_interval", 30.0),
    )
    return await runner.execute(
        session_id=session_id,
        run_id=run_id,
        job_try=job_try,
        max_tries=max_tries,
        logic_callable=_stub_session_logic,
        job_deadline=ctx.get("job_deadline"),
    )


async def reconcile_stale_runs_task(ctx: dict[str, Any]) -> dict[str, int]:
    """Периодическая задача ARQ: восстанавливает зависшие запуски и завершает poison pill."""
    reconciler = RunReconciler(
        dao=ctx["dao"],
        redis=ctx["redis"],
        stale_running_seconds=ctx.get("stale_running_seconds", 120.0),
        stale_queued_seconds=ctx.get("stale_queued_seconds", 120.0),
        max_reconcile_attempts=ctx.get("max_reconcile_attempts", 3),
    )
    report = await reconciler.reconcile()
    return {
        "failed_count": report.failed_count,
        "requeued_count": report.requeued_count,
        "enqueued_count": report.enqueued_count,
    }
