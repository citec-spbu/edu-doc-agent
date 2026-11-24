"""Фоновый Reconciler для восстановления зависших запусков (runs) и защиты от poison pill."""

import logging
from dataclasses import dataclass

from arq.connections import ArqRedis

from edu_doc_agent.dao import SessionRunDAO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconcileReport:
    """Итоги одного прохода Reconciler'а."""

    failed_count: int
    requeued_count: int
    enqueued_count: int


class RunReconciler:
    """Находит зависшие запуски в PostgreSQL, переводит poison pill в failed
    и повторно ставит восстановимые запуски в очередь ARQ."""

    def __init__(
        self,
        *,
        dao: SessionRunDAO,
        redis: ArqRedis,
        stale_running_seconds: float = 120.0,
        stale_queued_seconds: float = 120.0,
        max_reconcile_attempts: int = 3,
    ) -> None:
        self._dao = dao
        self._redis = redis
        self._stale_running_seconds = stale_running_seconds
        self._stale_queued_seconds = stale_queued_seconds
        self._max_reconcile_attempts = max_reconcile_attempts

    async def reconcile(self) -> ReconcileReport:
        """Выполняет один проход реконсиляции."""
        failed_ids = await self._dao.fail_exhausted_stale_runs(
            stale_running_seconds=self._stale_running_seconds,
            stale_queued_seconds=self._stale_queued_seconds,
            max_reconcile_attempts=self._max_reconcile_attempts,
        )
        for run_id in failed_ids:
            logger.error(
                "Run переведён в 'failed' (превышен лимит восстановлений %d): run_id=%s",
                self._max_reconcile_attempts,
                run_id,
            )

        requeued_runs = await self._dao.requeue_stale_runs(
            stale_running_seconds=self._stale_running_seconds,
            stale_queued_seconds=self._stale_queued_seconds,
            max_reconcile_attempts=self._max_reconcile_attempts,
        )

        enqueued_count = 0
        for run in requeued_runs:
            run_id_str = str(run.run_id)
            job_id = f"{run_id_str}:reconcile:{run.reconcile_attempts}"
            try:
                job = await self._redis.enqueue_job(
                    "process_session_task",
                    session_id=run.session_id,
                    run_id=run_id_str,
                    _job_id=job_id,
                )
            except Exception:
                logger.exception(
                    "Не удалось повторно поставить run в очередь ARQ: session_id=%s, run_id=%s",
                    run.session_id,
                    run_id_str,
                )
                continue

            if job is None:
                logger.warning(
                    "ARQ не подтвердил повторную постановку run: "
                    "session_id=%s, run_id=%s, job_id=%s",
                    run.session_id,
                    run_id_str,
                    job_id,
                )
                continue

            enqueued_count += 1
            logger.info(
                "Run восстановлен и поставлен в очередь: session_id=%s, run_id=%s (попытка %d/%d)",
                run.session_id,
                run_id_str,
                run.reconcile_attempts,
                self._max_reconcile_attempts,
            )

        return ReconcileReport(
            failed_count=len(failed_ids),
            requeued_count=len(requeued_runs),
            enqueued_count=enqueued_count,
        )
