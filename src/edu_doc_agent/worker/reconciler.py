"""Фоновый Reconciler для восстановления зависших запусков (runs) и защиты от poison pill."""

import asyncio
import logging
from dataclasses import dataclass

from arq.connections import ArqRedis
from arq.constants import default_queue_name
from redis.exceptions import RedisError

from edu_doc_agent.dao import ReconciledRun, SessionRunDAO

logger = logging.getLogger(__name__)

REDIS_OP_TIMEOUT_SECONDS = 5.0
DEFAULT_BATCH_LIMIT = 100


@dataclass(frozen=True)
class ReconcileReport:
    """Итоги одного прохода Reconciler'а."""

    failed_count: int
    requeued_count: int
    enqueued_count: int


def _current_job_id(run: ReconciledRun) -> str:
    """Текущий ARQ job_id для запуска с учётом числа восстановлений."""
    if run.reconcile_attempts == 0:
        return str(run.run_id)
    return f"{run.run_id}:reconcile:{run.reconcile_attempts}"


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
        queue_name: str = default_queue_name,
        batch_limit: int = DEFAULT_BATCH_LIMIT,
    ) -> None:
        self._dao = dao
        self._redis = redis
        self._stale_running_seconds = stale_running_seconds
        self._stale_queued_seconds = stale_queued_seconds
        self._max_reconcile_attempts = max_reconcile_attempts
        self._queue_name = queue_name
        self._batch_limit = batch_limit

    async def _rollback_batch(self, runs: list[ReconciledRun]) -> None:
        """Откатывает инкремент reconcile_attempts для непоставленных в Redis задач."""
        for run in runs:
            try:
                await self._dao.rollback_reconcile_attempt(
                    run.run_id,
                    expected_reconcile_attempts=run.reconcile_attempts,
                )
            except Exception:
                logger.exception(
                    "Не удалось откатить reconcile_attempts для run_id=%s",
                    run.run_id,
                )

    async def reconcile(self) -> ReconcileReport:
        """Выполняет один проход реконсиляции."""
        try:
            async with asyncio.timeout(REDIS_OP_TIMEOUT_SECONDS):
                await self._redis.ping()
        except (RedisError, TimeoutError):
            logger.warning("Пропуск реконсиляции: Redis недоступен")
            return ReconcileReport(failed_count=0, requeued_count=0, enqueued_count=0)

        failed_ids = await self._dao.fail_exhausted_stale_running_runs(
            stale_running_seconds=self._stale_running_seconds,
            max_reconcile_attempts=self._max_reconcile_attempts,
            limit=self._batch_limit,
        )
        for run_id in failed_ids:
            logger.error(
                "Run переведён в 'failed' (превышен лимит восстановлений %d): run_id=%s",
                self._max_reconcile_attempts,
                run_id,
            )

        requeued_runs = await self._dao.requeue_stale_running_runs(
            stale_running_seconds=self._stale_running_seconds,
            max_reconcile_attempts=self._max_reconcile_attempts,
            limit=self._batch_limit,
        )

        # Проверяем зависшие 'queued': если джоба ещё в очереди Redis — не трогаем.
        queued_candidates = await self._dao.list_stale_queued_runs(
            stale_queued_seconds=self._stale_queued_seconds,
            limit=self._batch_limit,
        )
        for candidate in queued_candidates:
            job_id = _current_job_id(candidate)
            try:
                async with asyncio.timeout(REDIS_OP_TIMEOUT_SECONDS):
                    in_queue = await self._redis.zscore(self._queue_name, job_id)
            except (RedisError, TimeoutError):
                logger.warning("Прерывание проверки queued-задач: Redis недоступен")
                break

            if in_queue is not None:
                continue

            if candidate.reconcile_attempts >= self._max_reconcile_attempts:
                if await self._dao.fail_stale_queued_run(
                    candidate.run_id,
                    expected_reconcile_attempts=candidate.reconcile_attempts,
                ):
                    failed_ids.append(candidate.run_id)
                    logger.error(
                        "Потерянный queued run переведён в 'failed' (лимит %d): run_id=%s",
                        self._max_reconcile_attempts,
                        candidate.run_id,
                    )
            else:
                bumped = await self._dao.bump_stale_queued_run(
                    candidate.run_id,
                    expected_reconcile_attempts=candidate.reconcile_attempts,
                )
                if bumped is not None:
                    requeued_runs.append(bumped)

        enqueued_count = 0
        for idx, run in enumerate(requeued_runs):
            run_id_str = str(run.run_id)
            job_id = f"{run_id_str}:reconcile:{run.reconcile_attempts}"
            try:
                async with asyncio.timeout(REDIS_OP_TIMEOUT_SECONDS):
                    job = await self._redis.enqueue_job(
                        "process_session_task",
                        session_id=run.session_id,
                        run_id=run_id_str,
                        _job_id=job_id,
                        _queue_name=self._queue_name,
                    )
            except (RedisError, TimeoutError):
                logger.exception(
                    "Redis недоступен при постановке run, откатываем оставшиеся попытки: run_id=%s",
                    run_id_str,
                )
                await self._rollback_batch(requeued_runs[idx:])
                break
            except Exception:
                logger.exception(
                    "Не удалось повторно поставить run в очередь ARQ: session_id=%s, run_id=%s",
                    run.session_id,
                    run_id_str,
                )
                await self._rollback_batch([run])
                continue

            if job is None:
                try:
                    async with asyncio.timeout(REDIS_OP_TIMEOUT_SECONDS):
                        in_queue = await self._redis.zscore(self._queue_name, job_id)
                except (RedisError, TimeoutError):
                    in_queue = None

                if in_queue is not None:
                    enqueued_count += 1
                    logger.info(
                        "Run уже присутствует в очереди ARQ: session_id=%s, run_id=%s",
                        run.session_id,
                        run_id_str,
                    )
                else:
                    logger.warning(
                        "ARQ не подтвердил повторную постановку run "
                        "(возможен конфликт job_id): session_id=%s, run_id=%s",
                        run.session_id,
                        run_id_str,
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
