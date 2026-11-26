"""Фоновый Reconciler для восстановления зависших запусков (runs) и защиты от poison pill."""

import asyncio
import logging
from dataclasses import dataclass

from arq.connections import ArqRedis
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
        queue_name: str | None = None,
        batch_limit: int = DEFAULT_BATCH_LIMIT,
    ) -> None:
        self._dao = dao
        self._redis = redis
        self._stale_running_seconds = stale_running_seconds
        self._stale_queued_seconds = stale_queued_seconds
        self._max_reconcile_attempts = max_reconcile_attempts
        self._queue_name = redis.default_queue_name if queue_name is None else queue_name
        self._batch_limit = batch_limit

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

        reserved_runs = await self._dao.requeue_stale_running_runs(
            stale_running_seconds=self._stale_running_seconds,
            max_reconcile_attempts=self._max_reconcile_attempts,
            limit=self._batch_limit,
        )
        reserved_generations = {(run.run_id, run.reconcile_attempts) for run in reserved_runs}
        requeued_count = len(reserved_runs)
        enqueued_count = 0
        # Доставляем сразу после резервирования: обход queued-страницы может
        # занять весь срок cron. При отмене оставшиеся поколения сохраняют pending.
        for run in reserved_runs:
            try:
                enqueued_count += await self._enqueue_run(run)
            except (RedisError, TimeoutError):
                return ReconcileReport(len(failed_ids), requeued_count, enqueued_count)

        # Проверяем зависшие 'queued': если джоба ещё в очереди Redis — не трогаем.
        queued_candidates = await self._dao.list_stale_queued_runs(
            stale_queued_seconds=self._stale_queued_seconds,
            limit=self._batch_limit,
        )
        for candidate in queued_candidates:
            if (candidate.run_id, candidate.reconcile_attempts) in reserved_generations:
                continue
            job_id = _current_job_id(candidate)
            try:
                async with asyncio.timeout(REDIS_OP_TIMEOUT_SECONDS):
                    in_queue = await self._redis.zscore(self._queue_name, job_id)
            except (RedisError, TimeoutError):
                logger.warning("Прерывание проверки queued-задач: Redis недоступен")
                break

            if in_queue is not None:
                if candidate.reconcile_pending:
                    await self._dao.confirm_reconcile_enqueue(
                        candidate.run_id,
                        expected_reconcile_attempts=candidate.reconcile_attempts,
                    )
                await self._dao.mark_queued_run_checked(
                    candidate.run_id,
                    expected_reconcile_attempts=candidate.reconcile_attempts,
                )
                continue

            if candidate.reconcile_pending:
                # Cron мог завершиться после commit резервирования, но до enqueue.
                # Повторяем доставку того же поколения, не списывая новую попытку.
                run = candidate
            elif candidate.reconcile_attempts >= self._max_reconcile_attempts:
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
                continue
            else:
                bumped = await self._dao.bump_stale_queued_run(
                    candidate.run_id,
                    expected_reconcile_attempts=candidate.reconcile_attempts,
                )
                if bumped is None:
                    continue
                run = bumped

            requeued_count += 1
            try:
                enqueued_count += await self._enqueue_run(run)
            except (RedisError, TimeoutError):
                break

        return ReconcileReport(
            failed_count=len(failed_ids),
            requeued_count=requeued_count,
            enqueued_count=enqueued_count,
        )

    async def _enqueue_run(self, run: ReconciledRun) -> bool:
        """Доставляет одно поколение; при неизвестном исходе оставляет pending."""
        run_id_str = str(run.run_id)
        job_id = _current_job_id(run)
        # Поколение — fencing token: после commit его нельзя уменьшать даже
        # при сбое Redis. Потерянные queued runs подберёт следующий проход.
        try:
            async with asyncio.timeout(REDIS_OP_TIMEOUT_SECONDS):
                job = await self._redis.enqueue_job(
                    "process_session_task",
                    session_id=run.session_id,
                    run_id=run_id_str,
                    expected_reconcile_attempts=run.reconcile_attempts,
                    _job_id=job_id,
                    _queue_name=self._queue_name,
                )
        except (RedisError, TimeoutError):
            logger.exception(
                "Redis недоступен при постановке run; поколения сохранены: run_id=%s",
                run_id_str,
            )
            raise
        except Exception:
            logger.exception(
                "Не удалось повторно поставить run в очередь ARQ: session_id=%s, run_id=%s",
                run.session_id,
                run_id_str,
            )
            return False

        if job is None:
            try:
                async with asyncio.timeout(REDIS_OP_TIMEOUT_SECONDS):
                    in_queue = await self._redis.zscore(self._queue_name, job_id)
            except (RedisError, TimeoutError):
                in_queue = None

            if in_queue is None:
                logger.warning(
                    "ARQ не подтвердил повторную постановку run "
                    "(возможен конфликт job_id): session_id=%s, run_id=%s",
                    run.session_id,
                    run_id_str,
                )
                return False

        await self._dao.confirm_reconcile_enqueue(
            run.run_id,
            expected_reconcile_attempts=run.reconcile_attempts,
        )
        logger.info(
            "Run восстановлен или уже присутствует в очереди: "
            "session_id=%s, run_id=%s (попытка %d/%d)",
            run.session_id,
            run_id_str,
            run.reconcile_attempts,
            self._max_reconcile_attempts,
        )
        return True
