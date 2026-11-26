"""Поколение восстановления связывает ARQ-задачу с владельцем Run."""

import asyncio
import importlib
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from arq.connections import ArqRedis
from redis.exceptions import ConnectionError as RedisConnectionError

from edu_doc_agent.dao import ReconciledRun, SessionRunDAO
from tests.fakes import InMemorySessionRunDAO


@pytest.mark.parametrize(
    "error", [RedisConnectionError("synthetic outage"), TimeoutError(), RuntimeError()]
)
def test_enqueue_failure_preserves_generations(error: Exception) -> None:
    reconciler_module = importlib.import_module("edu_doc_agent.worker.reconciler")

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        runs = [await dao.create_queued_run(f"synthetic-{i}") for i in range(2)]
        for run in runs:
            await dao.claim_run_for_execution(run.id)
            run.last_heartbeat_at = datetime.now(UTC) - timedelta(minutes=5)
        redis = redis_for_reconcile()
        redis.enqueue_job.side_effect = error

        report = await reconciler_module.RunReconciler(dao=dao, redis=redis).reconcile()

        assert report.requeued_count == 2
        assert report.enqueued_count == 0
        for run in runs:
            assert (run.status, run.reconcile_attempts, run.reconcile_pending) == (
                "queued",
                1,
                True,
            )
            assert not await dao.mark_run_failed(run.id, expected_reconcile_attempts=0)

    asyncio.run(scenario())


def test_reconciler_includes_generation_in_job_payload() -> None:
    reconciler_module = importlib.import_module("edu_doc_agent.worker.reconciler")

    async def scenario() -> None:
        dao = AsyncMock(spec=SessionRunDAO)
        run = ReconciledRun(uuid4(), "synthetic-session", 2)
        dao.fail_exhausted_stale_running_runs.return_value = []
        dao.requeue_stale_running_runs.return_value = [run]
        dao.list_stale_queued_runs.return_value = []
        redis = AsyncMock(spec=ArqRedis)
        redis.default_queue_name = "arq:queue"
        redis.ping = AsyncMock()

        report = await reconciler_module.RunReconciler(dao=dao, redis=redis).reconcile()

        assert report.enqueued_count == 1
        payload = redis.enqueue_job.call_args.kwargs
        assert payload["expected_reconcile_attempts"] == 2
        assert payload["_job_id"] == f"{run.run_id}:reconcile:2"

    asyncio.run(scenario())


@pytest.mark.parametrize("generation", [0, 1])
def test_old_job_on_last_try_cannot_claim_recovered_run(
    monkeypatch: pytest.MonkeyPatch, generation: int
) -> None:
    tasks = importlib.import_module("edu_doc_agent.worker.tasks")

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        run.reconcile_attempts = 2
        logic = AsyncMock(side_effect=asyncio.CancelledError)
        monkeypatch.setattr(tasks, "_stub_session_logic", logic)
        # Старые API-задачи не имеют поколения в payload и относятся к поколению 0.
        payload = {"expected_reconcile_attempts": generation} if generation else {}
        result = await tasks.process_session_task(
            {"dao": dao, "job_try": 3}, run.session_id, str(run.id), **payload
        )
        assert result["status"] == "skipped"
        assert run.status == "queued"
        assert run.reconcile_attempts == 2
        logic.assert_not_awaited()

        # Штатный retry сохраняет поколение, а счётчик ARQ начинается заново.
        with pytest.raises(asyncio.CancelledError):
            await tasks.process_session_task(
                {"dao": dao, "job_try": 1},
                run.session_id,
                str(run.id),
                expected_reconcile_attempts=2,
            )
        assert run.status == "queued"
        logic.side_effect = None
        result = await tasks.process_session_task(
            {"dao": dao, "job_try": 2},
            run.session_id,
            str(run.id),
            expected_reconcile_attempts=2,
        )
        assert result["status"] == "completed"
        assert run.status == "completed"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def redis_for_reconcile(queue_name: str = "arq:queue") -> AsyncMock:
    redis = AsyncMock(spec=ArqRedis)
    redis.default_queue_name = queue_name
    redis.ping = AsyncMock()
    redis.zscore = AsyncMock(return_value=None)
    return redis


def test_repeated_cron_cancellation_before_enqueue_does_not_exhaust_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        await dao.claim_run_for_execution(run.id)
        run.last_heartbeat_at = datetime.now(UTC) - timedelta(minutes=5)
        original_list = dao.list_stale_queued_runs
        # Cron отменён после commit партии, до первого enqueue, три раза подряд.
        monkeypatch.setattr(
            dao, "list_stale_queued_runs", AsyncMock(side_effect=asyncio.CancelledError)
        )
        redis = redis_for_reconcile()
        for _ in range(3):
            with pytest.raises(asyncio.CancelledError):
                await reconciler_class(dao=dao, redis=redis, stale_queued_seconds=0).reconcile()
            assert (run.status, run.reconcile_attempts, run.reconcile_pending) == (
                "queued",
                1,
                True,
            )
        redis.enqueue_job.assert_not_awaited()

        monkeypatch.setattr(dao, "list_stale_queued_runs", original_list)
        report = await reconciler_class(dao=dao, redis=redis, stale_queued_seconds=0).reconcile()
        assert report.enqueued_count == 1
        assert (run.reconcile_attempts, run.reconcile_pending) == (1, False)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [RedisConnectionError(), TimeoutError(), RuntimeError(), None])
def test_unconfirmed_last_delivery_can_be_retried_without_a_new_generation(failure) -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        run.reconcile_attempts = 2  # Две ранее подтверждённые доставки.
        redis = redis_for_reconcile()
        redis.enqueue_job.side_effect = failure
        redis.enqueue_job.return_value = None
        for _ in range(4):
            report = await reconciler_class(
                dao=dao, redis=redis, stale_queued_seconds=0
            ).reconcile()
            assert (report.failed_count, report.enqueued_count) == (0, 0)
            assert (run.status, run.reconcile_attempts, run.reconcile_pending) == (
                "queued",
                3,
                True,
            )
        assert {c.kwargs["_job_id"] for c in redis.enqueue_job.await_args_list} == {
            f"{run.id}:reconcile:3"
        }
        redis.enqueue_job.side_effect = None
        redis.enqueue_job.return_value = object()
        report = await reconciler_class(dao=dao, redis=redis, stale_queued_seconds=0).reconcile()
        assert report.enqueued_count == 1
        assert not run.reconcile_pending
        # Подтверждённая, но снова потерянная третья доставка исчерпывает лимит.
        report = await reconciler_class(dao=dao, redis=redis, stale_queued_seconds=0).reconcile()
        assert report.failed_count == 1
        assert run.status == "failed"

    asyncio.run(scenario())


def test_cancellation_after_redis_accepts_job_recovers_unprocessed_batch() -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        runs = [await dao.create_queued_run(f"synthetic-{i}") for i in range(2)]
        for run in runs:
            await dao.claim_run_for_execution(run.id)
            run.last_heartbeat_at = datetime.now(UTC) - timedelta(minutes=5)
        queue = set()
        redis = redis_for_reconcile()

        async def accepted_then_cancelled(*args, **kwargs):
            queue.add(kwargs["_job_id"])
            raise asyncio.CancelledError

        redis.enqueue_job.side_effect = accepted_then_cancelled
        with pytest.raises(asyncio.CancelledError):
            await reconciler_class(dao=dao, redis=redis).reconcile()
        assert all(r.reconcile_attempts == 1 and r.reconcile_pending for r in runs)
        redis.enqueue_job.side_effect = None

        async def score(queue_name, job_id):
            return 1 if job_id in queue else None

        redis.zscore.side_effect = score
        report = await reconciler_class(dao=dao, redis=redis, stale_queued_seconds=0).reconcile()
        assert report.enqueued_count == 1  # Первая job уже в Redis, вторая доставлена сейчас.
        assert all(r.reconcile_attempts == 1 and not r.reconcile_pending for r in runs)

    asyncio.run(scenario())


@pytest.mark.parametrize("override", [None, "synthetic:override-queue"])
def test_cron_uses_worker_pool_queue_for_lookup_and_enqueue(override: str | None) -> None:
    tasks = importlib.import_module("edu_doc_agent.worker.tasks")

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        existing = await dao.create_queued_run("existing")
        missing = await dao.create_queued_run("missing")
        redis = redis_for_reconcile("synthetic:named-queue")
        expected_queue = override or redis.default_queue_name

        async def score(queue_name, job_id):
            assert queue_name == expected_queue
            return 1 if job_id == str(existing.id) else None

        redis.zscore.side_effect = score
        report = await tasks.reconcile_stale_runs_task(
            {"dao": dao, "redis": redis, "stale_queued_seconds": 0, "queue_name": override}
        )
        assert report["enqueued_count"] == 1
        assert existing.reconcile_attempts == 0
        redis.enqueue_job.assert_awaited_once_with(
            "process_session_task",
            session_id=missing.session_id,
            run_id=str(missing.id),
            expected_reconcile_attempts=1,
            _job_id=f"{missing.id}:reconcile:1",
            _queue_name=expected_queue,
        )

    asyncio.run(scenario())


def test_checked_page_does_not_starve_the_101st_lost_run() -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        old = datetime.now(UTC) - timedelta(minutes=5)
        runs = []
        for i in range(101):
            run = await dao.create_queued_run(f"synthetic-{i}", run_id=UUID(int=i + 1))
            run.updated_at = old
            runs.append(run)
        existing_jobs = {str(r.id) for r in runs[:100]}
        redis = redis_for_reconcile()

        async def score(queue_name, job_id):
            return 1 if job_id in existing_jobs else None

        redis.zscore.side_effect = score
        first = await reconciler_class(dao=dao, redis=redis, stale_queued_seconds=0).reconcile()
        assert first.enqueued_count == 0
        assert all(r.updated_at > old for r in runs[:100])
        # Новый экземпляр cron, без сохранённого в памяти курсора.
        second = await reconciler_class(dao=dao, redis=redis, stale_queued_seconds=0).reconcile()
        assert second.enqueued_count == 1
        assert redis.enqueue_job.call_args.kwargs["run_id"] == str(runs[100].id)
        assert runs[100].reconcile_attempts == 1

    asyncio.run(scenario())
