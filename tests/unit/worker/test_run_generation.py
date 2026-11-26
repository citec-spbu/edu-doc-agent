"""Поколение восстановления связывает ARQ-задачу с владельцем Run."""

import asyncio
import importlib
from unittest.mock import AsyncMock
from uuid import uuid4

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
        dao = AsyncMock(spec=SessionRunDAO)
        runs = [ReconciledRun(uuid4(), f"synthetic-{i}", 1) for i in range(2)]
        dao.fail_exhausted_stale_running_runs.return_value = []
        dao.requeue_stale_running_runs.return_value = runs
        dao.list_stale_queued_runs.return_value = []
        redis = AsyncMock(spec=ArqRedis)
        redis.ping = AsyncMock()
        redis.enqueue_job.side_effect = error

        report = await reconciler_module.RunReconciler(dao=dao, redis=redis).reconcile()

        assert report.requeued_count == 2
        assert report.enqueued_count == 0
        # Ни у сбойной задачи, ни у ещё не поставленной не возвращается старый токен.
        assert not any(call[0] == "rollback_reconcile_attempt" for call in dao.mock_calls)

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
