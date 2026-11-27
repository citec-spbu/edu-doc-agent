"""Обязательные сценарии приёмки issue #47 без внешних сервисов."""

import asyncio
import importlib
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from tests.fakes import InMemorySessionRunDAO


def test_stale_running_is_recovered_while_healthy_running_is_untouched(
    redis_pool: AsyncMock,
) -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        stale = await dao.create_queued_run("stale")
        healthy = await dao.create_queued_run("healthy")
        for run in (stale, healthy):
            await dao.claim_run_for_execution(run.id)
        stale.last_heartbeat_at = datetime.now(UTC) - timedelta(minutes=5)
        healthy.updated_at = stale.last_heartbeat_at
        redis_pool.ping = AsyncMock()

        report = await reconciler_class(dao=dao, redis=redis_pool).reconcile()

        assert (report.failed_count, report.requeued_count, report.enqueued_count) == (0, 1, 1)
        assert (stale.status, stale.reconcile_attempts, stale.reconcile_pending) == (
            "queued",
            1,
            False,
        )
        assert (healthy.status, healthy.reconcile_attempts) == ("running", 0)
        assert not await dao.mark_run_failed(stale.id, expected_reconcile_attempts=0)
        assert await dao.claim_run_for_execution(stale.id, expected_reconcile_attempts=1) == 1
        assert await dao.mark_run_completed(stale.id, expected_reconcile_attempts=1)

    asyncio.run(scenario())


def test_returning_old_runner_cannot_complete_or_fail_the_recovered_owner(
    redis_pool: AsyncMock,
) -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("returning-worker")
        old_started, old_resume = asyncio.Event(), asyncio.Event()
        new_started, new_finish = asyncio.Event(), asyncio.Event()

        async def old_logic(session_id):
            old_started.set()
            await old_resume.wait()

        async def new_logic(session_id):
            new_started.set()
            await new_finish.wait()

        old_task = asyncio.create_task(
            runner_class(dao, heartbeat_interval=None).execute(run.session_id, run.id, old_logic)
        )
        new_task = None
        try:
            await old_started.wait()
            run.last_heartbeat_at = datetime.now(UTC) - timedelta(minutes=5)
            redis_pool.ping = AsyncMock()
            await reconciler_class(dao=dao, redis=redis_pool).reconcile()
            new_task = asyncio.create_task(
                runner_class(dao).execute(
                    run.session_id, run.id, new_logic, expected_reconcile_attempts=1
                )
            )
            await new_started.wait()
            old_resume.set()
            assert (await old_task)["status"] == "conflict"
            assert not await dao.mark_run_failed(run.id, expected_reconcile_attempts=0)
            assert (run.status, run.reconcile_attempts) == ("running", 1)
            new_finish.set()
            assert (await new_task)["status"] == "completed"
        finally:
            pending = [task for task in (old_task, new_task) if task is not None]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_lost_queued_is_recovered_while_existing_and_fresh_jobs_are_untouched(
    redis_pool: AsyncMock,
) -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        lost = await dao.create_queued_run("lost")
        existing = await dao.create_queued_run("existing")
        fresh = await dao.create_queued_run("fresh")
        lost.updated_at = existing.updated_at = datetime.now(UTC) - timedelta(minutes=5)
        redis_pool.ping = AsyncMock()
        redis_pool.zscore = AsyncMock(
            side_effect=lambda _, job: 1 if job == str(existing.id) else None
        )

        report = await reconciler_class(dao=dao, redis=redis_pool).reconcile()

        assert (report.failed_count, report.requeued_count, report.enqueued_count) == (0, 1, 1)
        assert (lost.status, lost.reconcile_attempts, lost.reconcile_pending) == (
            "queued",
            1,
            False,
        )
        assert existing.reconcile_attempts == fresh.reconcile_attempts == 0
        assert redis_pool.enqueue_job.call_args.kwargs["run_id"] == str(lost.id)

    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["running", "queued"])
def test_exhausted_reconciliation_fails_run_and_releases_session_over_http(
    client: TestClient, dao: InMemorySessionRunDAO, redis_pool: AsyncMock, status: str
) -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler
    url = "/api/v1/sessions/exhausted/process"
    first = client.post(url)
    assert first.status_code == 202
    run_id = UUID(first.json()["run_id"])
    assert client.post(url).status_code == 409
    redis_pool.enqueue_job.reset_mock()

    async def scenario() -> None:
        run = dao.runs[run_id]
        if status == "running":
            await dao.claim_run_for_execution(run.id)
        run.reconcile_attempts = 3
        run.updated_at = run.last_heartbeat_at = datetime.now(UTC) - timedelta(minutes=5)
        redis_pool.ping = AsyncMock()
        redis_pool.zscore = AsyncMock(return_value=None)
        report = await reconciler_class(dao=dao, redis=redis_pool).reconcile()
        assert (report.failed_count, report.requeued_count, report.enqueued_count) == (1, 0, 0)
        assert run.status == "failed"
        redis_pool.enqueue_job.assert_not_awaited()

    asyncio.run(scenario())
    result = client.get(f"/api/v1/sessions/exhausted/runs/{run_id}")
    assert result.json()["status"] == "failed"
    accepted = client.post(url)
    assert accepted.status_code == 202
    assert accepted.json()["run_id"] != str(run_id)


@pytest.mark.parametrize("committed", [False, True])
def test_unknown_enqueue_outcome_retries_the_same_delivery(
    redis_pool: AsyncMock, committed: bool
) -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("unknown-outcome")
        queued = set()
        attempts = []

        async def enqueue(*args, **kwargs):
            attempts.append(kwargs["_job_id"])
            if len(attempts) == 1:
                if committed:
                    queued.add(kwargs["_job_id"])
                raise RedisConnectionError("synthetic lost response")
            queued.add(kwargs["_job_id"])
            return object()

        redis_pool.ping = AsyncMock()
        redis_pool.zscore = AsyncMock(side_effect=lambda _, job: 1 if job in queued else None)
        redis_pool.enqueue_job.side_effect = enqueue
        reconciler = reconciler_class(dao=dao, redis=redis_pool, stale_queued_seconds=0)
        await reconciler.reconcile()
        assert run.reconcile_pending
        await reconciler.reconcile()
        assert (run.status, run.reconcile_attempts, run.reconcile_pending) == ("queued", 1, False)
        assert set(attempts) == {f"{run.id}:reconcile:1"}
        assert len(attempts) == (1 if committed else 2)
        assert queued == {f"{run.id}:reconcile:1"}

    asyncio.run(scenario())
