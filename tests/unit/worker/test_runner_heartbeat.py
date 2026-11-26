"""Lease ограничивает работу при потере БД; отмена не превращается в completed."""

import asyncio
import importlib
from unittest.mock import AsyncMock

import pytest

from tests.fakes import InMemorySessionRunDAO


@pytest.mark.parametrize("cancel_runner", [False, True])
def test_completed_business_is_saved_before_slow_heartbeat_cleanup(
    monkeypatch: pytest.MonkeyPatch, cancel_runner: bool
) -> None:
    module = importlib.import_module("edu_doc_agent.worker.runner")

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        pulse_started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        effects = []

        async def heartbeat(*args, **kwargs):
            pulse_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await module._finish_status_update(release.wait())

        async def logic(session_id: str) -> None:
            await pulse_started.wait()
            effects.append(session_id)

        monkeypatch.setattr(dao, "touch_heartbeat", heartbeat)
        runner = module.SessionRunner(dao, heartbeat_interval=0.01, lease_timeout=0.12)
        task = asyncio.create_task(runner.execute(run.session_id, run.id, logic))
        cancelled = False
        try:
            await cleaning.wait()
            assert run.status == "completed"
            await asyncio.sleep(0.15)  # Исходный lease истёк, очистка ещё ждёт.
            assert not task.done()
            assert (
                await dao.requeue_stale_running_runs(
                    stale_running_seconds=0, max_reconcile_attempts=3
                )
                == []
            )
            if cancel_runner:
                task.cancel()
                cancelled = True
        finally:
            release.set()
            if cancelled:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await task
        if not cancelled:
            assert result["status"] == "completed"
        assert run.status == "completed"
        assert (await runner.execute(run.session_id, run.id, logic))["status"] == "skipped"
        assert effects == [run.session_id]

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_lease_timer_stops_logic_before_heartbeat_cleanup_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("edu_doc_agent.worker.runner")

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        cleanup_started, cleanup_release = asyncio.Event(), asyncio.Event()
        logic_stopped = asyncio.Event()
        effects = []

        async def heartbeat(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                # Как shielded rollback/close SQLAlchemy: отмена не завершает очистку.
                await module._finish_status_update(cleanup_release.wait())

        async def logic(session_id: str) -> None:
            try:
                await asyncio.sleep(0.18)
                effects.append("business effect after lease expiry")
            finally:
                logic_stopped.set()

        monkeypatch.setattr(dao, "touch_heartbeat", heartbeat)
        task = asyncio.create_task(
            module.SessionRunner(dao, heartbeat_interval=0.01, lease_timeout=0.08).execute(
                run.session_id, run.id, logic
            )
        )
        try:
            await cleanup_started.wait()
            await asyncio.sleep(0.20)
            assert logic_stopped.is_set()
            assert effects == []
            assert not task.done()  # Очистка БД ещё не завершена.
        finally:
            cleanup_release.set()
            result = await task
        assert result["status"] == "conflict"
        assert run.status == "running"

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_delayed_claim_acknowledgement_gets_an_immediate_first_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        claim = dao.claim_run_for_execution
        pulse = asyncio.Event()
        touch = dao.touch_heartbeat

        async def delayed_claim(*args, **kwargs):
            token = await claim(*args, **kwargs)
            await asyncio.sleep(0.08)
            return token

        async def heartbeat(*args, **kwargs):
            updated = await touch(*args, **kwargs)
            pulse.set()
            return updated

        async def logic(session_id: str) -> None:
            await pulse.wait()

        monkeypatch.setattr(dao, "claim_run_for_execution", delayed_claim)
        monkeypatch.setattr(dao, "touch_heartbeat", AsyncMock(side_effect=heartbeat))
        result = await runner_class(dao, heartbeat_interval=0.06, lease_timeout=0.12).execute(
            run.session_id, run.id, logic
        )
        assert result["status"] == "completed"
        dao.touch_heartbeat.assert_awaited_once()

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("failure", ["error", "hang"])
def test_database_outage_stops_logic_when_local_lease_expires(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        stopped = asyncio.Event()

        async def heartbeat(*args, **kwargs):
            if failure == "error":
                raise RuntimeError("synthetic database outage")
            await asyncio.Event().wait()

        monkeypatch.setattr(dao, "touch_heartbeat", AsyncMock(side_effect=heartbeat))
        original_completed = dao.mark_run_completed
        completed, failed = AsyncMock(), AsyncMock()
        monkeypatch.setattr(dao, "mark_run_completed", completed)
        monkeypatch.setattr(dao, "mark_run_failed", failed)

        async def logic(session_id: str) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        result = await runner_class(dao, heartbeat_interval=0.01, lease_timeout=0.08).execute(
            run.session_id, run.id, logic
        )
        assert result["status"] == "conflict"
        assert stopped.is_set()
        assert run.status == "running"  # Восстановлением займётся Reconciler.
        completed.assert_not_awaited()
        failed.assert_not_awaited()

        # Восстановленный владелец начинает работу после остановки старой логики.
        recovered = await dao.requeue_stale_running_runs(
            stale_running_seconds=0, max_reconcile_attempts=3
        )
        assert len(recovered) == 1
        monkeypatch.setattr(dao, "mark_run_completed", original_completed)

        async def new_logic(session_id: str) -> None:
            assert stopped.is_set()

        result = await runner_class(dao).execute(
            run.session_id, run.id, new_logic, expected_reconcile_attempts=1
        )
        assert result["status"] == "completed"
        assert run.status == "completed"

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_claim_acknowledgement_cannot_extend_local_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        original = dao.claim_run_for_execution

        async def delayed_ack(*args, **kwargs):
            token = await original(*args, **kwargs)
            await asyncio.sleep(0.08)
            return token

        monkeypatch.setattr(dao, "claim_run_for_execution", delayed_ack)
        logic = AsyncMock()
        result = await runner_class(dao, heartbeat_interval=0.01, lease_timeout=0.04).execute(
            run.session_id, run.id, logic
        )
        assert result["status"] == "conflict"
        logic.assert_not_awaited()
        assert run.status == "running"

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_logic_that_suppresses_lease_cancellation_cannot_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        monkeypatch.setattr(dao, "touch_heartbeat", AsyncMock(return_value=False))
        completed = AsyncMock()
        monkeypatch.setattr(dao, "mark_run_completed", completed)

        async def logic(session_id: str) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return

        result = await runner_class(dao, heartbeat_interval=0.01).execute(
            run.session_id, run.id, logic
        )
        assert result["status"] == "conflict"
        completed.assert_not_awaited()

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_arq_task_uses_reconciler_threshold_for_local_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = importlib.import_module("edu_doc_agent.worker.tasks")

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        monkeypatch.setattr(
            dao, "touch_heartbeat", AsyncMock(side_effect=RuntimeError("synthetic outage"))
        )

        async def logic(session_id: str) -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(tasks, "_stub_session_logic", logic)
        result = await tasks.process_session_task(
            {"dao": dao, "heartbeat_interval": 0.01, "stale_running_seconds": 0.08},
            run.session_id,
            str(run.id),
        )
        assert result["status"] == "conflict"

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_successful_heartbeat_renews_local_lease() -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")

        async def logic(session_id: str) -> None:
            await asyncio.sleep(0.2)

        result = await runner_class(dao, heartbeat_interval=0.01, lease_timeout=0.08).execute(
            run.session_id, run.id, logic
        )
        assert result["status"] == "completed"
        assert run.status == "completed"

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("heartbeat_returns", [False, True])
@pytest.mark.parametrize("job_try", [1, 3])
def test_runner_cancellation_when_heartbeat_finishes_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, heartbeat_returns: bool, job_try: int
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        started = asyncio.Event()
        runner = runner_class(dao)

        async def heartbeat(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Отмена Runner и завершение heartbeat происходят в одном обороте loop.
                assert run.status == "completed"
                task.cancel()
                if not heartbeat_returns:
                    raise

        monkeypatch.setattr(runner, "_heartbeat_loop", heartbeat)
        completed = AsyncMock(wraps=dao.mark_run_completed)
        monkeypatch.setattr(dao, "mark_run_completed", completed)

        async def logic(session_id: str) -> None:
            await started.wait()

        task = asyncio.create_task(runner.execute(run.session_id, run.id, logic, job_try=job_try))
        with pytest.raises(asyncio.CancelledError):
            await task
        assert run.status == "completed"
        completed.assert_awaited_once_with(run.id, expected_reconcile_attempts=0)

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_runner_cancellation_survives_logic_suppression_after_heartbeat_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        heartbeat_done = asyncio.Event()
        runner = runner_class(dao)

        async def heartbeat(*args, **kwargs):
            heartbeat_done.set()

        async def logic(session_id: str) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return

        monkeypatch.setattr(runner, "_heartbeat_loop", heartbeat)
        task = asyncio.create_task(runner.execute(run.session_id, run.id, logic))
        await heartbeat_done.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert run.status == "queued"

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))
