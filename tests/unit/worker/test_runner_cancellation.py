"""Отмена на границах записи статусов не оставляет потерянный running run."""

import asyncio
import importlib
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from tests.fakes import InMemorySessionRunDAO


@pytest.mark.parametrize("commit_before_pause", [False, True])
@pytest.mark.parametrize("job_try", [1, 3])
@pytest.mark.parametrize(
    "operation", ["claim_run_for_execution", "mark_run_completed", "mark_run_failed"]
)
def test_cancel_during_status_update_waits_for_outcome(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    job_try: int,
    commit_before_pause: bool,
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        entered, release = asyncio.Event(), asyncio.Event()
        original = getattr(dao, operation)

        async def paused_update(run_id: UUID, **kwargs: object) -> object:
            if commit_before_pause:
                result = await original(run_id, **kwargs)
            entered.set()
            await release.wait()
            return result if commit_before_pause else await original(run_id, **kwargs)

        monkeypatch.setattr(dao, operation, paused_update)
        logic = AsyncMock(
            side_effect=(
                RuntimeError("synthetic failure") if operation == "mark_run_failed" else None
            )
        )
        runner = runner_class(dao)
        task = asyncio.create_task(runner.execute(run.session_id, run.id, logic, job_try=job_try))
        try:
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), "Отмена не должна обгонять завершение операции БД"
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), "Повторная отмена не должна прерывать операцию БД"
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        if operation == "claim_run_for_execution":
            assert run.status == ("failed" if job_try == 3 else "queued")
            logic.assert_not_awaited()
        else:
            assert run.status == ("failed" if operation == "mark_run_failed" else "completed")
            logic.assert_awaited_once_with(run.session_id)

        # Повтор того же run действительно выполняется после requeue, а уже
        # зафиксированный терминальный исход не вызывает бизнес-логику повторно.
        retry_logic = AsyncMock()
        result = await runner.execute(run.session_id, run.id, retry_logic)
        if operation == "claim_run_for_execution" and job_try == 1:
            assert result["status"] == "completed"
            retry_logic.assert_awaited_once_with(run.session_id)
        else:
            assert result["status"] == "skipped"
            retry_logic.assert_not_awaited()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_cancelled_unsuccessful_claim_does_not_requeue_another_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        assert await dao.claim_run_for_execution(run.id) == 0
        entered, release = asyncio.Event(), asyncio.Event()
        original = dao.claim_run_for_execution

        async def paused_claim(run_id: UUID, **kwargs: object) -> int | None:
            claimed = await original(run_id, **kwargs)
            entered.set()
            await release.wait()
            return claimed

        monkeypatch.setattr(dao, "claim_run_for_execution", paused_claim)
        logic = AsyncMock()
        task = asyncio.create_task(runner_class(dao).execute(run.session_id, run.id, logic))
        try:
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert run.status == "running"
        logic.assert_not_awaited()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


@pytest.mark.parametrize(
    ("job_try", "operation", "expected_status"),
    [(1, "requeue_running_run", "queued"), (3, "mark_run_failed", "failed")],
)
def test_repeated_cancellation_waits_for_cleanup(
    monkeypatch: pytest.MonkeyPatch, job_try: int, operation: str, expected_status: str
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = getattr(dao, operation)

        async def logic(session_id: str) -> None:
            started.set()
            await asyncio.Event().wait()

        async def paused_cleanup(run_id: UUID, **kwargs: object) -> bool:
            cleaning.set()
            await release.wait()
            return await original(run_id, **kwargs)

        monkeypatch.setattr(dao, operation, paused_cleanup)
        task = asyncio.create_task(
            runner_class(dao).execute(run.session_id, run.id, logic, job_try=job_try)
        )
        try:
            await started.wait()
            task.cancel()
            await cleaning.wait()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert run.status == "running"
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert run.status == expected_status

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_default_runner_allows_work_beyond_570_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    runner_module = importlib.import_module("edu_doc_agent.worker.runner")

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        original_time = loop.time
        elapsed = 0
        monkeypatch.setattr(loop, "time", lambda: original_time() + elapsed)
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")

        async def logic(session_id: str) -> None:
            nonlocal elapsed
            elapsed = 580
            # Два оборота loop позволяют сработать просроченному callback таймаута.
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        # Здесь проверяется отсутствие внутреннего лимита исполнения 570 с.
        # Прыжок часов без heartbeat отдельно означал бы истечение lease.
        result = await runner_module.SessionRunner(dao, heartbeat_interval=None).execute(
            run.session_id, run.id, logic
        )
        assert result["status"] == "completed"
        assert run.status == "completed"

    asyncio.run(scenario())
