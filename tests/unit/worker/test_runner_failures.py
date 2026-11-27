"""Ошибки записи статусов не подменяют исход бизнес-функции или отмену worker."""

import asyncio
import importlib
import logging
from unittest.mock import AsyncMock

import pytest

from tests.fakes import InMemorySessionRunDAO


@pytest.mark.parametrize("kind", ["future", "custom-awaitable"])
def test_business_callable_accepts_any_awaitable(kind: str) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)

        class CustomAwaitable:
            def __await__(self):
                return future.__await__()

        def logic(session_id: str):
            assert session_id == run.session_id
            return future if kind == "future" else CustomAwaitable()

        result = await runner_class(dao).execute(run.session_id, run.id, logic)
        assert result["status"] == "completed"
        assert run.status == "completed"

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["sync-error", "future-error"])
def test_business_callable_failure_before_coroutine_marks_run_failed(kind: str) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        error = ValueError("synthetic business failure")

        def logic(session_id: str):
            if kind == "sync-error":
                raise error
            future = asyncio.get_running_loop().create_future()
            future.set_exception(error)
            return future

        with pytest.raises(ValueError) as caught:
            await runner_class(dao).execute(run.session_id, run.id, logic)
        assert caught.value is error
        assert run.status == "failed"

    asyncio.run(scenario())


def test_runner_cancellation_propagates_to_business_future() -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        future = asyncio.get_running_loop().create_future()
        started = asyncio.Event()

        def logic(session_id: str):
            started.set()
            return future

        task = asyncio.create_task(runner_class(dao).execute(run.session_id, run.id, logic))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert future.cancelled()
        assert run.status == "queued"

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_failed_write_preserves_original_business_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runner_module = importlib.import_module("edu_doc_agent.worker.runner")

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        business_error = ValueError("synthetic business failure")
        write_error = RuntimeError("synthetic failed-status write failure")
        monkeypatch.setattr(dao, "mark_run_failed", AsyncMock(side_effect=write_error))
        logic = AsyncMock(side_effect=business_error)
        with pytest.raises(ValueError) as caught:
            await runner_module.SessionRunner(dao).execute(run.session_id, run.id, logic)
        assert caught.value is business_error
        assert run.status == "running"  # Неудачная запись не изображает сохранённый failed.
        assert any(
            record.exc_info is not None
            and record.exc_info[1] is write_error
            and run.session_id in record.getMessage()
            and str(run.id) in record.getMessage()
            for record in caplog.records
        )

    with caplog.at_level(logging.ERROR, logger=runner_module.__name__):
        asyncio.run(scenario())


def test_completed_write_error_propagates_without_reclassifying_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        write_error = RuntimeError("synthetic completed-status write failure")
        failed_write = AsyncMock()
        monkeypatch.setattr(dao, "mark_run_completed", AsyncMock(side_effect=write_error))
        monkeypatch.setattr(dao, "mark_run_failed", failed_write)
        logic = AsyncMock()
        with pytest.raises(RuntimeError) as caught:
            await runner_class(dao).execute(run.session_id, run.id, logic)
        assert caught.value is write_error
        assert run.status == "running"
        logic.assert_awaited_once_with(run.session_id)
        failed_write.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("job_try", [1, 3])
def test_cancellation_cleanup_error_is_logged_and_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, job_try: int
) -> None:
    runner_module = importlib.import_module("edu_doc_agent.worker.runner")

    async def scenario() -> None:
        dao = InMemorySessionRunDAO()
        run = await dao.create_queued_run("synthetic-session")
        write_error = RuntimeError("synthetic cancellation cleanup failure")
        operation = "mark_run_failed" if job_try == 3 else "requeue_running_run"
        monkeypatch.setattr(dao, operation, AsyncMock(side_effect=write_error))
        with pytest.raises(asyncio.CancelledError):
            await runner_module.SessionRunner(dao).execute(
                run.session_id,
                run.id,
                AsyncMock(side_effect=asyncio.CancelledError),
                job_try=job_try,
            )
        assert run.status == "running"
        assert any(
            record.exc_info is not None
            and record.exc_info[1] is write_error
            and run.session_id in record.getMessage()
            and str(run.id) in record.getMessage()
            for record in caplog.records
        )

    with caplog.at_level(logging.ERROR, logger=runner_module.__name__):
        asyncio.run(scenario())
