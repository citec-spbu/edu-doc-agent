"""Задача-заглушка и подключение конфигурации к настоящему ARQ."""

import asyncio
import importlib
import logging
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from arq.worker import create_worker

from edu_doc_agent.config import Settings
from tests.fakes import InMemorySessionRunDAO


@pytest.fixture
def tasks() -> ModuleType:
    # Импорт после isolated_app: worker читает настройки при импорте пакета.
    return importlib.import_module("edu_doc_agent.worker.tasks")


@pytest.fixture
def runner_module() -> ModuleType:
    return importlib.import_module("edu_doc_agent.worker.runner")


def test_task_waits_and_returns_identifiers(
    tasks: ModuleType,
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    before_work: list[logging.LogRecord] = []

    async def work(delay: float) -> None:
        before_work.extend(caplog.records)

    sleep = AsyncMock(side_effect=work)
    monkeypatch.setattr(tasks, "asyncio", SimpleNamespace(sleep=sleep))
    dao = InMemorySessionRunDAO()

    async def scenario():
        run = await dao.create_queued_run("synthetic-session")
        result = await tasks.process_session_task(
            {"dao": dao, "job_id": "synthetic-job"}, "synthetic-session", str(run.id)
        )
        return run, result

    with caplog.at_level(logging.INFO, logger=runner_module.__name__):
        run, result = asyncio.run(scenario())

    assert result == {
        "status": "completed",
        "session_id": "synthetic-session",
        "run_id": str(run.id),
    }
    assert run.status == "completed"
    sleep.assert_awaited_once_with(3)
    for phase in (before_work, caplog.records[len(before_work) :]):
        assert any(
            record.name == runner_module.__name__
            and record.levelno == logging.INFO
            and "synthetic-session" in record.getMessage()
            and str(run.id) in record.getMessage()
            for record in phase
        )


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        pytest.param(RuntimeError("synthetic failure"), "failed", id="error"),
        pytest.param(asyncio.CancelledError(), "queued", id="cancellation"),
    ],
)
def test_task_propagates_error_or_cancellation(
    tasks: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    expected_status: str,
) -> None:
    monkeypatch.setattr(tasks, "asyncio", SimpleNamespace(sleep=AsyncMock(side_effect=error)))
    dao = InMemorySessionRunDAO()

    async def scenario():
        run = await dao.create_queued_run("synthetic-session")
        # Ошибка/отмена должна выйти к ARQ, а не превратиться в completed.
        with pytest.raises(type(error)):
            await tasks.process_session_task({"dao": dao}, "synthetic-session", str(run.id))
        return run

    run = asyncio.run(scenario())
    # failed при ошибке; queued при отмене на первой попытке (ARQ сделает retry)
    assert run.status == expected_status


def test_arq_loads_the_registered_task_and_execution_limits(tasks: ModuleType) -> None:
    module = importlib.import_module("edu_doc_agent.worker.settings")
    package = importlib.import_module("edu_doc_agent.worker")

    async def scenario() -> None:
        worker = create_worker(module.WorkerSettings, handle_signals=False)
        assert worker.functions["process_session_task"].coroutine is tasks.process_session_task
        assert worker.job_timeout_s == 600
        assert worker.max_tries == 3
        assert worker.redis_settings == module.settings.arq_redis_settings
        assert worker.on_startup is module.startup
        assert worker.on_shutdown is module.shutdown

    asyncio.run(scenario())
    assert package.WorkerSettings is module.WorkerSettings


def test_worker_lifecycle_hooks_configure_logging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tasks: ModuleType
) -> None:
    module = importlib.import_module("edu_doc_agent.worker.settings")
    monkeypatch.setattr(module, "settings", Settings(_env_file=None, log_level="DEBUG"))
    basic_config = Mock()
    monkeypatch.setattr(module.logging, "basicConfig", basic_config)
    context = {"redis": "synthetic-pool"}

    async def scenario() -> None:
        await module.startup(context)
        assert any(
            record.name == module.__name__ and record.levelno == logging.INFO
            for record in caplog.records
        )
        caplog.clear()
        await module.shutdown(context)
        assert any(
            record.name == module.__name__ and record.levelno == logging.INFO
            for record in caplog.records
        )

    with caplog.at_level(logging.INFO, logger=module.__name__):
        asyncio.run(scenario())

    assert any(call.kwargs.get("level") == "DEBUG" for call in basic_config.call_args_list)
