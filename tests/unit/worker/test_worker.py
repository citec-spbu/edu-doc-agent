"""Задача-заглушка и подключение конфигурации к настоящему ARQ."""

import asyncio
import importlib
import logging
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest
from arq.worker import create_worker

from edu_doc_agent.config import Settings


@pytest.fixture
def tasks() -> ModuleType:
    # Импорт после isolated_app: worker читает настройки при импорте пакета.
    return importlib.import_module("edu_doc_agent.worker.tasks")


def test_task_waits_and_returns_identifiers(
    tasks: ModuleType, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(tasks.asyncio, "sleep", sleep)
    context = {"job_id": "synthetic-job"}

    with caplog.at_level(logging.INFO, logger=tasks.__name__):
        result = asyncio.run(tasks.process_session_task(context, "synthetic-session", "run-123"))

    assert result == {
        "status": "completed",
        "session_id": "synthetic-session",
        "run_id": "run-123",
    }
    sleep.assert_awaited_once_with(3)
    assert context == {"job_id": "synthetic-job"}
    assert len(caplog.records) == 2
    assert "Начата обработка" in caplog.records[0].message
    assert "успешно завершена" in caplog.records[1].message
    assert all("synthetic-session" in record.message for record in caplog.records)
    assert all("run-123" in record.message for record in caplog.records)


@pytest.mark.parametrize("error", [RuntimeError("synthetic failure"), asyncio.CancelledError()])
def test_task_does_not_claim_completion_after_error_or_cancellation(
    tasks: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: BaseException,
) -> None:
    monkeypatch.setattr(tasks.asyncio, "sleep", AsyncMock(side_effect=error))

    with caplog.at_level(logging.INFO, logger=tasks.__name__), pytest.raises(type(error)):
        asyncio.run(tasks.process_session_task({}, "synthetic-session", "failed-run"))

    assert len(caplog.records) == 1
    assert "Начата обработка" in caplog.text
    assert "успешно завершена" not in caplog.text


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
        await module.shutdown(context)

    with caplog.at_level(logging.INFO, logger=module.__name__):
        asyncio.run(scenario())

    basic_config.assert_called_once()
    assert basic_config.call_args.kwargs["level"] == "DEBUG"
    assert "starting up" in caplog.text
    assert "shutting down" in caplog.text
    assert context == {"redis": "synthetic-pool"}
