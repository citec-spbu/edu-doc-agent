"""Общие фикстуры и настройки pytest для всего проекта."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, Mock

import pytest
from arq.connections import ArqRedis
from arq.jobs import Job
from fastapi.testclient import TestClient
from starlette.datastructures import State

from edu_doc_agent import config, main
from edu_doc_agent.config import Settings
from edu_doc_agent.main import app


def pytest_addoption(parser: pytest.Parser) -> None:
    """Регистрирует кастомные флаги командной строки.

    По умолчанию тесты, которым нужна настоящая LLM или БД,
    пропускаются: они медленные, дорогие и/или требуют
    окружения, которого нет в CI.
    """
    parser.addoption(
        "--run-llm",
        action="store_true",
        default=False,
        help="Запустить тесты, которые ходят в настоящую LLM.",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Запустить интеграционные тесты (нужна БД).",
    )
    parser.addoption(
        "--run-compose",
        action="store_true",
        default=False,
        help="Запустить изолированный Compose smoke (Docker и TEST_APP_IMAGE).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "compose: запускает отдельный тестовый Compose-проект")
    config.addinivalue_line("markers", "pr_review: обязательные регрессии по ревью PR #29")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Снимает skip с тестов по флагам --run-llm и --run-integration."""
    if not config.getoption("--run-llm"):
        skip_llm = pytest.mark.skip(reason="нужен флаг --run-llm")
        for item in items:
            if item.get_closest_marker("llm"):
                item.add_marker(skip_llm)

    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="нужен флаг --run-integration")
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip_integration)

    if not config.getoption("--run-compose"):
        for item in items:
            if item.get_closest_marker("compose"):
                item.add_marker(pytest.mark.skip(reason="нужен флаг --run-compose"))


@pytest.fixture(autouse=True)
def isolated_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Не читать настоящий .env и не оставлять app.state между тестами."""
    settings = Settings(_env_file=None, redis_url="redis://127.0.0.1:6379/15")
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(app, "state", State())


@pytest.fixture
def redis_pool() -> AsyncMock:
    pool = AsyncMock(spec=ArqRedis)
    # redis.close обёрнут декоратором deprecation: autospec считает его sync.
    pool.close = AsyncMock()
    pool.enqueue_job.return_value = Mock(spec=Job)
    return pool


@pytest.fixture
def client(redis_pool: AsyncMock) -> Iterator[TestClient]:
    """HTTP-клиент к FastAPI-приложению для юнит-тестов API."""
    app.state.redis_pool = redis_pool

    with TestClient(app) as test_client:
        yield test_client
