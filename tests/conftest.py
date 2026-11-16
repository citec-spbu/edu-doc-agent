"""Общие фикстуры и настройки pytest для всего проекта."""

from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from arq.connections import ArqRedis
from arq.jobs import Job
from fastapi.testclient import TestClient
from starlette.datastructures import State

from edu_doc_agent import config, main
from edu_doc_agent.api import dependencies as api_dependencies
from edu_doc_agent.api import router as api_router
from edu_doc_agent.config import Settings
from edu_doc_agent.dao import ActiveRunConflictError, Run, Session, SessionRunDAO
from edu_doc_agent.main import app
from edu_doc_agent.worker import settings as worker_settings
from edu_doc_agent.worker import tasks as worker_tasks

ACTIVE_STATUSES = {"queued", "running", "waiting_for_input"}


class InMemorySessionRunDAO(SessionRunDAO):
    """In-memory реализация SessionRunDAO для unit-тестов без подключения к PostgreSQL."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.runs: dict[UUID, Run] = {}

    async def create_queued_run(
        self,
        session_id: str,
        run_id: UUID | None = None,
    ) -> Run:
        now = datetime.now(UTC)
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(
                id=session_id,
                status="active",
                created_at=now,
                updated_at=now,
            )
        for existing in self.runs.values():
            if existing.session_id == session_id and existing.status in ACTIVE_STATUSES:
                raise ActiveRunConflictError(session_id)

        actual_id = run_id if run_id is not None else uuid4()
        run = Run(
            id=actual_id,
            session_id=session_id,
            status="queued",
            reconcile_attempts=0,
            created_at=now,
            updated_at=now,
        )
        self.runs[actual_id] = run
        return run

    async def mark_failed_if_queued(self, run_id: UUID) -> bool:
        run = self.runs.get(run_id)
        if run is None or run.status != "queued":
            return False
        run.status = "failed"
        run.updated_at = datetime.now(UTC)
        return True

    async def claim_run_for_execution(self, run_id: UUID) -> bool:
        run = self.runs.get(run_id)
        if run is None or run.status != "queued":
            return False
        now = datetime.now(UTC)
        run.status = "running"
        run.last_heartbeat_at = now
        run.updated_at = now
        return True

    async def mark_run_completed(self, run_id: UUID) -> bool:
        run = self.runs.get(run_id)
        if run is None or run.status != "running":
            return False
        run.status = "completed"
        run.updated_at = datetime.now(UTC)
        return True

    async def mark_run_failed(self, run_id: UUID) -> bool:
        run = self.runs.get(run_id)
        if run is None or run.status not in ("queued", "running"):
            return False
        run.status = "failed"
        run.updated_at = datetime.now(UTC)
        return True

    async def requeue_running_run(self, run_id: UUID) -> bool:
        run = self.runs.get(run_id)
        if run is None or run.status != "running":
            return False
        run.status = "queued"
        run.updated_at = datetime.now(UTC)
        return True

    async def get_run(self, session_id: str, run_id: UUID) -> Run | None:
        run = self.runs.get(run_id)
        if run is None or run.session_id != session_id:
            return None
        return run


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


@pytest.fixture
def session_run_dao() -> InMemorySessionRunDAO:
    return InMemorySessionRunDAO()


@pytest.fixture(autouse=True)
def isolated_app(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    session_run_dao: InMemorySessionRunDAO,
) -> None:
    """Не читать настоящий .env и не оставлять app.state между тестами."""
    settings = Settings(_env_file=None, redis_url="redis://127.0.0.1:6379/15")
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    state = State()
    if not request.node.get_closest_marker("integration"):
        state.session_run_dao = session_run_dao

        class _BoundInMemoryDAO(InMemorySessionRunDAO):
            """Класс-прокси: остаётся типом для isinstance(), но при вызове возвращает фикстуру."""

            def __new__(cls, *args: object, **kwargs: object) -> InMemorySessionRunDAO:
                return session_run_dao

            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        monkeypatch.setattr(api_dependencies, "SessionRunDAO", _BoundInMemoryDAO)
        monkeypatch.setattr(api_router, "SessionRunDAO", _BoundInMemoryDAO)
        monkeypatch.setattr(worker_tasks, "SessionRunDAO", _BoundInMemoryDAO)
        monkeypatch.setattr(worker_settings, "SessionRunDAO", _BoundInMemoryDAO)
    monkeypatch.setattr(app, "state", state)


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
