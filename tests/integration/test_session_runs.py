"""Матрица #36: HTTP и Runner используют каноническое состояние настоящего PostgreSQL."""

import asyncio
import importlib
from collections.abc import Iterator
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from edu_doc_agent.api import router
from edu_doc_agent.api.dependencies import get_arq_pool, get_session_run_dao
from edu_doc_agent.dao import Run, Session, SessionRunDAO
from edu_doc_agent.main import app
from tests.integration.dao.support import ACTIVE_STATUSES, TERMINAL_STATUSES, Database

pytestmark = pytest.mark.integration
ALL_STATUSES = (*ACTIVE_STATUSES, *TERMINAL_STATUSES)


@pytest.fixture
def run_sessions(migrated_database: Database) -> Iterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(migrated_database.url, poolclass=NullPool)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        asyncio.run(engine.dispose())


@pytest.fixture
def run_dao(run_sessions: async_sessionmaker[AsyncSession]) -> SessionRunDAO:
    return SessionRunDAO(run_sessions)


@pytest.fixture
def runs_api(
    run_dao: SessionRunDAO,
    redis_pool: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> httpx.AsyncClient:
    monkeypatch.setitem(app.dependency_overrides, get_session_run_dao, lambda: run_dao)
    app.state.redis_pool = redis_pool
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=app, raise_app_exceptions=getattr(request, "param", True)
        ),
        base_url="http://test",
    )


async def seed_run(
    sessions: async_sessionmaker[AsyncSession],
    dao: SessionRunDAO,
    status: str = "queued",
    session_id: str = "synthetic-session",
) -> Run:
    run = await dao.create_queued_run(session_id)
    if status != "queued":
        async with sessions() as db:
            await db.execute(update(Run).where(Run.id == run.id).values(status=status))
            await db.commit()
    return await read_run(sessions, run.id)


async def read_run(sessions: async_sessionmaker[AsyncSession], run_id: UUID) -> Run:
    async with sessions() as db:
        run = await db.get(Run, run_id)
        assert run is not None
        return run


async def counts(sessions: async_sessionmaker[AsyncSession]) -> tuple[int, int]:
    async with sessions() as db:
        return (
            await db.scalar(select(func.count()).select_from(Session)),
            await db.scalar(select(func.count()).select_from(Run)),
        )


def test_post_commits_session_and_run_before_enqueue(
    runs_api: httpx.AsyncClient,
    run_sessions: async_sessionmaker[AsyncSession],
    redis_pool: AsyncMock,
) -> None:
    async def enqueue(function: str, **kwargs):
        assert function == "process_session_task"
        assert kwargs["_job_id"] == kwargs["run_id"]
        async with run_sessions() as db:
            session = await db.get(Session, kwargs["session_id"])
            run = await db.get(Run, UUID(kwargs["run_id"]))
            assert session is not None and session.status == "active"
            assert run is not None and run.session_id == session.id and run.status == "queued"
        return object()

    redis_pool.enqueue_job.side_effect = enqueue

    async def scenario() -> None:
        async with runs_api as client:
            response = await client.post("/api/v1/sessions/synthetic-session/process")
        assert response.status_code == 202
        body = response.json()
        assert set(body) == {"run_id", "session_id", "status"}
        assert body["status"] == "queued" and body["session_id"] == "synthetic-session"
        run = await read_run(run_sessions, UUID(body["run_id"]))
        assert run.status == "queued"
        assert await counts(run_sessions) == (1, 1)
        redis_pool.enqueue_job.assert_awaited_once()

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_concurrent_posts_create_one_session_one_active_run_and_one_job(
    runs_api: httpx.AsyncClient,
    run_dao: SessionRunDAO,
    run_sessions: async_sessionmaker[AsyncSession],
    redis_pool: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        original = run_dao.create_queued_run
        arrived = 0
        both_requests = asyncio.Event()

        async def start_together(*args, **kwargs):
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                both_requests.set()
            await both_requests.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(run_dao, "create_queued_run", start_together)
        async with runs_api as client:
            responses = await asyncio.gather(
                client.post("/api/v1/sessions/new-session/process"),
                client.post("/api/v1/sessions/new-session/process"),
            )
        assert sorted(response.status_code for response in responses) == [202, 409]
        assert await counts(run_sessions) == (1, 1)
        accepted = next(response.json() for response in responses if response.status_code == 202)
        run = await read_run(run_sessions, UUID(accepted["run_id"]))
        assert run.status == "queued" and run.session_id == "new-session"
        redis_pool.enqueue_job.assert_awaited_once_with(
            "process_session_task",
            session_id=run.session_id,
            run_id=str(run.id),
            _job_id=str(run.id),
        )

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("status", ALL_STATUSES)
def test_post_respects_active_and_terminal_runs(
    runs_api: httpx.AsyncClient,
    run_dao: SessionRunDAO,
    run_sessions: async_sessionmaker[AsyncSession],
    redis_pool: AsyncMock,
    status: str,
) -> None:
    async def scenario() -> None:
        previous = await seed_run(run_sessions, run_dao, status)
        async with runs_api as client:
            response = await client.post(f"/api/v1/sessions/{previous.session_id}/process")
        unchanged = await read_run(run_sessions, previous.id)
        assert (unchanged.status, unchanged.updated_at) == (status, previous.updated_at)
        if status in ACTIVE_STATUSES:
            assert response.status_code == 409
            assert await counts(run_sessions) == (1, 1)
            redis_pool.enqueue_job.assert_not_awaited()
        else:
            assert response.status_code == 202
            assert UUID(response.json()["run_id"]) != previous.id
            assert await counts(run_sessions) == (1, 2)
            redis_pool.enqueue_job.assert_awaited_once()

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("runs_api", [False], indirect=True)
def test_unrelated_integrity_error_is_not_a_409(
    runs_api: httpx.AsyncClient,
    run_dao: SessionRunDAO,
    run_sessions: async_sessionmaker[AsyncSession],
    redis_pool: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        previous = await seed_run(run_sessions, run_dao, "completed")
        monkeypatch.setattr(router, "uuid4", lambda: previous.id)
        async with runs_api as client:
            response = await client.post("/api/v1/sessions/another-session/process")
        assert response.status_code == 500
        assert await counts(run_sessions) == (1, 1)  # Upsert Session тоже откатился.
        assert (await read_run(run_sessions, previous.id)).status == "completed"
        redis_pool.enqueue_job.assert_not_awaited()

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize(
    ("failure", "expected_http"),
    [("timeout", 503), ("redis", 503), ("none", 503), ("internal", 500), ("deadline", 503)],
)
def test_enqueue_failure_is_compensated_and_allows_a_new_run(
    runs_api: httpx.AsyncClient,
    run_sessions: async_sessionmaker[AsyncSession],
    redis_pool: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_http: int,
) -> None:
    async def scenario() -> None:
        run_ids = []

        async def enqueue(*args, **kwargs):
            run_ids.append(UUID(kwargs["run_id"]))
            assert (await read_run(run_sessions, run_ids[-1])).status == "queued"
            if failure == "deadline":
                await asyncio.Event().wait()
            errors = {
                "timeout": TimeoutError("synthetic timeout"),
                "redis": RedisConnectionError("synthetic redis failure"),
                "internal": RuntimeError("synthetic internal failure"),
            }
            if failure in errors:
                raise errors[failure]
            return None

        if failure == "deadline":
            monkeypatch.setattr(router, "ENQUEUE_DEADLINE_SECONDS", 0.05)
        redis_pool.enqueue_job.side_effect = enqueue
        async with runs_api as client:
            response = await client.post("/api/v1/sessions/failed-enqueue/process")
            assert response.status_code == expected_http
            redis_pool.enqueue_job.assert_awaited_once()
            run = await read_run(run_sessions, run_ids[0])
            assert run.status == "failed" and run.updated_at >= run.created_at
            redis_pool.enqueue_job.side_effect = None
            retry = await client.post("/api/v1/sessions/failed-enqueue/process")
        assert retry.status_code == 202
        assert UUID(retry.json()["run_id"]) != run.id
        assert await counts(run_sessions) == (1, 2)

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("winner", ["compensation", "running", "completed"])
def test_compensation_and_worker_preserve_the_winning_transition(
    runs_api: httpx.AsyncClient,
    run_dao: SessionRunDAO,
    run_sessions: async_sessionmaker[AsyncSession],
    redis_pool: AsyncMock,
    winner: str,
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        started, release = asyncio.Event(), asyncio.Event()
        worker = None
        run_id = None

        async def business(session_id: str) -> None:
            started.set()
            await release.wait()

        logic = AsyncMock(side_effect=business)

        async def lost_ack(*args, **kwargs):
            nonlocal worker, run_id
            run_id = UUID(kwargs["run_id"])
            if winner != "compensation":
                worker = asyncio.create_task(
                    runner_class(run_dao).execute(kwargs["session_id"], run_id, logic)
                )
                await started.wait()
                if winner == "completed":
                    release.set()
                    await worker
            return None

        redis_pool.enqueue_job.side_effect = lost_ack
        try:
            async with runs_api as client:
                response = await client.post("/api/v1/sessions/race-session/process")
            assert response.status_code == 503
            assert run_id is not None
            expected_status = "failed" if winner == "compensation" else winner
            assert (await read_run(run_sessions, run_id)).status == expected_status
            redis_pool.enqueue_job.assert_awaited_once()
            if winner == "compensation":
                result = await runner_class(run_dao).execute("race-session", run_id, logic)
                assert result["status"] == "skipped"
                logic.assert_not_awaited()
            else:
                release.set()
                assert (await worker)["status"] == "completed"
                assert (await read_run(run_sessions, run_id)).status == "completed"
                logic.assert_awaited_once_with("race-session")
        finally:
            release.set()
            if worker is not None:
                await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_two_executions_of_one_run_call_business_once(
    run_dao: SessionRunDAO, run_sessions: async_sessionmaker[AsyncSession]
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        run = await seed_run(run_sessions, run_dao)
        release = asyncio.Event()

        async def business(session_id: str) -> None:
            await release.wait()

        logic = AsyncMock(side_effect=business)
        runner = runner_class(run_dao)
        tasks = [
            asyncio.create_task(runner.execute(run.session_id, run.id, logic)) for _ in range(2)
        ]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            assert len(done) == len(pending) == 1
            assert next(iter(done)).result()["status"] == "skipped"
            assert (await read_run(run_sessions, run.id)).status == "running"
        finally:
            release.set()
            results = await asyncio.gather(*tasks)
        assert sorted(result["status"] for result in results) == ["completed", "skipped"]
        logic.assert_awaited_once_with(run.session_id)
        assert (await read_run(run_sessions, run.id)).status == "completed"

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_parallel_runs_release_db_connection_and_isolate_failure(
    migrated_database: Database,
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        engine = create_async_engine(
            migrated_database.url, pool_size=1, max_overflow=0, pool_timeout=1
        )
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        dao = SessionRunDAO(sessions)
        started = {key: asyncio.Event() for key in ("first", "second")}
        release = {key: asyncio.Event() for key in started}
        error = RuntimeError("synthetic first-run failure")
        tasks = []

        async def business(session_id: str) -> None:
            started[session_id].set()
            await release[session_id].wait()
            if session_id == "first":
                raise error

        try:
            runs = {key: await seed_run(sessions, dao, session_id=key) for key in started}
            tasks = [
                asyncio.create_task(runner_class(dao).execute(key, run.id, business))
                for key, run in runs.items()
            ]
            await asyncio.gather(*(event.wait() for event in started.values()))
            # Единственное соединение доступно, пока обе бизнес-функции приостановлены.
            for run in runs.values():
                assert (await read_run(sessions, run.id)).status == "running"
            release["first"].set()
            with pytest.raises(RuntimeError) as caught:
                await tasks[0]
            assert caught.value is error
            # Независимое чтение после исключения доказывает commit failed до его выхода.
            assert (await read_run(sessions, runs["first"].id)).status == "failed"
            assert (await read_run(sessions, runs["second"].id)).status == "running"
            release["second"].set()
            assert (await tasks[1])["status"] == "completed"
            assert (await read_run(sessions, runs["second"].id)).status == "completed"
        finally:
            for event in release.values():
                event.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await engine.dispose()

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("status", [*ALL_STATUSES[1:], "missing"])
def test_nonqueued_or_missing_run_skips_business(
    run_dao: SessionRunDAO, run_sessions: async_sessionmaker[AsyncSession], status: str
) -> None:
    runner_class = importlib.import_module("edu_doc_agent.worker.runner").SessionRunner

    async def scenario() -> None:
        run = None if status == "missing" else await seed_run(run_sessions, run_dao, status)
        run_id = uuid4() if run is None else run.id
        logic = AsyncMock()
        result = await runner_class(run_dao).execute("synthetic-session", run_id, logic)
        assert result == {
            "status": "skipped",
            "session_id": "synthetic-session",
            "run_id": str(run_id),
        }
        logic.assert_not_awaited()
        if run is not None:
            unchanged = await read_run(run_sessions, run.id)
            assert (unchanged.status, unchanged.updated_at) == (status, run.updated_at)

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("status", ALL_STATUSES)
def test_get_returns_database_status_identifiers_and_timestamps_without_redis(
    runs_api: httpx.AsyncClient,
    run_dao: SessionRunDAO,
    run_sessions: async_sessionmaker[AsyncSession],
    redis_pool: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    def forbidden_redis_dependency():
        pytest.fail("GET статуса не должен получать Redis dependency")

    monkeypatch.setitem(app.dependency_overrides, get_arq_pool, forbidden_redis_dependency)

    async def scenario() -> None:
        run = await seed_run(run_sessions, run_dao, status)
        async with runs_api as client:
            response = await client.get(f"/api/v1/sessions/{run.session_id}/runs/{run.id}")
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"run_id", "session_id", "status", "created_at", "updated_at"}
        assert (UUID(body["run_id"]), body["session_id"], body["status"]) == (
            run.id,
            run.session_id,
            status,
        )
        assert datetime.fromisoformat(body["created_at"]) == run.created_at
        assert datetime.fromisoformat(body["updated_at"]) == run.updated_at
        assert not redis_pool.mock_calls

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("case", ["missing", "wrong-session"])
def test_get_returns_404_for_missing_run_or_wrong_session(
    runs_api: httpx.AsyncClient,
    run_dao: SessionRunDAO,
    run_sessions: async_sessionmaker[AsyncSession],
    redis_pool: AsyncMock,
    case: str,
) -> None:
    async def scenario() -> None:
        run = await seed_run(run_sessions, run_dao)
        session_id = "another-session" if case == "wrong-session" else run.session_id
        run_id = uuid4() if case == "missing" else run.id
        async with runs_api as client:
            response = await client.get(f"/api/v1/sessions/{session_id}/runs/{run_id}")
        assert response.status_code == 404
        assert (await read_run(run_sessions, run.id)).status == "queued"
        assert not redis_pool.mock_calls

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("original_status", ["running", "queued"])
@pytest.mark.parametrize("failure", ["redis", "timeout", "internal", "none"])
def test_reconcile_enqueue_failure_keeps_old_worker_fenced(
    run_dao: SessionRunDAO,
    run_sessions: async_sessionmaker[AsyncSession],
    redis_pool: AsyncMock,
    original_status: str,
    failure: str,
) -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler

    async def scenario() -> None:
        run = await run_dao.create_queued_run("synthetic-session")
        if original_status == "running":
            assert await run_dao.claim_run_for_execution(run.id) == 0
        async with run_sessions() as db:
            stale = func.clock_timestamp() - timedelta(minutes=5)
            await db.execute(
                update(Run)
                .where(Run.id == run.id)
                .values(updated_at=stale, last_heartbeat_at=stale)
            )
            await db.commit()

        redis_pool.ping = AsyncMock()
        redis_pool.zscore = AsyncMock(return_value=None)
        errors = {
            "redis": RedisConnectionError("synthetic outage"),
            "timeout": TimeoutError("synthetic timeout"),
            "internal": RuntimeError("synthetic enqueue failure"),
        }
        redis_pool.enqueue_job.side_effect = errors.get(failure)
        redis_pool.enqueue_job.return_value = None
        report = await reconciler_class(dao=run_dao, redis=redis_pool).reconcile()
        recovered = await read_run(run_sessions, run.id)
        assert (report.requeued_count, report.enqueued_count) == (1, 0)
        assert (recovered.status, recovered.reconcile_attempts) == ("queued", 1)
        assert recovered.reconcile_pending is True

        assert not await run_dao.mark_run_failed(run.id, expected_reconcile_attempts=0)
        assert not await run_dao.requeue_running_run(run.id, expected_reconcile_attempts=0)
        assert await run_dao.claim_run_for_execution(run.id) is None
        assert await run_dao.claim_run_for_execution(run.id, expected_reconcile_attempts=1) == 1
        assert (await read_run(run_sessions, run.id)).reconcile_pending is False
        assert not await run_dao.touch_heartbeat(run.id, expected_reconcile_attempts=0)
        assert not await run_dao.mark_run_completed(run.id, expected_reconcile_attempts=0)
        assert not await run_dao.mark_run_failed(run.id, expected_reconcile_attempts=0)
        assert await run_dao.mark_run_completed(run.id, expected_reconcile_attempts=1)

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("job_generation", [0, 1])
def test_old_arq_job_cannot_claim_a_new_generation_on_last_try(
    run_dao: SessionRunDAO,
    run_sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
    job_generation: int,
) -> None:
    tasks = importlib.import_module("edu_doc_agent.worker.tasks")

    async def scenario() -> None:
        run = await run_dao.create_queued_run("synthetic-session")
        for generation in range(2):
            assert (
                await run_dao.bump_stale_queued_run(run.id, expected_reconcile_attempts=generation)
                is not None
            )
            assert await run_dao.confirm_reconcile_enqueue(
                run.id, expected_reconcile_attempts=generation + 1
            )

        logic = AsyncMock(side_effect=asyncio.CancelledError)
        monkeypatch.setattr(tasks, "_stub_session_logic", logic)
        payload = {"expected_reconcile_attempts": job_generation} if job_generation else {}
        result = await tasks.process_session_task(
            {"dao": run_dao, "job_try": 3}, run.session_id, str(run.id), **payload
        )
        assert result["status"] == "skipped"
        logic.assert_not_awaited()
        unchanged = await read_run(run_sessions, run.id)
        assert (unchanged.status, unchanged.reconcile_attempts) == ("queued", 2)

        logic.side_effect = None
        result = await tasks.process_session_task(
            {"dao": run_dao, "job_try": 1},
            run.session_id,
            str(run.id),
            expected_reconcile_attempts=2,
        )
        assert result["status"] == "completed"
        assert (await read_run(run_sessions, run.id)).status == "completed"

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_cancelled_cron_preserves_pending_delivery_without_exhausting_budget(
    run_dao: SessionRunDAO,
    run_sessions: async_sessionmaker[AsyncSession],
    redis_pool: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler

    async def scenario() -> None:
        run = await seed_run(run_sessions, run_dao, "running")
        async with run_sessions() as db:
            await db.execute(
                update(Run)
                .where(Run.id == run.id)
                .values(last_heartbeat_at=func.clock_timestamp() - timedelta(minutes=5))
            )
            await db.commit()
        redis_pool.ping = AsyncMock()
        redis_pool.zscore = AsyncMock(return_value=None)
        original = run_dao.list_stale_queued_runs
        monkeypatch.setattr(
            run_dao, "list_stale_queued_runs", AsyncMock(side_effect=asyncio.CancelledError)
        )
        for _ in range(3):
            with pytest.raises(asyncio.CancelledError):
                await reconciler_class(dao=run_dao, redis=redis_pool).reconcile()
            stored = await read_run(run_sessions, run.id)
            assert (stored.status, stored.reconcile_attempts, stored.reconcile_pending) == (
                "queued",
                1,
                True,
            )
        redis_pool.enqueue_job.assert_not_awaited()
        monkeypatch.setattr(run_dao, "list_stale_queued_runs", original)
        report = await reconciler_class(
            dao=run_dao, redis=redis_pool, stale_queued_seconds=0
        ).reconcile()
        assert report.enqueued_count == 1
        stored = await read_run(run_sessions, run.id)
        assert (stored.reconcile_attempts, stored.reconcile_pending) == (1, False)

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_pending_last_delivery_cannot_be_failed_or_replaced(
    run_dao: SessionRunDAO,
    run_sessions: async_sessionmaker[AsyncSession],
) -> None:
    async def scenario() -> None:
        run = await run_dao.create_queued_run("synthetic-session")
        async with run_sessions() as db:
            await db.execute(update(Run).where(Run.id == run.id).values(reconcile_attempts=2))
            await db.commit()
        pending = await run_dao.bump_stale_queued_run(run.id, expected_reconcile_attempts=2)
        assert pending is not None and pending.reconcile_pending
        assert pending.reconcile_attempts == 3
        assert not await run_dao.fail_stale_queued_run(run.id, expected_reconcile_attempts=3)
        assert await run_dao.bump_stale_queued_run(run.id, expected_reconcile_attempts=3) is None
        assert not await run_dao.confirm_reconcile_enqueue(run.id, expected_reconcile_attempts=2)
        assert await run_dao.confirm_reconcile_enqueue(run.id, expected_reconcile_attempts=3)
        assert not await run_dao.confirm_reconcile_enqueue(run.id, expected_reconcile_attempts=3)
        assert await run_dao.fail_stale_queued_run(run.id, expected_reconcile_attempts=3)

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_reconciler_reaches_lost_run_after_a_full_page_of_existing_jobs(
    run_dao: SessionRunDAO,
    run_sessions: async_sessionmaker[AsyncSession],
    redis_pool: AsyncMock,
) -> None:
    reconciler_class = importlib.import_module("edu_doc_agent.worker.reconciler").RunReconciler

    async def scenario() -> None:
        ids = [UUID(int=i + 1) for i in range(101)]
        async with run_sessions() as db:
            db.add_all([Session(id=f"synthetic-{i}", status="active") for i in range(101)])
            await db.flush()
            db.add_all(
                [
                    Run(id=identifier, session_id=f"synthetic-{i}", status="queued")
                    for i, identifier in enumerate(ids)
                ]
            )
            await db.flush()
            await db.execute(
                update(Run).values(updated_at=func.clock_timestamp() - timedelta(minutes=5))
            )
            await db.commit()
        existing_jobs = {str(identifier) for identifier in ids[:100]}
        redis_pool.ping = AsyncMock()

        async def score(queue_name, job_id):
            return 1 if job_id in existing_jobs else None

        redis_pool.zscore = AsyncMock(side_effect=score)
        first = await reconciler_class(
            dao=run_dao, redis=redis_pool, stale_queued_seconds=0
        ).reconcile()
        assert first.enqueued_count == 0
        second = await reconciler_class(
            dao=run_dao, redis=redis_pool, stale_queued_seconds=0
        ).reconcile()
        assert second.enqueued_count == 1
        assert redis_pool.enqueue_job.call_args.kwargs["run_id"] == str(ids[100])

    asyncio.run(asyncio.wait_for(scenario(), 30))
