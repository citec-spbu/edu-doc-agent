"""Контракт HTTP → очередь, включая регрессии из PR #29."""

import asyncio
import logging
from contextlib import suppress
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError, ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from edu_doc_agent.api import router as api_router
from edu_doc_agent.main import app


@pytest.fixture
def fast_enqueue_timeout(monkeypatch: pytest.MonkeyPatch) -> float:
    """Ускоряем продуктовый deadline постановки в очередь: 0.05 с вместо 2.0 с."""
    deadline = 0.05
    # Меняется только число в роуте; таймеры HTTP-клиента и самого теста не затронуты.
    monkeypatch.setattr(api_router, "ENQUEUE_DEADLINE_SECONDS", deadline)
    return deadline


@pytest.mark.parametrize("session_id", ["test-123", "сессия-ОП", "arbitrary.string"])
def test_process_enqueues_job_with_response_identifiers(
    client: TestClient, redis_pool: AsyncMock, session_id: str
) -> None:
    response = client.post(f"/api/v1/sessions/{session_id}/process")

    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"run_id", "session_id", "status"}
    UUID(body["run_id"])  # Контракт требует корректный UUID без ограничения версии.
    assert body["session_id"] == session_id
    assert body["status"] == "queued"
    redis_pool.enqueue_job.assert_awaited_once_with(
        "process_session_task",
        session_id=session_id,
        run_id=body["run_id"],
        _job_id=body["run_id"],
    )


def test_different_sessions_create_distinct_runs(client: TestClient, redis_pool: AsyncMock) -> None:
    responses = [
        client.post(f"/api/v1/sessions/{session_id}/process")
        for session_id in ("first-session", "second-session")
    ]

    assert [response.status_code for response in responses] == [202, 202]
    ids = [response.json()["run_id"] for response in responses]
    assert ids[0] != ids[1]
    assert redis_pool.enqueue_job.await_count == 2
    assert [call.kwargs["_job_id"] for call in redis_pool.enqueue_job.await_args_list] == ids


def test_response_does_not_wait_for_worker(client: TestClient, redis_pool: AsyncMock) -> None:
    # Объект задания не обязан предоставлять результат во время HTTP-запроса.
    job = redis_pool.enqueue_job.return_value
    job.result = AsyncMock(side_effect=AssertionError("API must not wait for the worker"))

    assert client.post("/api/v1/sessions/pending/process").status_code == 202
    job.result.assert_not_awaited()


# PR #29 P3: https://github.com/citec-spbu/edu-doc-agent/pull/29#discussion_r4478047239
@pytest.mark.pr_review
def test_unconfirmed_enqueue_does_not_claim_a_new_job(
    client: TestClient, redis_pool: AsyncMock
) -> None:
    redis_pool.enqueue_job.return_value = None

    response = client.post("/api/v1/sessions/unconfirmed/process")

    assert response.status_code == 503
    assert set(response.json()) == {"detail"}
    assert isinstance(response.json()["detail"], str)
    redis_pool.enqueue_job.assert_awaited_once()


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        pytest.param(TimeoutError("synthetic-private-details"), 503, id="python-timeout"),
        pytest.param(RedisConnectionError("synthetic-private-details"), 503, id="redis-connection"),
        pytest.param(RedisTimeoutError("synthetic-private-details"), 503, id="redis-timeout"),
        pytest.param(RedisError("synthetic-private-details"), 503, id="redis-base-error"),
        pytest.param(ResponseError("synthetic-private-details"), 503, id="redis-response-error"),
        pytest.param(RuntimeError("synthetic-private-details"), 500, id="internal-error"),
    ],
)
# PR #29 P2: https://github.com/citec-spbu/edu-doc-agent/pull/29#discussion_r4478029967
@pytest.mark.pr_review
def test_queue_errors_have_a_service_error_response(
    redis_pool: AsyncMock, error: Exception, expected_status: int
) -> None:
    app.state.redis_pool = redis_pool
    redis_pool.enqueue_job.side_effect = error
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/sessions/unavailable/process")

    assert response.status_code == expected_status
    body = response.json()
    assert set(body) == {"detail"}
    assert isinstance(body["detail"], str) and body["detail"]
    assert str(error) not in response.text
    redis_pool.enqueue_job.assert_awaited_once()


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        pytest.param(RedisError("synthetic redis failure"), 503, id="redis"),
        pytest.param(RuntimeError("synthetic internal failure"), 500, id="internal"),
    ],
)
def test_queue_error_logs_include_session_and_cause(
    client: TestClient,
    redis_pool: AsyncMock,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    expected_status: int,
) -> None:
    redis_pool.enqueue_job.side_effect = error

    with caplog.at_level(logging.ERROR, logger=api_router.__name__):
        response = client.post("/api/v1/sessions/logged-session/process")

    assert response.status_code == expected_status
    records = [record for record in caplog.records if record.name == api_router.__name__]
    assert any(
        record.levelno == logging.ERROR
        and "logged-session" in record.getMessage()
        and str(error) in record.getMessage()
        for record in records
    )
    assert str(error) not in response.text


# PR #29 P2: https://github.com/citec-spbu/edu-doc-agent/pull/29#discussion_r4478029967
@pytest.mark.pr_review
def test_silent_redis_has_a_deadline_without_blind_retry(
    redis_pool: AsyncMock, fast_enqueue_timeout: float
) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()
        recorded_jobs = []

        async def written_but_no_reply(*args, **kwargs):
            recorded_jobs.append(kwargs["_job_id"])
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        redis_pool.enqueue_job.side_effect = written_but_no_reply
        app.state.redis_pool = redis_pool
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            request = asyncio.create_task(client.post("/api/v1/sessions/silent/process"))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                health = await client.get("/health")
                assert health.status_code == 200
                # Watchdog с запасом относительно короткого тестового таймера.
                done, _ = await asyncio.wait({request}, timeout=2)
                assert done, "enqueue_job did not stop after its deadline"
                response = request.result()
                assert response.status_code == 503
                assert set(response.json()) == {"detail"}
                assert isinstance(response.json()["detail"], str) and response.json()["detail"]
                await asyncio.wait_for(cancelled.wait(), timeout=1)
                assert len(recorded_jobs) == 1
                redis_pool.enqueue_job.assert_awaited_once()
            finally:
                request.cancel()
                with suppress(asyncio.CancelledError):
                    await request

    asyncio.run(scenario())


def test_enqueue_deadline_default() -> None:
    assert api_router.ENQUEUE_DEADLINE_SECONDS == 2.0


def test_external_cancellation_reaches_enqueue_without_error_response_or_retry(
    redis_pool: AsyncMock, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def pending_enqueue(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        redis_pool.enqueue_job.side_effect = pending_enqueue
        request = asyncio.create_task(api_router.process_session("cancelled-session", redis_pool))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert request.cancelled()
            assert cancelled.is_set()
            redis_pool.enqueue_job.assert_awaited_once()
        finally:
            request.cancel()
            with suppress(asyncio.CancelledError):
                await request

    with caplog.at_level(logging.ERROR, logger=api_router.__name__):
        asyncio.run(scenario())
    assert not [record for record in caplog.records if record.name == api_router.__name__]


def test_openapi_exposes_process_contract(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/sessions/{session_id}/process"]["post"]

    assert operation["responses"]["202"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ProcessResponse"
    }
    response_schema = schema["components"]["schemas"]["ProcessResponse"]
    assert response_schema["properties"]["run_id"]["format"] == "uuid"
    assert {"run_id", "session_id"} <= set(response_schema["required"])
    assert operation["parameters"][0]["name"] == "session_id"
    assert operation["parameters"][0]["required"] is True


def test_pool_dependency_returns_the_application_pool(redis_pool: AsyncMock) -> None:
    from starlette.requests import Request

    from edu_doc_agent.api.dependencies import get_arq_pool

    app.state.redis_pool = redis_pool
    assert get_arq_pool(Request({"type": "http", "app": app})) is redis_pool


def test_get_does_not_enqueue_a_job(client: TestClient, redis_pool: AsyncMock) -> None:
    assert client.get("/api/v1/sessions/test/process").status_code == 405
    redis_pool.enqueue_job.assert_not_awaited()
