"""Контракт HTTP → очередь, включая регрессии из PR #29."""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from edu_doc_agent.main import app


@pytest.mark.parametrize("session_id", ["test-123", "сессия-ОП", "arbitrary.string"])
def test_process_enqueues_job_with_response_identifiers(
    client: TestClient, redis_pool: AsyncMock, session_id: str
) -> None:
    response = client.post(f"/api/v1/sessions/{session_id}/process")

    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"run_id", "session_id", "status"}
    assert UUID(body["run_id"]).version == 4
    assert body["session_id"] == session_id
    assert body["status"] == "queued"
    redis_pool.enqueue_job.assert_awaited_once_with(
        "process_session_task",
        session_id=session_id,
        run_id=body["run_id"],
        _job_id=body["run_id"],
    )


def test_repeated_requests_create_distinct_runs(client: TestClient, redis_pool: AsyncMock) -> None:
    responses = [client.post("/api/v1/sessions/same-session/process") for _ in range(2)]

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
def test_enqueue_conflict_does_not_claim_a_new_job(
    client: TestClient, redis_pool: AsyncMock
) -> None:
    redis_pool.enqueue_job.return_value = None

    response = client.post("/api/v1/sessions/conflict/process")

    assert response.status_code == 409
    assert isinstance(response.json()["detail"], str)
    redis_pool.enqueue_job.assert_awaited_once()


@pytest.mark.parametrize("error", [RedisConnectionError("synthetic failure"), TimeoutError()])
# PR #29 P2: https://github.com/citec-spbu/edu-doc-agent/pull/29#discussion_r4478029967
@pytest.mark.pr_review
def test_queue_errors_have_a_service_error_response(
    redis_pool: AsyncMock, error: Exception
) -> None:
    app.state.redis_pool = redis_pool
    redis_pool.enqueue_job.side_effect = error
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/sessions/unavailable/process")

    assert response.status_code in {503, 504}
    assert isinstance(response.json()["detail"], str)
    redis_pool.enqueue_job.assert_awaited_once()


# PR #29 P2: https://github.com/citec-spbu/edu-doc-agent/pull/29#discussion_r4478029967
@pytest.mark.pr_review
def test_silent_redis_has_a_deadline_without_blind_retry(redis_pool: AsyncMock) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        recorded_jobs = []

        async def written_but_no_reply(*args, **kwargs):
            recorded_jobs.append(kwargs["_job_id"])
            started.set()
            await asyncio.Event().wait()

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
                # Защитный предел теста; production deadline должен быть меньше него.
                done, _ = await asyncio.wait({request}, timeout=2)
                assert done, "enqueue_job must have a server-side deadline below 2 seconds"
                response = request.result()
                assert response.status_code in {503, 504}
                assert isinstance(response.json()["detail"], str)
                assert len(recorded_jobs) == 1
                redis_pool.enqueue_job.assert_awaited_once()
            finally:
                request.cancel()
                with suppress(asyncio.CancelledError):
                    await request

    asyncio.run(scenario())


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
