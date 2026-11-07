"""Настоящие Redis и ARQ; только отдельная очередь и известные тестовые ключи."""

import asyncio
import importlib
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from arq import create_pool
from arq.connections import RedisSettings
from arq.constants import (
    health_check_key_suffix,
    in_progress_key_prefix,
    job_key_prefix,
    result_key_prefix,
    retry_key_prefix,
)
from arq.jobs import Job, JobStatus
from arq.worker import create_worker

from edu_doc_agent import main
from edu_doc_agent.api import router
from edu_doc_agent.config import Settings

pytestmark = pytest.mark.integration


def _redis_container_command(*arguments: str, missing_ok: bool = False) -> str:
    completed = subprocess.run(
        ["docker", *arguments], capture_output=True, text=True, timeout=30, check=False
    )
    if completed.returncode:
        if missing_ok and "No such container" in completed.stderr:
            return ""
        raise RuntimeError(
            f"Redis test container command failed: {arguments!r}\n{completed.stderr}"
        )
    return completed.stdout.strip()


def _start_redis_container(name: str, port: int | None = None) -> None:
    _redis_container_command(
        "run",
        "--detach",
        "--pull=never",
        "--name",
        name,
        "--publish",
        f"127.0.0.1:{port or ''}:6379",
        "redis:7-alpine",
        "redis-server",
        "--save",
        "",
        "--appendonly",
        "no",
    )


@pytest.fixture
def managed_redis() -> Iterator[tuple[str, int]]:
    """Отдельный Redis для fault-тестов; TEST_REDIS_URL не затрагивается."""
    name = f"edu-doc-recovery-{uuid4().hex}"
    try:
        _start_redis_container(name)
        address = _redis_container_command("port", name, "6379")
        yield name, int(address.rsplit(":", 1)[1])
    finally:
        _redis_container_command("rm", "--force", name, missing_ok=True)


async def _wait_for_managed_redis(port: int) -> None:
    settings = RedisSettings(host="127.0.0.1", port=port, conn_retries=5, conn_retry_delay=1)
    probe = await asyncio.wait_for(create_pool(settings), timeout=10)
    await probe.aclose(close_connection_pool=True)


@pytest.fixture
def redis_dsn() -> str:
    dsn = os.environ.get("TEST_REDIS_URL")
    if not dsn:
        pytest.skip("нужен TEST_REDIS_URL для отдельного тестового Redis")
    return dsn


@asynccontextmanager
async def queue(redis_dsn: str, monkeypatch: pytest.MonkeyPatch):
    name = f"edu-doc-agent:test:{uuid4().hex}"
    settings = RedisSettings.from_dsn(redis_dsn)
    settings.conn_retries = 0
    settings.conn_timeout = 1
    pool = await create_pool(settings, default_queue_name=name)
    identifiers = []

    def new_run_id():
        identifier = uuid4()
        identifiers.append(str(identifier))
        return identifier

    monkeypatch.setattr(router, "uuid4", new_run_id)
    try:
        yield pool, name, identifiers
    finally:
        # Не использовать FLUSHDB/FLUSHALL, SCAN по общим префиксам или рабочую очередь.
        keys = [name, name + health_check_key_suffix]
        for identifier in identifiers:
            keys.extend(
                prefix + identifier
                for prefix in (
                    job_key_prefix,
                    result_key_prefix,
                    retry_key_prefix,
                    in_progress_key_prefix,
                )
            )
        try:
            await pool.delete(*keys)
        finally:
            await pool.aclose(close_connection_pool=True)


def test_http_to_redis_to_worker(redis_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        settings = importlib.import_module("edu_doc_agent.worker.settings")
        tasks = importlib.import_module("edu_doc_agent.worker.tasks")
        sleep = AsyncMock()
        # Сохраняем реальный asyncio.sleep для ARQ; подменяем только ссылку в tasks.
        from types import SimpleNamespace

        monkeypatch.setattr(tasks, "asyncio", SimpleNamespace(sleep=sleep))
        async with queue(redis_dsn, monkeypatch) as (pool, name, _):
            factory = AsyncMock(return_value=pool)
            monkeypatch.setattr(main, "create_pool", factory)
            monkeypatch.setattr(
                main, "get_settings", lambda: Settings(_env_file=None, redis_url=redis_dsn)
            )
            async with main.lifespan(main.app):
                factory.assert_awaited_once_with(RedisSettings.from_dsn(redis_dsn))
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=main.app), base_url="http://test"
                ) as client:
                    response = await client.post("/api/v1/sessions/integration-session/process")
                assert response.status_code == 202
                body = response.json()
                job = Job(body["run_id"], pool, _queue_name=name)
                assert await job.status() == JobStatus.queued
                info = await job.info()
                assert info is not None
                assert info.function == "process_session_task"
                assert info.kwargs == {
                    "session_id": "integration-session",
                    "run_id": body["run_id"],
                }
                worker = create_worker(
                    settings.WorkerSettings,
                    redis_pool=pool,
                    queue_name=name,
                    burst=True,
                    max_burst_jobs=1,
                    poll_delay=0.01,
                    handle_signals=False,
                )
                try:
                    assert await asyncio.wait_for(worker.run_check(), timeout=5) == 1
                    assert await job.status() == JobStatus.complete
                    assert await job.result(timeout=1) == {
                        "status": "completed",
                        "session_id": "integration-session",
                        "run_id": body["run_id"],
                    }
                    sleep.assert_awaited_once_with(3)
                finally:
                    await worker.close()

    asyncio.run(scenario())


def test_local_api_and_arq_cli_process_a_session(
    redis_dsn: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Реальные отдельные процессы Uvicorn/ARQ, localhost HTTP и Redis."""

    async def scenario() -> None:
        async with queue(redis_dsn, monkeypatch) as (pool, name, identifiers):
            # Сохраняем точки входа приложения; меняется только имя тестовой очереди.
            (tmp_path / "local_api.py").write_text(
                "from functools import partial\n"
                "from arq import create_pool\n"
                "from edu_doc_agent import main\n"
                f"main.create_pool = partial(create_pool, default_queue_name={name!r})\n"
                "app = main.app\n"
            )
            (tmp_path / "local_worker.py").write_text(
                "from edu_doc_agent.worker.settings import WorkerSettings\n"
                f"WorkerSettings.queue_name = {name!r}\n"
            )
            environment = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
                "REDIS_URL": redis_dsn,
                "LOG_LEVEL": "INFO",
                "DATABASE_URL": "postgresql+psycopg://edu:synthetic@localhost:5432/edu",
            }
            processes = []
            api_log = tmp_path / "api.log"
            worker_log = tmp_path / "worker.log"
            with api_log.open("w") as api_output, worker_log.open("w") as worker_output:
                try:
                    processes.append(
                        subprocess.Popen(
                            [
                                sys.executable,
                                "-B",
                                "-m",
                                "uvicorn",
                                "local_api:app",
                                "--host",
                                "127.0.0.1",
                                "--port",
                                "0",
                            ],
                            cwd=tmp_path,
                            env=environment,
                            stdout=api_output,
                            stderr=subprocess.STDOUT,
                        )
                    )
                    processes.append(
                        subprocess.Popen(
                            [sys.executable, "-B", "-m", "arq", "local_worker.WorkerSettings"],
                            cwd=tmp_path,
                            env=environment,
                            stdout=worker_output,
                            stderr=subprocess.STDOUT,
                        )
                    )
                    async with asyncio.timeout(15):
                        while True:
                            if any(process.poll() is not None for process in processes):
                                raise RuntimeError("Local API or worker exited before the smoke")
                            address = re.search(r"http://127\.0\.0\.1:(\d+)", api_log.read_text())
                            if address:
                                break
                            await asyncio.sleep(0.05)
                    async with httpx.AsyncClient(
                        base_url=f"http://127.0.0.1:{address[1]}", trust_env=False
                    ) as client:
                        assert (await client.get("/health")).status_code == 200
                        assert (await client.get("/docs")).status_code == 200
                        response = await client.post("/api/v1/sessions/local-synthetic/process")
                    assert response.status_code == 202
                    body = response.json()
                    identifiers.append(body["run_id"])
                    result = await Job(body["run_id"], pool, _queue_name=name).result(
                        timeout=15, poll_delay=0.05
                    )
                    assert result == {
                        "status": "completed",
                        "session_id": "local-synthetic",
                        "run_id": body["run_id"],
                    }
                    assert body["run_id"] in worker_log.read_text()
                    assert all(process.poll() is None for process in processes)
                finally:
                    for process in processes:
                        process.terminate()
                    for process in processes:
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)

    asyncio.run(scenario())


# PR #29 P3: https://github.com/citec-spbu/edu-doc-agent/pull/29#discussion_r4478047239
@pytest.mark.pr_review
def test_active_session_rejects_a_second_run(
    redis_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        # Документированный контракт: один активный run на сессию.
        # Worker не запущен: первая задача остаётся в очереди до второго запроса.
        async with queue(redis_dsn, monkeypatch) as (pool, name, _):
            main.app.state.redis_pool = pool
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                first = await client.post("/api/v1/sessions/synthetic-duplicate/process")
                second = await client.post("/api/v1/sessions/synthetic-duplicate/process")
            assert first.status_code == 202
            assert second.status_code == 409
            assert isinstance(second.json()["detail"], str)
            assert "run_id" not in second.json()
            assert await pool.zcard(name) == 1
            first_job = await Job(first.json()["run_id"], pool, _queue_name=name).info()
            assert first_job is not None
            assert first_job.kwargs["session_id"] == "synthetic-duplicate"

    asyncio.run(scenario())


@pytest.mark.pr_review
@pytest.mark.parametrize(
    ("session_names", "expected_statuses"),
    [
        pytest.param(("same", "same"), [202, 409], id="same-session"),
        pytest.param(("first", "second"), [202, 202], id="different-sessions"),
    ],
)
def test_concurrent_requests_preserve_session_isolation(
    redis_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    session_names: tuple[str, str],
    expected_statuses: list[int],
) -> None:
    """Одна сессия допускает один run; разные сессии не блокируют друг друга."""

    async def scenario() -> None:
        suffix = uuid4().hex
        session_ids = [f"synthetic-{name}-{suffix}" for name in session_names]
        async with queue(redis_dsn, monkeypatch) as (pool, name, _):
            main.app.state.redis_pool = pool
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                ready = asyncio.Barrier(2)

                async def post(session_id: str) -> httpx.Response:
                    # Оба запроса готовы до начала обработки, без произвольных sleep.
                    await ready.wait()
                    return await client.post(f"/api/v1/sessions/{session_id}/process")

                responses = await asyncio.wait_for(
                    asyncio.gather(*(post(session_id) for session_id in session_ids)), timeout=10
                )

            # Worker не запущен: очередь отражает все принятые задачи.
            queued_ids = {value.decode() for value in await pool.zrange(name, 0, -1)}
            actual_statuses = sorted(response.status_code for response in responses)
            assert actual_statuses == expected_statuses, (
                f"statuses={actual_statuses}, queued_jobs={len(queued_ids)}"
            )
            accepted = [response for response in responses if response.status_code == 202]
            run_ids = {response.json()["run_id"] for response in accepted}
            assert len(run_ids) == expected_statuses.count(202)
            assert queued_ids == run_ids

            for session_id, response in zip(session_ids, responses, strict=True):
                body = response.json()
                if response.status_code == 409:
                    assert isinstance(body["detail"], str) and body["detail"]
                    assert "run_id" not in body
                    continue
                assert body["session_id"] == session_id
                assert body["status"] == "queued"
                job = await Job(body["run_id"], pool, _queue_name=name).info()
                assert job is not None
                assert job.kwargs == {"session_id": session_id, "run_id": body["run_id"]}

    asyncio.run(scenario())


@pytest.mark.parametrize("outage", ["stopped", "paused"])
def test_redis_recovers_after_outage_with_the_same_api_pool(
    managed_redis: tuple[str, int], monkeypatch: pytest.MonkeyPatch, outage: str
) -> None:
    """503 при отказе/молчании Redis и 202 после восстановления на том же адресе."""
    container, port = managed_redis

    async def scenario() -> None:
        await _wait_for_managed_redis(port)
        dsn = f"redis://127.0.0.1:{port}/15"
        async with queue(dsn, monkeypatch) as (pool, name, _):
            main.app.state.redis_pool = pool
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                before = await client.post("/api/v1/sessions/before-outage/process")
                assert before.status_code == 202
                before_info = await Job(before.json()["run_id"], pool, _queue_name=name).info()
                assert before_info is not None

                if outage == "paused":
                    _redis_container_command("pause", container)
                else:
                    _redis_container_command("stop", "--timeout", "1", container)
                try:
                    assert (await client.get("/health")).status_code == 200
                    failed = await asyncio.wait_for(
                        client.post("/api/v1/sessions/during-outage/process"), timeout=5
                    )
                    assert failed.status_code == 503
                    assert set(failed.json()) == {"detail"}
                    assert isinstance(failed.json()["detail"], str) and failed.json()["detail"]
                finally:
                    if outage == "paused":
                        _redis_container_command("unpause", container)
                    else:
                        _redis_container_command("rm", container)
                        _start_redis_container(container, port)

                await _wait_for_managed_redis(port)
                after = await client.post("/api/v1/sessions/after-outage/process")
                assert after.status_code == 202
                assert main.app.state.redis_pool is pool
                assert after.json()["run_id"] != before.json()["run_id"]
                after_info = await Job(after.json()["run_id"], pool, _queue_name=name).info()
                assert after_info is not None
                assert after_info.kwargs == {
                    "session_id": "after-outage",
                    "run_id": after.json()["run_id"],
                }
                # После timeout результат записи неизвестен: не требуем отсутствия
                # задания during-outage, которое Redis мог успеть принять.

    asyncio.run(scenario())
