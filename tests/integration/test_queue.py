"""Настоящие Redis и ARQ; только отдельная очередь и известные тестовые ключи."""

import asyncio
import importlib
import os
import re
import subprocess
import sys
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
def test_duplicate_job_id_is_not_reported_as_a_new_run(
    redis_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with queue(redis_dsn, monkeypatch) as (pool, name, identifiers):
            fixed_id = uuid4()
            identifiers.append(str(fixed_id))
            monkeypatch.setattr(router, "uuid4", lambda: fixed_id)
            main.app.state.redis_pool = pool
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                first = await client.post("/api/v1/sessions/synthetic-duplicate/process")
                second = await client.post("/api/v1/sessions/synthetic-duplicate/process")
            assert first.status_code == 202
            assert await pool.zcard(name) == 1
            assert second.status_code == 409
            assert isinstance(second.json()["detail"], str)

    asyncio.run(scenario())
