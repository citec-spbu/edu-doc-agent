"""Настоящие Redis и ARQ; только отдельная очередь и известные тестовые ключи."""

import asyncio
import importlib
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from types import SimpleNamespace
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
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from redis.exceptions import ConnectionError as RedisConnectionError

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


@pytest.mark.pr_review
@pytest.mark.parametrize(
    ("session_names", "expected_statuses"),
    [
        pytest.param(("first", "second"), [202, 202], id="different-sessions"),
    ],
)
def test_concurrent_requests_for_different_sessions(
    redis_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    session_names: tuple[str, str],
    expected_statuses: list[int],
) -> None:
    """Конкурентные запросы разных сессий создают отдельные задания."""

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


@pytest.mark.pr_review
def test_enqueue_deadline_holds_when_redis_hangs_after_watch(
    managed_redis: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redis замолкает после настоящего WATCH; cleanup не продлевает HTTP-дедлайн."""
    _, port = managed_redis
    pause_ms = 6000
    max_response_seconds = 3.0  # дедлайн постановки 2.0 + запас

    async def scenario() -> None:
        await _wait_for_managed_redis(port)
        dsn = f"redis://127.0.0.1:{port}/15"
        admin = Redis.from_url(dsn)
        injected: list[bool] = []
        original_exists = Pipeline.exists

        async def exists_after_pause(self, *args, **kwargs):
            # WATCH уже выполнен; замораживаем Redis один раз, до первого EXISTS.
            assert self.watching
            if not injected:
                injected.append(True)
                await admin.execute_command("CLIENT", "PAUSE", pause_ms, "ALL")
            return await original_exists(self, *args, **kwargs)

        try:
            async with queue(dsn, monkeypatch) as (pool, _, _):
                main.app.state.redis_pool = pool
                monkeypatch.setattr(Pipeline, "exists", exists_after_pause)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=main.app), base_url="http://test"
                ) as client:
                    started = time.monotonic()
                    response = await client.post("/api/v1/sessions/hang-after-watch/process")
                    elapsed = time.monotonic() - started

                assert injected, "пауза не была внедрена: тест не проверил нужный этап"
                assert response.status_code == 503
                assert set(response.json()) == {"detail"}
                assert elapsed < max_response_seconds, (
                    f"503 пришёл через {elapsed:.2f}s: очистка pipeline "
                    f"продлила запрос сверх дедлайна"
                )
        finally:
            # Закрытие (в том числе cleanup очереди выше) дождётся конца паузы.
            await admin.aclose()

    asyncio.run(scenario())


@pytest.mark.pr_review
def test_session_can_retry_after_enqueue_times_out_before_exec(
    redis_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """До EXEC задание не записано; после восстановления возможна подтверждённая постановка."""
    session_id = "timeout-before-exec"
    url = f"/api/v1/sessions/{session_id}/process"

    async def scenario() -> None:
        async with queue(redis_dsn, monkeypatch) as (pool, name, _):
            main.app.state.redis_pool = pool
            watch_reached = asyncio.Event()

            async def wait_before_exec(pipeline: Pipeline, *args, **kwargs) -> None:
                assert pipeline.watching
                watch_reached.set()
                await asyncio.Event().wait()

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                with monkeypatch.context() as fault:
                    fault.setattr(Pipeline, "exists", wait_before_exec)
                    first = await asyncio.wait_for(client.post(url), timeout=5)
                assert watch_reached.is_set()
                assert first.status_code == 503
                assert await pool.zcard(name) == 0
                retry = await client.post(url)

            assert retry.status_code == 202, retry.text
            assert await pool.zrange(name, 0, -1) == [retry.json()["run_id"].encode()]

    asyncio.run(scenario())


@pytest.mark.pr_review
@pytest.mark.parametrize("failure", ["before-exec", "after-exec", "after-exec-timeout"])
def test_pipeline_failure_does_not_blindly_retry_enqueue(
    redis_dsn: str, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Настоящие redis-py/ARQ обрабатывают обрыв WATCH-транзакции внутри pipeline."""
    session_id = "lost-enqueue-confirmation"

    async def scenario() -> None:
        async with queue(redis_dsn, monkeypatch) as (pool, name, identifiers):
            main.app.state.redis_pool = pool
            watch = Pipeline.watch
            execute_transaction = Pipeline._execute_transaction
            parse_response = Pipeline.parse_response
            watched, attempts, injected = [], [], []

            async def watch_once(pipeline, *keys):
                watched.append(keys)
                return await watch(pipeline, *keys)

            async def fail_before_exec(pipeline, connection, commands, raise_on_error):
                assert pipeline.watching
                attempts.append(True)
                if failure == "before-exec":
                    injected.append(True)
                    raise RedisConnectionError("synthetic disconnect before MULTI/EXEC")
                return await execute_transaction(pipeline, connection, commands, raise_on_error)

            async def lose_exec_response(pipeline, connection, command_name, **options):
                response = await parse_response(pipeline, connection, command_name, **options)
                if command_name == "_" and isinstance(response, list):
                    # MULTI/QUEUED уже прочитаны, EXEC реально выполнен. Теряем
                    # его ответ до сброса watching в _execute_transaction().
                    assert pipeline.watching
                    assert len(response) == len(pipeline.command_stack) == 2
                    injected.append(True)
                    if failure == "after-exec-timeout":
                        await asyncio.Event().wait()
                    raise RedisConnectionError("synthetic disconnect after EXEC committed")
                return response

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                with monkeypatch.context() as fault:
                    # enqueue_job() не подменяем: redis-py сам превращает обрыв
                    # в WatchError, а ARQ сам возвращает None. WATCH считает
                    # каждую попытку, включая повтор с уже существующим job_id.
                    fault.setattr(Pipeline, "watch", watch_once)
                    fault.setattr(Pipeline, "_execute_transaction", fail_before_exec)
                    fault.setattr(Pipeline, "parse_response", lose_exec_response)
                    response = await asyncio.wait_for(
                        client.post(f"/api/v1/sessions/{session_id}/process"), timeout=5
                    )
            assert response.status_code == 503
            assert set(response.json()) == {"detail"}
            assert len(identifiers) == len(attempts) == len(injected) == 1
            run_id = identifiers[0]
            assert watched == [(job_key_prefix + run_id,)]
            queued_ids = await pool.zrange(name, 0, -1)
            if failure == "before-exec":
                assert queued_ids == []
                assert not await pool.exists(job_key_prefix + run_id)
            else:
                assert queued_ids == [run_id.encode()]
                info = await Job(run_id, pool, _queue_name=name).info()
                assert info is not None
                assert info.kwargs == {"session_id": session_id, "run_id": run_id}

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["task-error", "max-tries", "expired"])
def test_arq_records_terminal_failure(
    redis_dsn: str, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Включая исходы ARQ, при которых coroutine задачи вообще не вызывается."""
    session_id = f"terminal-{failure}"
    url = f"/api/v1/sessions/{session_id}/process"

    async def scenario() -> None:
        settings = importlib.import_module("edu_doc_agent.worker.settings")
        tasks = importlib.import_module("edu_doc_agent.worker.tasks")
        sleep = AsyncMock(side_effect=RuntimeError("synthetic task error"))
        monkeypatch.setattr(tasks, "asyncio", SimpleNamespace(sleep=sleep))
        async with queue(redis_dsn, monkeypatch) as (pool, name, _):
            main.app.state.redis_pool = pool
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                first = await client.post(url)
                assert first.status_code == 202
                run_id = first.json()["run_id"]
                if failure == "expired":
                    # Моделируем истечение payload, не меняя членство в очереди.
                    await pool.delete(job_key_prefix + run_id)
                worker = create_worker(
                    settings.WorkerSettings,
                    redis_pool=pool,
                    queue_name=name,
                    burst=True,
                    max_burst_jobs=1,
                    max_tries=0 if failure == "max-tries" else 3,
                    poll_delay=0.01,
                    handle_signals=False,
                )
                try:
                    await asyncio.wait_for(worker.async_run(), timeout=5)
                    assert worker.jobs_failed == 1
                    if failure == "task-error":
                        sleep.assert_awaited_once_with(3)
                    else:
                        sleep.assert_not_called()
                    info = await Job(run_id, pool, _queue_name=name).result_info()
                    assert info is not None and not info.success
                    assert await pool.zcard(name) == 0
                finally:
                    await worker.close()

    asyncio.run(scenario())


@pytest.mark.pr_review
def test_worker_shutdown_leaves_job_queued_for_retry(
    redis_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Штатная остановка worker оставляет задание ARQ для retry с тем же ID."""
    session_id = "worker-shutdown-retry"
    url = f"/api/v1/sessions/{session_id}/process"

    async def scenario() -> None:
        settings = importlib.import_module("edu_doc_agent.worker.settings")
        tasks = importlib.import_module("edu_doc_agent.worker.tasks")
        started = asyncio.Event()

        async def wait_for_shutdown(delay: float) -> None:
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(tasks, "asyncio", SimpleNamespace(sleep=wait_for_shutdown))
        async with queue(redis_dsn, monkeypatch) as (pool, name, _):
            main.app.state.redis_pool = pool
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                first = await client.post(url)
                assert first.status_code == 202
                run_id = first.json()["run_id"]
                job = Job(run_id, pool, _queue_name=name)
                worker = create_worker(
                    settings.WorkerSettings,
                    redis_pool=pool,
                    queue_name=name,
                    poll_delay=0.01,
                    handle_signals=False,
                )
                running = asyncio.create_task(worker.async_run())
                try:
                    await asyncio.wait_for(started.wait(), timeout=5)
                    worker.handle_sig(signal.SIGTERM)
                    with suppress(asyncio.CancelledError):
                        await running
                    await asyncio.wait_for(asyncio.gather(*worker.tasks.values()), timeout=5)
                    assert worker.jobs_retried == 1
                    assert await job.status() == JobStatus.queued
                    assert await pool.zrange(name, 0, -1) == [run_id.encode()]
                finally:
                    running.cancel()
                    await asyncio.gather(running, return_exceptions=True)
                    await worker.close()

    asyncio.run(scenario())
