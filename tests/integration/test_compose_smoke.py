"""Опциональная проверка контейнерной связки в отдельном Compose-проекте."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from textwrap import dedent
from uuid import UUID, uuid4

import httpx
import pytest
import yaml
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.compose


def test_compose_stack_processes_a_session(tmp_path: Path) -> None:
    """Использует готовый образ; не собирает и не скачивает зависимости."""
    image = os.environ.get("TEST_APP_IMAGE")
    if not image:
        pytest.skip("нужен TEST_APP_IMAGE с готовым образом приложения")

    configuration = yaml.safe_load((ROOT / "compose.yaml").read_text())
    services = configuration["services"]
    services.pop("ollama", None)
    storage = tmp_path / "storage"
    storage.mkdir()
    for service in services.values():
        service.pop("env_file", None)
        if "build" in service:
            service.pop("build")
            service["image"] = image
        service.pop("ports", None)

    environment = {
        **os.environ,
        "POSTGRES_USER": "edu",
        "POSTGRES_PASSWORD": "synthetic@pass%40",
        "POSTGRES_DB": "edu",
    }
    for name in ("api", "worker"):
        services[name]["environment"]["REDIS_URL"] = "redis://redis:6379/15"
        services[name]["volumes"] = [f"{storage}:/app/storage"]
    services["api"]["ports"] = ["127.0.0.1::8000"]
    services["ui"]["ports"] = ["127.0.0.1::8501"]

    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(yaml.safe_dump(configuration))
    env_file = tmp_path / "empty.env"
    env_file.write_text("")
    project = f"edu-doc-tests-{uuid4().hex[:12]}"
    command = [
        "docker",
        "compose",
        "--project-name",
        project,
        "--env-file",
        str(env_file),
        "--file",
        str(compose_file),
    ]

    def compose(*arguments: str, timeout: int = 30) -> str:
        completed = subprocess.run(
            [*command, *arguments],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Compose {' '.join(arguments)} failed:\n{completed.stdout}\n{completed.stderr}"
            )
        return completed.stdout

    try:
        # --pull never не обращается к registry; все три образа готовятся заранее.
        compose(
            "up",
            "--detach",
            "--no-build",
            "--pull",
            "never",
            "--wait",
            "--wait-timeout",
            "60",
            timeout=90,
        )
        output = compose("ps", "--all", "--format", "json")
        # Compose разных версий возвращает массив либо JSON Lines.
        try:
            containers = json.loads(output)
        except json.JSONDecodeError:
            containers = [json.loads(line) for line in output.splitlines() if line.strip()]
        if isinstance(containers, dict):
            containers = [containers]
        by_service = {container["Service"]: container for container in containers}
        assert set(by_service) == {"db", "redis", "api", "worker", "ui"}
        assert all(container["State"] == "running" for container in containers)

        # Не выдавать smoke устаревшего локального образа за проверку текущего кода.
        sources = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ("src", "ui", "alembic")
            for path in (ROOT / directory).rglob("*.py")
        }
        sources["alembic.ini"] = hashlib.sha256((ROOT / "alembic.ini").read_bytes()).hexdigest()
        check_sources = dedent("""\
            import hashlib, json, sys
            from pathlib import Path
            expected = json.loads(sys.argv[1])
            mismatched = [name for name, digest in expected.items()
                          if not (Path('/app') / name).is_file()
                          or hashlib.sha256((Path('/app') / name).read_bytes()).hexdigest()
                          != digest]
            assert not mismatched, f'TEST_APP_IMAGE has stale sources: {mismatched}'
        """)
        compose("exec", "--no-TTY", "api", "python", "-c", check_sources, json.dumps(sources))

        # Alembic использует исходный пароль и остальные POSTGRES_* из Compose.
        compose("exec", "--no-TTY", "api", "alembic", "upgrade", "head")
        expected_revision = ScriptDirectory.from_config(
            Config(str(ROOT / "alembic.ini"))
        ).get_current_head()
        actual_revision = compose(
            "exec",
            "--no-TTY",
            "db",
            "psql",
            "--username=edu",
            "--dbname=edu",
            "--tuples-only",
            "--no-align",
            "--command=SELECT version_num FROM alembic_version",
        )
        assert actual_revision.strip() == expected_revision
        dao_revision_script = dedent("""\
            import asyncio
            from sqlalchemy import text
            from edu_doc_agent.dao.database import async_session_maker, engine

            async def check():
                try:
                    async with async_session_maker() as session:
                        print(await session.scalar(text("SELECT version_num FROM alembic_version")))
                finally:
                    await engine.dispose()

            asyncio.run(check())
        """)
        dao_revision = compose("exec", "--no-TTY", "api", "python", "-c", dao_revision_script)
        assert dao_revision.strip() == expected_revision

        api_port = compose("port", "api", "8000").strip().rsplit(":", 1)[1]
        ui_port = compose("port", "ui", "8501").strip().rsplit(":", 1)[1]
        with httpx.Client(base_url=f"http://127.0.0.1:{api_port}", trust_env=False) as client:
            # Docker Desktop/WSL может открыть опубликованный порт чуть позже
            # внутреннего healthcheck контейнера.
            deadline = time.monotonic() + 15
            while True:
                try:
                    ready = client.get("/health").status_code == 200
                except httpx.TransportError:
                    ready = False
                if ready:
                    break
                assert time.monotonic() < deadline, "Published API port did not become ready"
                time.sleep(0.05)
            assert client.get("/docs").status_code == 200
            response = client.post("/api/v1/sessions/compose-synthetic/process")
            assert response.status_code == 202
            body = response.json()
            UUID(body["run_id"])
            assert body["session_id"] == "compose-synthetic"
            assert body["status"] == "queued"

        result_script = dedent(f"""\
            import asyncio, json
            from arq import create_pool
            from arq.jobs import Job
            from edu_doc_agent.config import get_settings

            async def check():
                pool = await create_pool(get_settings().arq_redis_settings)
                try:
                    result = await Job({body["run_id"]!r}, pool).result(timeout=15, poll_delay=0.1)
                    print(json.dumps(result))
                finally:
                    await pool.aclose(close_connection_pool=True)

            asyncio.run(check())
        """)
        result = json.loads(compose("exec", "--no-TTY", "api", "python", "-c", result_script))
        assert result == {
            "status": "completed",
            "session_id": "compose-synthetic",
            "run_id": body["run_id"],
        }
        worker_logs = compose("logs", "--no-color", "worker")
        assert body["run_id"] in worker_logs

        with httpx.Client(base_url=f"http://127.0.0.1:{ui_port}", trust_env=False) as client:
            deadline = time.monotonic() + 15
            while True:
                try:
                    ready = client.get("/_stcore/health").status_code == 200
                except httpx.TransportError:
                    ready = False
                if ready:
                    break
                assert time.monotonic() < deadline, "Streamlit did not become ready"
                time.sleep(0.2)
    finally:
        # Только уникальный проект этого теста, включая его временные тома.
        compose("down", "--volumes", "--remove-orphans", timeout=60)
