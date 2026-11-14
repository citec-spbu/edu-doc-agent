"""Изоляция PostgreSQL и запуск настоящего Alembic без рабочего .env."""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError

ROOT = Path(__file__).resolve().parents[3]
INITIAL_REVISION = "504d7cc74177"
ACTIVE_INDEX = "uq_runs_active_session"
ACTIVE_STATUSES = ("queued", "running", "waiting_for_input")
TERMINAL_STATUSES = ("completed", "failed", "cancelled")


def head_revision() -> str:
    revision = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini"))).get_current_head()
    assert revision is not None
    return revision


def assert_pg_error(error: IntegrityError, code: str, constraint: str | None = None) -> None:
    assert error.orig.sqlstate == code
    if constraint is not None:
        assert error.orig.diag.constraint_name == constraint


@dataclass
class Database:
    url: URL
    engine: Engine
    directory: Path
    schema: str

    def alembic(self, *arguments: str, url: URL | None = None) -> subprocess.CompletedProcess[str]:
        # Дочерний процесс запускается вне корня проекта: .env не читается.
        environment = {
            key: os.environ[key]
            for key in ("PATH", "SYSTEMROOT", "HOME", "TMPDIR")
            if key in os.environ
        }
        environment.update(
            DATABASE_URL=(url or self.url).render_as_string(hide_password=False),
            PYTHONPATH=str(ROOT / "src"),
            PYTHONDONTWRITEBYTECODE="1",
        )
        return subprocess.run(
            [sys.executable, "-m", "alembic", "-c", str(ROOT / "alembic.ini"), *arguments],
            cwd=self.directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def migrate(self, revision: str = "head") -> None:
        result = self.alembic("upgrade", revision)
        assert result.returncode == 0, result.stdout + result.stderr
