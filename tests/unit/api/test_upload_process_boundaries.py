"""Реальные границы процесса и прав доступа, синтетические HTTP и storage."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[3]


def run_python(directory, code=None, storage_dir=None):
    environment = {
        name: os.environ[name] for name in ("PATH", "HOME", "SYSTEMROOT") if name in os.environ
    }
    environment.update(
        PYTHONPATH=str(PROJECT / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(directory),
        STORAGE_DIR=str(storage_dir or directory / "storage"),
    )
    command = [sys.executable]
    command += ["-c", code] if code else ["-m", "edu_doc_agent.reconcile_uploads"]
    return subprocess.run(
        command, cwd=directory, env=environment, capture_output=True, text=True, timeout=20
    )


@pytest.mark.parametrize("checkpoint", ["marker-entry", "content-written"])
def test_process_death_cannot_leave_untracked_upload(tmp_path, checkpoint):
    root = tmp_path / "storage"
    root.mkdir()
    code = """
import os
from types import SimpleNamespace
from fastapi.testclient import TestClient
from edu_doc_agent.config import Settings
from edu_doc_agent import config, main
from edu_doc_agent.api.dependencies import get_document_dao, get_file_storage
from edu_doc_agent.dao import LocalFileStorage

settings = Settings(_env_file=None)
config.get_settings = lambda: settings
main.get_settings = lambda: settings
storage = LocalFileStorage(settings.storage_dir)
real_save = storage.save
async def crash_save(**kwargs):
    if CHECKPOINT == "marker-entry" and kwargs["filename"] == "upload-reconciliation.json":
        os._exit(73)
    key = await real_save(**kwargs)
    if CHECKPOINT == "content-written" and kwargs["filename"] == "a.pdf":
        os._exit(73)
    return key
storage.save = crash_save
async def unexpected_dao(**kwargs):
    os._exit(74)
main.app.dependency_overrides[get_document_dao] = lambda: SimpleNamespace(
    create_document=unexpected_dao
)
main.app.dependency_overrides[get_file_storage] = lambda: storage
client = TestClient(main.app, raise_server_exceptions=False)
client.post("/api/v1/sessions/synthetic/documents", files={"file": ("a.pdf", b"A")})
""".replace("CHECKPOINT", repr(checkpoint))
    stopped = run_python(tmp_path, code)
    assert stopped.returncode == 73, stopped.stdout + stopped.stderr
    # Настоящая команда -m до проверки состояния: без маркера БД не нужна.
    if checkpoint == "marker-entry":
        initial_cli = run_python(tmp_path)
        assert initial_cli.returncode == 0, initial_cli.stdout + initial_cli.stderr
        assert json.loads(initial_cli.stdout) == {}
    content = list(root.rglob("*.pdf"))
    markers = list(root.rglob("*upload-reconciliation.json"))
    if checkpoint == "marker-entry":
        # Новый порядок прекращает загрузку до байтов; старый оставлял PDF.
        assert content == []
        assert markers == []
    else:
        assert len(content) == len(markers) == 1
        assert content[0].read_bytes() == b"A"
        original_marker = markers[0].read_bytes()
        # CLI выполняется в новом процессе. Только граница БД заменена пустым
        # SELECT; файловый обход, чтение, сверка и main() — штатные.
        reconcile = """
import runpy
from unittest.mock import AsyncMock
from types import SimpleNamespace
import edu_doc_agent.dao as dao
dao.DocumentDAO = lambda: SimpleNamespace(
    get_document_for_reconciliation=AsyncMock(return_value=None)
)
runpy.run_module("edu_doc_agent.reconcile_uploads", run_name="__main__")
"""
        for _ in range(2):
            result = run_python(tmp_path, reconcile)
            assert result.returncode == 1, result.stdout + result.stderr
            assert json.loads(result.stdout) == {"unresolved": 1}
            assert content[0].read_bytes() == b"A"
            assert markers[0].read_bytes() == original_marker


def test_cli_reports_inaccessible_nested_directory(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("Проверка chmod требует обычного пользователя")
    root = tmp_path / "storage"
    session = root / "session"
    session.mkdir(parents=True)
    marker = session / "upload-reconciliation.json"
    marker.write_bytes(b"{}")
    session.chmod(0)
    try:
        denied = run_python(tmp_path)
        assert denied.returncode != 0, denied.stdout + denied.stderr
        assert str(session) in denied.stderr
        assert "Permission denied" in denied.stderr
        assert denied.stdout.strip() != "{}"
    finally:
        session.chmod(0o700)
    available = run_python(tmp_path)
    assert available.returncode == 1, available.stdout + available.stderr
    assert json.loads(available.stdout) == {"invalid": 1}
    marker.unlink()
    empty = run_python(tmp_path)
    assert empty.returncode == 0, empty.stdout + empty.stderr
    assert json.loads(empty.stdout) == {}
