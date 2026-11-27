"""Подтверждённый rollback, гонка POST/DELETE и настоящий процесс CLI сверки."""

import asyncio
import errno
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from edu_doc_agent.dao import Document, DocumentDAO, LocalFileStorage
from edu_doc_agent.dao import storage as storage_module
from tests.integration.dao.support import Database
from tests.integration.test_documents_api import (
    CONTENT,
    SESSION_ID,
    URL,
    assert_absent,
    documents_client,
    read_document,
    stored_files,
)
from tests.integration.test_documents_api import document_sessions as document_sessions
from tests.integration.test_documents_api import document_storage as document_storage

pytestmark = pytest.mark.integration


async def cli_process(database: Database, directory: Path, storage_dir: Path):
    environment = {
        name: os.environ[name] for name in ("PATH", "HOME", "SYSTEMROOT") if name in os.environ
    }
    environment.update(
        DATABASE_URL=database.url.render_as_string(hide_password=False),
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(directory),
    )
    return await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            "-m",
            "edu_doc_agent.reconcile_uploads",
            "--storage-dir",
            str(storage_dir),
        ],
        env=environment,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_confirmed_rollback_and_once_failed_compensation_are_cleaned_by_new_cli_process(
    document_sessions,
    document_storage,
    migrated_database,
    tmp_path,
    monkeypatch,
):
    rollbacks, commits, failed_unlinks = [], [], []

    class FailBeforeCommit(AsyncSession):
        async def flush(self, objects=None):
            await super().flush(objects)
            raise RuntimeError("synthetic failure after INSERT, before COMMIT")

        async def rollback(self):
            await super().rollback()
            rollbacks.append(True)

        async def commit(self):
            commits.append(True)
            await super().commit()

    failing_sessions = async_sessionmaker(
        document_sessions.kw["bind"], class_=FailBeforeCommit, expire_on_commit=False
    )
    real_unlink = storage_module.os.unlink

    def fail_content_unlink_once(name, *, dir_fd=None):
        if str(name).endswith("synthetic.pdf") and dir_fd is not None and not failed_unlinks:
            failed_unlinks.append(name)
            raise OSError(errno.EIO, "synthetic one-time compensation unlink failure")
        return real_unlink(name, dir_fd=dir_fd)

    async def scenario():
        with monkeypatch.context() as boundary:
            boundary.setattr(storage_module.os, "unlink", fail_content_unlink_once)
            async with documents_client(DocumentDAO(failing_sessions), document_storage) as api:
                response = await api.post(URL, files={"file": ("synthetic.pdf", CONTENT)})
                assert response.status_code == 500, response.text
        assert rollbacks == [True]
        assert commits == []
        assert len(failed_unlinks) == 1
        async with document_sessions() as db:
            assert (await db.scalars(select(Document))).all() == []
        markers = await document_storage.list_upload_intents()
        assert len(markers) == 1
        marker = json.loads(await document_storage.read(markers[0]))
        assert marker["outcome"] == "rolled_back"
        assert len(stored_files(document_storage)) == 2
        original_path = next(
            path for path in stored_files(document_storage) if path.name.endswith("synthetic.pdf")
        )
        assert original_path.read_bytes() == CONTENT

        result = await cli_process(migrated_database, tmp_path, document_storage.base_dir)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout) == {"rolled_back": 1}
        assert stored_files(document_storage) == []
        assert not original_path.exists()
        repeated = await cli_process(migrated_database, tmp_path, document_storage.base_dir)
        assert repeated.returncode == 0, repeated.stdout + repeated.stderr
        assert json.loads(repeated.stdout) == {}
        print(
            f"rollback: POST={response.status_code}, "
            f"CLI={result.returncode} {result.stdout.strip()}, "
            f"repeat={repeated.returncode} {repeated.stdout.strip()}, DB/files empty"
        )
        assert stored_files(document_storage) == []
        async with document_sessions() as db:
            assert (await db.scalars(select(Document))).all() == []

    asyncio.run(scenario())


def test_post_lost_ack_after_logical_delete_confirms_same_document_and_delete_completes(
    document_sessions, document_storage, monkeypatch
):
    async def scenario():
        committed, return_lost_ack = asyncio.Event(), asyncio.Event()
        cleanup_entered, release_cleanup = asyncio.Event(), asyncio.Event()
        dao = DocumentDAO(document_sessions)
        real_create = dao.create_document
        committed_rows = []

        async def commit_then_lose_return(**kwargs):
            row = await real_create(**kwargs)
            committed_rows.append(row)
            committed.set()
            await return_lost_ack.wait()
            raise ConnectionError("synthetic lost return after committed logical DELETE")

        observed_read = AsyncMock(wraps=dao.get_document_for_reconciliation)
        deleting_storage = LocalFileStorage(document_storage.base_dir)
        real_delete = deleting_storage.delete

        async def pause_content_cleanup(key):
            assert key == committed_rows[0].storage_key
            cleanup_entered.set()
            await release_cleanup.wait()
            return await real_delete(key)

        with monkeypatch.context() as boundary:
            boundary.setattr(dao, "create_document", commit_then_lose_return)
            boundary.setattr(dao, "get_document_for_reconciliation", observed_read)
            boundary.setattr(deleting_storage, "delete", pause_content_cleanup)
            async with (
                documents_client(dao, document_storage) as uploading_api,
                documents_client(DocumentDAO(document_sessions), deleting_storage) as deleting_api,
            ):
                uploading = asyncio.create_task(
                    uploading_api.post(URL, files={"file": ("synthetic.pdf", CONTENT)})
                )
                deleting = None
                try:
                    await asyncio.wait_for(committed.wait(), 5)
                    row = committed_rows[0]
                    deleting = asyncio.create_task(deleting_api.delete(f"{URL}/{row.id}"))
                    await asyncio.wait_for(cleanup_entered.wait(), 5)
                    pending = await read_document(document_sessions, str(row.id))
                    assert pending is not None and pending.deletion_pending
                    assert await document_storage.read(row.storage_key) == CONTENT
                    assert (await deleting_api.get(f"{URL}/{row.id}")).status_code == 404
                    return_lost_ack.set()
                    response = await asyncio.wait_for(uploading, 5)
                    assert response.status_code == 201, response.text
                    assert response.json()["id"] == str(row.id)
                    observed_read.assert_awaited_once_with(session_id=SESSION_ID, doc_id=row.id)
                    assert (await deleting_api.get(f"{URL}/{row.id}")).status_code == 404
                    listing = await deleting_api.get(URL)
                    assert listing.status_code == 200 and listing.json() == {"items": []}
                    assert not deleting.done()
                    release_cleanup.set()
                    deleted = await asyncio.wait_for(deleting, 5)
                    assert deleted.status_code == 204, deleted.text
                    print(f"POST/DELETE race: POST={response.status_code}, GET=404, DELETE=204")
                    await assert_absent(
                        deleting_api,
                        document_sessions,
                        document_storage,
                        str(row.id),
                        row.storage_key,
                    )
                finally:
                    return_lost_ack.set()
                    release_cleanup.set()
                    tasks = [uploading] + ([deleting] if deleting is not None else [])
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("exists", [False, True], ids=["missing-storage", "empty-storage"])
def test_reconciliation_cli_distinguishes_missing_storage_from_empty_directory(
    migrated_database, tmp_path, exists
):
    storage_dir = tmp_path / "explicit-storage"
    if exists:
        storage_dir.mkdir()

    async def scenario():
        result = await cli_process(migrated_database, tmp_path, storage_dir)
        print(
            f"CLI storage exists={exists}: exit={result.returncode}, stdout={result.stdout.strip()}"
        )
        if exists:
            assert result.returncode == 0, result.stdout + result.stderr
            assert json.loads(result.stdout) == {}
            assert list(storage_dir.iterdir()) == []
        else:
            assert result.returncode != 0, result.stdout + result.stderr
            assert str(storage_dir) in result.stdout + result.stderr
            assert not storage_dir.exists()

    asyncio.run(scenario())
