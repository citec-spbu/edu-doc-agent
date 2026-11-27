"""Сверка сохраняет неопределённый исход и завершает подтверждённый rollback."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from edu_doc_agent.dao import FileStorageError, LocalFileStorage
from edu_doc_agent.reconcile_uploads import (
    INTENT_FILENAME,
    clear_upload_marker,
    reconcile_uploads,
)


async def seed(storage):
    doc_id = uuid4()
    content_key = await storage.save(
        session_id="synthetic", doc_id=doc_id, filename="test.docx", content=b"synthetic bytes"
    )
    intent = {
        "version": 1,
        "session_id": "synthetic",
        "doc_id": str(doc_id),
        "storage_key": content_key,
    }
    marker = await storage.save(
        session_id="synthetic",
        doc_id=doc_id,
        filename=INTENT_FILENAME,
        content=json.dumps(intent).encode(),
    )
    return doc_id, content_key, marker


@pytest.mark.parametrize("state", ["confirmed", "pending", "absent", "mismatch", "unavailable"])
def test_reconciliation_retains_bytes_in_every_state(tmp_path, state):
    async def scenario():
        storage = LocalFileStorage(tmp_path)
        doc_id, key, marker = await seed(storage)
        row = SimpleNamespace(
            session_id="synthetic", id=doc_id, storage_key=key, deletion_pending=state == "pending"
        )
        if state == "mismatch":
            row.storage_key = "different/key"
        lookup = AsyncMock(return_value=None if state == "absent" else row)
        if state == "unavailable":
            lookup.side_effect = ConnectionError("synthetic outage")
        dao = SimpleNamespace(get_document_for_reconciliation=lookup)
        expected = {
            "confirmed": "confirmed",
            "pending": "confirmed",
            "absent": "unresolved",
            "mismatch": "mismatch",
            "unavailable": "database_unavailable",
        }[state]
        assert await reconcile_uploads(dao, storage) == {expected: 1}
        assert await storage.read(key) == b"synthetic bytes"
        assert await storage.exists(marker) == (expected != "confirmed")
        again = await reconcile_uploads(dao, storage)
        assert again == ({} if expected == "confirmed" else {expected: 1})

    asyncio.run(scenario())


def test_delete_hook_checks_identity_and_is_repeatable(tmp_path):
    async def scenario():
        storage = LocalFileStorage(tmp_path)
        doc_id, key, marker = await seed(storage)
        with pytest.raises(FileStorageError):
            await clear_upload_marker(
                storage, session_id="synthetic", doc_id=doc_id, storage_key="wrong/key"
            )
        assert await storage.exists(marker)
        for _ in range(2):
            await clear_upload_marker(
                storage, session_id="synthetic", doc_id=doc_id, storage_key=key
            )
        assert not await storage.exists(marker)
        assert await storage.read(key) == b"synthetic bytes"

    asyncio.run(scenario())


def test_invalid_marker_retained_without_querying_database(tmp_path):
    async def scenario():
        storage = LocalFileStorage(tmp_path)
        _, key, marker = await seed(storage)
        storage.resolve_path(marker).write_bytes(b"{}")
        dao = SimpleNamespace(get_document_for_reconciliation=AsyncMock())
        assert await reconcile_uploads(dao, storage) == {"invalid": 1}
        dao.get_document_for_reconciliation.assert_not_awaited()
        assert await storage.exists(marker)
        assert await storage.exists(key)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_at", ["content", "marker"])
def test_cli_rollback_cleanup_is_repeatable_after_another_disk_failure(
    tmp_path, monkeypatch, failure_at
):
    from edu_doc_agent.reconcile_uploads import record_upload_rollback

    async def scenario():
        storage = LocalFileStorage(tmp_path)
        doc_id, key, marker = await seed(storage)
        await record_upload_rollback(
            storage, session_id="synthetic", doc_id=doc_id, storage_key=key
        )
        real_delete = storage.delete

        async def fail_delete(storage_key):
            if storage_key == (key if failure_at == "content" else marker):
                raise FileStorageError("synthetic temporary disk failure")
            return await real_delete(storage_key)

        monkeypatch.setattr(storage, "delete", fail_delete)
        dao = SimpleNamespace(get_document_for_reconciliation=AsyncMock(return_value=None))
        assert await reconcile_uploads(dao, storage) == {"cleanup_failed": 1}
        assert await storage.exists(marker)
        assert json.loads(await storage.read(marker))["outcome"] == "rolled_back"
        assert await storage.exists(key) == (failure_at == "content")
        fresh_storage = LocalFileStorage(tmp_path)
        assert await reconcile_uploads(dao, fresh_storage) == {"rolled_back": 1}
        assert await reconcile_uploads(dao, fresh_storage) == {}
        assert not [path for path in tmp_path.rglob("*") if path.is_file()]

    asyncio.run(scenario())


@pytest.mark.parametrize("state", ["row-exists", "database-unavailable"])
def test_rollback_evidence_does_not_override_contradiction_or_database_outage(tmp_path, state):
    from edu_doc_agent.reconcile_uploads import record_upload_rollback

    async def scenario():
        storage = LocalFileStorage(tmp_path)
        doc_id, key, marker = await seed(storage)
        await record_upload_rollback(
            storage, session_id="synthetic", doc_id=doc_id, storage_key=key
        )
        lookup = AsyncMock(
            return_value=SimpleNamespace(session_id="synthetic", id=doc_id, storage_key=key)
        )
        if state == "database-unavailable":
            lookup.side_effect = ConnectionError("synthetic unavailable database")
        dao = SimpleNamespace(get_document_for_reconciliation=lookup)
        expected = "mismatch" if state == "row-exists" else "database_unavailable"
        assert await reconcile_uploads(dao, storage) == {expected: 1}
        assert await storage.exists(marker)
        assert await storage.read(key) == b"synthetic bytes"

    asyncio.run(scenario())


@pytest.mark.parametrize("root_kind", ["missing", "file", "unavailable", "empty"])
def test_cli_validates_storage_root(tmp_path, monkeypatch, capsys, root_kind):
    from edu_doc_agent import reconcile_uploads as module

    root = tmp_path / "storage"
    if root_kind == "file":
        root.write_bytes(b"synthetic file")
    elif root_kind != "missing":
        root.mkdir()
    monkeypatch.setattr("sys.argv", ["reconcile_uploads", "--storage-dir", str(root)])
    if root_kind == "unavailable":
        original_scandir = module.os.scandir

        def denied(path):
            if str(path) == str(root):
                raise PermissionError("synthetic permission denied")
            return original_scandir(path)

        monkeypatch.setattr(module.os, "scandir", denied)
    if root_kind == "empty":
        assert module.main() == 0
        assert json.loads(capsys.readouterr().out) == {}
    else:
        with pytest.raises(SystemExit) as caught:
            module.main()
        assert caught.value.code != 0
        output = capsys.readouterr()
        assert str(root) in output.err
        assert not output.out
