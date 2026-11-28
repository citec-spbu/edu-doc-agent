"""Завершение записи, компенсация и неопределённый исход загрузки #52."""

import asyncio
import errno
import json
import threading
from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile

from edu_doc_agent.api.documents_router import upload_document
from edu_doc_agent.dao import LocalFileStorage
from edu_doc_agent.dao import storage as storage_module
from edu_doc_agent.dao.documents import DocumentCreateRolledBackError

CONTENT = b"synthetic document bytes"


def document(**values):
    values["id"] = values.pop("doc_id")
    return SimpleNamespace(**values, derived_from=None, created_at=datetime.now(UTC))


async def upload(dao, storage):
    return await upload_document(
        session_id="synthetic-session",
        file=UploadFile(filename="test.docx", file=BytesIO(CONTENT)),
        dao=dao,
        storage=storage,
    )


def test_cancellation_during_replace_finishes_metadata(tmp_path, monkeypatch):
    """Два cancel во время реального os.replace не теряют storage_key."""
    storage = LocalFileStorage(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = storage_module.os.replace

    destinations = []

    def blocked_replace(*args, **kwargs):
        destinations.append(str(args[1]))
        if str(args[1]).endswith(".docx") and not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage_module.os, "replace", blocked_replace)
    dao = SimpleNamespace(create_document=AsyncMock(side_effect=document))

    async def scenario():
        task = asyncio.create_task(upload(dao, storage))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert destinations[-1].endswith(".docx"), destinations
            assert destinations[0].endswith("upload-reconciliation.json"), destinations
            dao.create_document.assert_not_awaited()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        dao.create_document.assert_awaited_once()
        values = dao.create_document.call_args.kwargs
        assert await storage.read(values["storage_key"]) == CONTENT
        assert len([p for p in tmp_path.rglob("*") if p.is_file()]) == 1
        assert await storage.list_upload_intents() == []

    asyncio.run(scenario())


@pytest.mark.parametrize("cleanup", ["delete", "already-absent", "failure"])
def test_rollback_marker_enospc_still_attempts_compensation(tmp_path, monkeypatch, caplog, cleanup):
    delete_fails = cleanup == "failure"
    storage = LocalFileStorage(tmp_path)
    dao = SimpleNamespace(
        create_document=AsyncMock(side_effect=DocumentCreateRolledBackError("confirmed rollback"))
    )
    real_save, real_delete = storage.save, storage.delete
    deleted = []

    async def fail_confirmation(**kwargs):
        if kwargs["filename"] == "upload-reconciliation.json":
            if json.loads(kwargs["content"]).get("outcome") == "rolled_back":
                if cleanup == "already-absent":
                    values = dao.create_document.call_args.kwargs
                    await real_delete(values["storage_key"])
                raise OSError(errno.ENOSPC, "synthetic full disk")
        return await real_save(**kwargs)

    async def delete(key):
        deleted.append(key)
        if delete_fails:
            raise OSError(errno.EIO, "synthetic unlink failure")
        return await real_delete(key)

    monkeypatch.setattr(storage, "save", fail_confirmation)
    monkeypatch.setattr(storage, "delete", delete)

    async def scenario():
        with pytest.raises(DocumentCreateRolledBackError):
            await upload(dao, storage)
        values = dao.create_document.call_args.kwargs
        key = values["storage_key"]
        assert deleted[0] == key
        assert await storage.exists(key) == delete_fails
        markers = await storage.list_upload_intents()
        assert len(markers) == int(delete_fails)
        assert len(deleted) == (1 if delete_fails else 2)
        assert str(values["doc_id"]) in caplog.text
        assert key in caplog.text
        if delete_fails:
            assert await storage.read(key) == CONTENT
            assert json.loads(await storage.read(markers[0])).get("outcome", "unknown") == "unknown"

    asyncio.run(scenario())


def test_repeated_cancellation_during_cleanup_waits_for_unlink(tmp_path, monkeypatch):
    storage = LocalFileStorage(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = storage_module.os.unlink

    def blocked_unlink(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage_module.os, "unlink", blocked_unlink)
    dao = SimpleNamespace(
        create_document=AsyncMock(side_effect=DocumentCreateRolledBackError("rollback confirmed"))
    )

    async def scenario():
        task = asyncio.create_task(upload(dao, storage))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        dao.create_document.assert_awaited_once()
        assert not list(tmp_path.rglob("*.*"))

    asyncio.run(scenario())


@pytest.mark.parametrize("lookup", [None, ConnectionError("database unavailable"), "other-key"])
def test_unknown_outcome_preserves_file_and_persistent_intent(tmp_path, lookup):
    storage = LocalFileStorage(tmp_path)
    values = {}

    async def unknown_create(**kwargs):
        values.update(kwargs)
        raise ConnectionError("commit acknowledgement lost")

    async def lookup_document(session_id, doc_id):
        assert (session_id, doc_id) == (values["session_id"], values["doc_id"])
        if isinstance(lookup, Exception):
            raise lookup
        if lookup == "other-key":
            return document(**(values | {"storage_key": lookup}))
        return lookup

    dao = SimpleNamespace(
        create_document=unknown_create, get_document_for_reconciliation=lookup_document
    )

    async def scenario():
        with pytest.raises(HTTPException) as caught:
            await upload(dao, storage)
        assert caught.value.status_code == 503
        assert await storage.read(values["storage_key"]) == CONTENT
        marker = next(tmp_path.rglob("*upload-reconciliation.json"))
        intent = json.loads(marker.read_bytes())
        assert intent["version"] == 1
        assert intent["doc_id"] == str(values["doc_id"])
        assert intent["session_id"] == values["session_id"]
        assert intent["storage_key"] == values["storage_key"]
        assert intent.get("outcome", "unknown") == "unknown"
        if lookup is None:
            from edu_doc_agent.reconcile_uploads import reconcile_uploads

            fresh_storage = LocalFileStorage(tmp_path)
            for _ in range(2):
                assert await reconcile_uploads(dao, fresh_storage) == {"unresolved": 1}
                assert await fresh_storage.read(values["storage_key"]) == CONTENT
                assert json.loads(marker.read_bytes()) == intent
        # Пустой SELECT может предшествовать позднему COMMIT: байты не удалены.
        assert len([p for p in tmp_path.rglob("*") if p.is_file()]) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["execute", "flush", "commit", "rollback"])
def test_dao_only_marks_confirmed_precommit_rollback(stage):
    from unittest.mock import Mock
    from uuid import uuid4

    from sqlalchemy.ext.asyncio import AsyncSession

    from edu_doc_agent.dao import DocumentDAO

    db = AsyncMock(spec=AsyncSession)
    db.__aenter__.return_value = db
    failure = RuntimeError(f"synthetic {stage} failure")
    getattr(db, stage).side_effect = failure
    if stage == "rollback":
        db.flush.side_effect = RuntimeError("synthetic flush failure")
    dao = DocumentDAO(Mock(return_value=db))

    async def scenario():
        with pytest.raises(Exception) as caught:
            await dao.create_document(
                session_id="synthetic-session",
                doc_id=uuid4(),
                filename="test.docx",
                content_type="application/octet-stream",
                size_bytes=len(CONTENT),
                storage_key="synthetic/key",
            )
        if stage in {"execute", "flush"}:
            assert isinstance(caught.value, DocumentCreateRolledBackError)
            assert caught.value.__cause__ is failure
            db.rollback.assert_awaited_once()
            db.commit.assert_not_awaited()
        else:
            assert caught.value is failure
            assert not isinstance(caught.value, DocumentCreateRolledBackError)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_at", ["content", "marker"])
def test_confirmed_rollback_survives_disk_cleanup_failure_and_new_process(
    tmp_path, monkeypatch, failure_at
):
    from edu_doc_agent.dao import FileStorageError
    from edu_doc_agent.reconcile_uploads import reconcile_uploads

    storage = LocalFileStorage(tmp_path)
    dao = SimpleNamespace(
        create_document=AsyncMock(side_effect=DocumentCreateRolledBackError("confirmed rollback"))
    )
    real_delete = storage.delete
    failed = False

    async def fail_once(key):
        nonlocal failed
        is_marker = key.endswith("upload-reconciliation.json")
        if not failed and is_marker == (failure_at == "marker"):
            failed = True
            raise FileStorageError("synthetic temporary disk failure")
        return await real_delete(key)

    monkeypatch.setattr(storage, "delete", fail_once)

    async def scenario():
        with pytest.raises(DocumentCreateRolledBackError):
            await upload(dao, storage)
        assert failed
        values = dao.create_document.call_args.kwargs
        marker = next(tmp_path.rglob("*upload-reconciliation.json"))
        intent = json.loads(marker.read_bytes())
        assert intent["outcome"] == "rolled_back"
        assert intent["doc_id"] == str(values["doc_id"])
        assert intent["storage_key"] == values["storage_key"]
        assert await storage.exists(values["storage_key"]) == (failure_at == "content")
        # Новые экземпляры не имеют доступа к состоянию исходного запроса.
        fresh_storage = LocalFileStorage(tmp_path)
        fresh_dao = SimpleNamespace(get_document_for_reconciliation=AsyncMock(return_value=None))
        assert await reconcile_uploads(fresh_dao, fresh_storage) == {"rolled_back": 1}
        assert not [path for path in tmp_path.rglob("*") if path.is_file()]
        assert await reconcile_uploads(fresh_dao, fresh_storage) == {}

    asyncio.run(scenario())
