"""Завершение записи, компенсация и неопределённый исход загрузки #52."""

import asyncio
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

    def blocked_replace(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage_module.os, "replace", blocked_replace)
    dao = SimpleNamespace(create_document=AsyncMock(side_effect=document))

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
        values = dao.create_document.call_args.kwargs
        assert await storage.read(values["storage_key"]) == CONTENT
        assert len(list(tmp_path.rglob("*.*"))) == 1

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

    async def lookup_document(**kwargs):
        assert kwargs == {"session_id": values["session_id"], "doc_id": values["doc_id"]}
        if isinstance(lookup, Exception):
            raise lookup
        if lookup == "other-key":
            return document(**(values | {"storage_key": lookup}))
        return lookup

    dao = SimpleNamespace(create_document=unknown_create, get_document=lookup_document)

    async def scenario():
        with pytest.raises(HTTPException) as caught:
            await upload(dao, storage)
        assert caught.value.status_code == 503
        assert await storage.read(values["storage_key"]) == CONTENT
        marker = next(tmp_path.rglob("*upload-reconciliation.json"))
        intent = json.loads(marker.read_bytes())
        assert intent == {
            "version": 1,
            "doc_id": str(values["doc_id"]),
            "session_id": values["session_id"],
            "storage_key": values["storage_key"],
        }
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
