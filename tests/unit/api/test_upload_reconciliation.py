"""Повторяемая локальная сверка никогда не удаляет содержимое документа."""

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
