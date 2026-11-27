"""Реальный PostgreSQL + FileStorage: исход COMMIT и отмена upload #52."""

import asyncio
import threading
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from edu_doc_agent.api.dependencies import get_document_dao, get_file_storage
from edu_doc_agent.dao import Document, DocumentDAO, LocalFileStorage, SessionRunDAO
from edu_doc_agent.dao import storage as storage_module
from edu_doc_agent.dao.documents import DocumentCreateRolledBackError
from edu_doc_agent.main import app
from tests.integration.dao.support import ACTIVE_STATUSES, TERMINAL_STATUSES

pytestmark = pytest.mark.integration
CONTENT = b"synthetic upload integration bytes"
SESSION_ID = "upload-completion-test"
URL = f"/api/v1/sessions/{SESSION_ID}/documents"


@pytest.fixture
def upload_backend(migrated_database, tmp_path, monkeypatch):
    engine = create_async_engine(migrated_database.url, poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    dao = DocumentDAO(sessions)
    storage = LocalFileStorage(tmp_path / "storage")
    monkeypatch.setitem(app.dependency_overrides, get_document_dao, lambda: dao)
    monkeypatch.setitem(app.dependency_overrides, get_file_storage, lambda: storage)
    yield sessions, dao, storage
    asyncio.run(engine.dispose())


def client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    )


async def assert_saved(api, sessions, storage):
    async with sessions() as db:
        rows = list((await db.scalars(select(Document))).all())
    assert len(rows) == 1
    row = rows[0]
    assert await storage.read(row.storage_key) == CONTENT
    listing = await api.get(URL)
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()["items"]] == [str(row.id)]
    download = await api.get(f"{URL}/{row.id}")
    assert download.status_code == 200
    assert download.content == CONTENT
    assert not list(storage.base_dir.rglob("*upload-reconciliation.json"))
    return row


@pytest.mark.parametrize("run_status", [None, *ACTIVE_STATUSES, *TERMINAL_STATUSES])
def test_committed_document_survives_lost_ack_and_downloads(
    upload_backend, monkeypatch, run_status
):
    sessions, dao, storage = upload_backend
    real_create = dao.create_document
    committed = []

    async def lose_return(**kwargs):
        # Настоящий DAO завершается, включая настоящий PostgreSQL COMMIT.
        row = await real_create(**kwargs)
        committed.append(row.id)
        raise ConnectionError("synthetic lost COMMIT acknowledgement on return")

    monkeypatch.setattr(dao, "create_document", lose_return)

    async def scenario():
        if run_status is not None:
            run = await SessionRunDAO(sessions).create_queued_run(SESSION_ID)
            async with sessions() as db:
                from edu_doc_agent.dao import Run

                stored = await db.get(Run, run.id)
                stored.status = run_status
                await db.commit()
        async with client() as api:
            response = await api.post(URL, files={"file": ("test.docx", CONTENT)})
            assert response.status_code == 201, response.text
            row = await assert_saved(api, sessions, storage)
            assert response.json()["id"] == str(row.id) == str(committed[0])

    asyncio.run(scenario())


def test_error_before_commit_rolls_back_and_removes_file(upload_backend, monkeypatch):
    sessions, dao, storage = upload_backend

    class FailAfterFlush(AsyncSession):
        async def flush(self, objects=None):
            await super().flush(objects)
            raise RuntimeError("synthetic failure before COMMIT")

    monkeypatch.setattr(
        dao,
        "_session_factory",
        async_sessionmaker(sessions.kw["bind"], class_=FailAfterFlush, expire_on_commit=False),
    )
    real_create = dao.create_document
    observed = []

    async def observe(**kwargs):
        try:
            return await real_create(**kwargs)
        except DocumentCreateRolledBackError:
            observed.append("rollback confirmed")
            raise

    monkeypatch.setattr(dao, "create_document", observe)

    async def scenario():
        async with client() as api:
            response = await api.post(URL, files={"file": ("test.docx", CONTENT)})
            assert response.status_code == 500
            assert observed == ["rollback confirmed"]
            async with sessions() as db:
                assert list((await db.scalars(select(Document))).all()) == []
            assert not [p for p in storage.base_dir.rglob("*") if p.is_file()]
            assert (await api.get(URL)).json()["items"] == []

    asyncio.run(scenario())


def test_cancellation_during_real_save_finishes_real_metadata(upload_backend, monkeypatch):
    sessions, dao, storage = upload_backend
    entered, release = threading.Event(), threading.Event()
    replace = storage_module.os.replace

    def blocked_replace(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return replace(*args, **kwargs)

    monkeypatch.setattr(storage_module.os, "replace", blocked_replace)
    create = AsyncMock(wraps=dao.create_document)
    monkeypatch.setattr(dao, "create_document", create)

    async def scenario():
        async with client() as api:
            task = asyncio.create_task(api.post(URL, files={"file": ("test.docx", CONTENT)}))
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
            create.assert_awaited_once()
            await assert_saved(api, sessions, storage)

    asyncio.run(scenario())


def test_committed_but_database_unavailable_for_reconciliation(upload_backend, monkeypatch):
    sessions, dao, storage = upload_backend
    real_create = dao.create_document
    real_get = dao.get_document

    async def lose_return(**kwargs):
        await real_create(**kwargs)
        raise ConnectionError("synthetic lost acknowledgement")

    monkeypatch.setattr(dao, "create_document", lose_return)
    monkeypatch.setattr(
        dao, "get_document", AsyncMock(side_effect=ConnectionError("synthetic unavailable DB"))
    )

    async def scenario():
        async with client() as api:
            response = await api.post(URL, files={"file": ("test.docx", CONTENT)})
            assert response.status_code == 503
            async with sessions() as db:
                row = (await db.scalars(select(Document))).one()
            assert await storage.read(row.storage_key) == CONTENT
            assert len(list(storage.base_dir.rglob("*upload-reconciliation.json"))) == 1
            # После восстановления БД доступны и список, и исходные байты.
            monkeypatch.setattr(dao, "get_document", real_get)
            assert (await api.get(URL)).json()["items"][0]["id"] == str(row.id)
            download = await api.get(f"{URL}/{row.id}")
            assert download.status_code == 200
            assert download.content == CONTENT

    asyncio.run(scenario())


def test_empty_select_does_not_delete_file_before_late_commit(upload_backend, monkeypatch):
    sessions, dao, storage = upload_backend

    async def scenario():
        committing, release = asyncio.Event(), asyncio.Event()
        outstanding = []

        class DelayedCommit(AsyncSession):
            async def commit(self):
                committing.set()
                await release.wait()
                await super().commit()

        delayed_dao = DocumentDAO(
            async_sessionmaker(sessions.kw["bind"], class_=DelayedCommit, expire_on_commit=False)
        )

        async def lose_return(**kwargs):
            outstanding.append(asyncio.create_task(delayed_dao.create_document(**kwargs)))
            await committing.wait()
            raise ConnectionError("transaction still committing")

        monkeypatch.setattr(dao, "create_document", lose_return)
        async with client() as api:
            try:
                response = await api.post(URL, files={"file": ("test.docx", CONTENT)})
                assert response.status_code == 503
                async with sessions() as db:
                    assert (await db.scalars(select(Document))).all() == []
                assert len([p for p in storage.base_dir.rglob("*") if p.is_file()]) == 2
            finally:
                release.set()
                await asyncio.gather(*outstanding)
            row = outstanding[0].result()
            assert await storage.read(row.storage_key) == CONTENT
            assert (await api.get(URL)).json()["items"][0]["id"] == str(row.id)
            download = await api.get(f"{URL}/{row.id}")
            assert download.status_code == 200
            assert download.content == CONTENT
            assert len(list(storage.base_dir.rglob("*upload-reconciliation.json"))) == 1

    asyncio.run(scenario())
