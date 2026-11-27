"""Documents API #52: HTTP, миграции PostgreSQL и реальные файлы, включая сбои."""

import asyncio
import errno
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from edu_doc_agent.api.dependencies import get_document_dao, get_file_storage
from edu_doc_agent.api.documents_router import router
from edu_doc_agent.dao import Document, DocumentDAO, LocalFileStorage, Run, SessionRunDAO
from edu_doc_agent.dao import storage as storage_module
from tests.integration.dao.support import Database

pytestmark = pytest.mark.integration
SESSION_ID = "synthetic-documents-session"
URL = f"/api/v1/sessions/{SESSION_ID}/documents"
CONTENT = b"%PDF-1.4\n% synthetic document bytes\n\x00\xff\r\n%%EOF\n"


@pytest.fixture
def document_sessions(migrated_database: Database) -> Iterator[async_sessionmaker[AsyncSession]]:
    """Общая фикстура применяет настоящий Alembic в отдельной схеме каждого теста."""
    engine = create_async_engine(migrated_database.url, poolclass=NullPool)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        asyncio.run(engine.dispose())


@pytest.fixture
def document_storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(tmp_path / "documents")


def documents_client(dao: DocumentDAO, storage: LocalFileStorage) -> httpx.AsyncClient:
    """Новый экземпляр HTTP-приложения и зависимостей; Redis здесь не используется."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_document_dao] = lambda: dao
    app.dependency_overrides[get_file_storage] = lambda: storage
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


async def read_document(sessions: async_sessionmaker[AsyncSession], doc_id: str) -> Document | None:
    # Независимая сессия читает именно зафиксированное состояние PostgreSQL.
    async with sessions() as db:
        return await db.get(Document, UUID(doc_id))


def stored_files(storage: LocalFileStorage) -> list[Path]:
    return sorted(path for path in storage.base_dir.rglob("*") if path.is_file())


async def upload(api: httpx.AsyncClient, filename: str = "synthetic.pdf") -> dict:
    response = await api.post(
        URL,
        files={"file": (filename, CONTENT, "application/octet-stream")},
        data={"doc_type": "rpd"},
    )
    assert response.status_code == 201, response.text
    return response.json()


async def assert_absent(
    api: httpx.AsyncClient,
    sessions: async_sessionmaker[AsyncSession],
    storage: LocalFileStorage,
    doc_id: str,
    storage_key: str,
) -> None:
    assert await read_document(sessions, doc_id) is None
    assert not await storage.exists(storage_key)
    assert stored_files(storage) == []
    listing = await api.get(URL)
    assert listing.status_code == 200
    assert listing.json() == {"items": []}
    assert (await api.get(f"{URL}/{doc_id}")).status_code == 404


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("synthetic.pdf", "application/pdf"),
        (
            "synthetic.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
    ],
)
def test_documents_http_lifecycle(
    document_sessions: async_sessionmaker[AsyncSession],
    document_storage: LocalFileStorage,
    filename: str,
    content_type: str,
) -> None:
    async def scenario() -> None:
        async with documents_client(DocumentDAO(document_sessions), document_storage) as api:
            metadata = await upload(api, filename)
            doc_id = metadata["id"]
            assert str(UUID(doc_id)) == doc_id
            row = await read_document(document_sessions, doc_id)
            assert row is not None
            assert metadata == {
                "id": doc_id,
                "session_id": SESSION_ID,
                "filename": filename,
                "content_type": content_type,
                "size_bytes": len(CONTENT),
                "doc_type": "rpd",
                "derived_from": None,
                "created_at": metadata["created_at"],
            }
            assert datetime.fromisoformat(metadata["created_at"]) == row.created_at
            for field in ("session_id", "filename", "content_type", "size_bytes", "doc_type"):
                assert metadata[field] == getattr(row, field)
            assert "storage_key" not in metadata
            assert await document_storage.read(row.storage_key) == CONTENT
            assert stored_files(document_storage) == [
                document_storage.resolve_path(row.storage_key)
            ]

            listing = await api.get(URL)
            assert listing.status_code == 200
            assert listing.json() == {"items": [metadata]}
            assert "storage_key" not in listing.text
            download = await api.get(f"{URL}/{doc_id}")
            assert download.status_code == 200
            assert download.content == CONTENT
            assert download.headers["content-type"] == content_type
            assert download.headers["content-disposition"].startswith("attachment;")

            other = "/api/v1/sessions/synthetic-other-session/documents"
            # Это существующая сессия, а не только неизвестный идентификатор.
            other_upload = await api.post(other, files={"file": ("other.pdf", CONTENT)})
            assert other_upload.status_code == 201
            other_id = other_upload.json()["id"]
            assert (await api.get(f"{other}/{doc_id}")).status_code == 404
            assert (await api.delete(f"{other}/{doc_id}")).status_code == 404
            assert await read_document(document_sessions, doc_id) is not None
            assert (await api.get(f"{URL}/{doc_id}")).content == CONTENT
            assert (await api.delete(f"{other}/{other_id}")).status_code == 204

            deleted = await api.delete(f"{URL}/{doc_id}")
            assert deleted.status_code == 204, deleted.text
            assert deleted.content == b""
            await assert_absent(api, document_sessions, document_storage, doc_id, row.storage_key)

    asyncio.run(scenario())


@pytest.mark.parametrize("run_status", ["waiting_for_input", "queued", "running"])
def test_active_run_allows_upload_but_prevents_delete(
    document_sessions: async_sessionmaker[AsyncSession],
    document_storage: LocalFileStorage,
    run_status: str,
) -> None:
    async def scenario() -> None:
        async with documents_client(DocumentDAO(document_sessions), document_storage) as api:
            await upload(api, "initial.docx")
            run = await SessionRunDAO(document_sessions).create_queued_run(SESSION_ID)
            async with document_sessions() as db:
                await db.execute(update(Run).where(Run.id == run.id).values(status=run_status))
                await db.commit()
            extra = await upload(api)
            row = await read_document(document_sessions, extra["id"])
            assert row is not None
            deleted = await api.delete(f"{URL}/{row.id}")
            assert deleted.status_code == 409, deleted.text
            preserved = await read_document(document_sessions, extra["id"])
            assert preserved is not None
            assert not preserved.deletion_pending
            assert await document_storage.read(row.storage_key) == CONTENT
            download = await api.get(f"{URL}/{row.id}")
            assert download.status_code == 200
            assert download.content == CONTENT
            listing = await api.get(URL)
            assert listing.status_code == 200
            assert len(listing.json()["items"]) == 2
            async with document_sessions() as db:
                assert (await db.get(Run, run.id)).status == run_status

    asyncio.run(scenario())


def test_physical_delete_failure_is_retryable_after_application_restart(
    document_sessions: async_sessionmaker[AsyncSession],
    document_storage: LocalFileStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        async with documents_client(DocumentDAO(document_sessions), document_storage) as api:
            metadata = await upload(api)
            row = await read_document(document_sessions, metadata["id"])
            assert row is not None
            path = document_storage.resolve_path(row.storage_key)
            real_unlink = storage_module.os.unlink
            calls = []

            def fail_document_unlink(name, *, dir_fd=None):
                # Только физический unlink нужного файла. DAO/commit остаются настоящими.
                if name == path.name and dir_fd is not None:
                    calls.append(name)
                    raise OSError(errno.EIO, "synthetic physical deletion failure")
                return real_unlink(name, dir_fd=dir_fd)

            with monkeypatch.context() as fault:
                fault.setattr(storage_module.os, "unlink", fail_document_unlink)
                response = await api.delete(f"{URL}/{row.id}")
            assert response.status_code == 500, response.text
            assert calls == [path.name]
            pending = await read_document(document_sessions, metadata["id"])
            assert pending is not None
            assert pending.deletion_pending
            assert pending.storage_key == row.storage_key
            assert path.read_bytes() == CONTENT

        # Старые app, DAO и storage не передаются новому клиенту. Восстановление
        # опирается только на ту же БД и тот же каталог файлов.
        restarted_storage = LocalFileStorage(document_storage.base_dir)
        async with documents_client(DocumentDAO(document_sessions), restarted_storage) as api:
            response = await api.delete(f"{URL}/{row.id}")
            assert response.status_code == 204, response.text
            await assert_absent(
                api, document_sessions, restarted_storage, metadata["id"], row.storage_key
            )

    asyncio.run(scenario())


def test_upload_failure_before_commit_rolls_back_and_cleans_files(
    document_sessions: async_sessionmaker[AsyncSession],
    document_storage: LocalFileStorage,
) -> None:
    flushed = []
    commits = []

    class FailBeforeCommit(AsyncSession):
        async def flush(self, objects=None):
            await super().flush(objects)
            # INSERT выполнен в настоящей транзакции, но COMMIT ещё не отправлен.
            flushed.append(True)
            assert stored_files(document_storage)
            raise RuntimeError("synthetic write failure before commit")

        async def commit(self):
            commits.append(True)
            await super().commit()

    failing_sessions = async_sessionmaker(
        document_sessions.kw["bind"], class_=FailBeforeCommit, expire_on_commit=False
    )

    async def scenario() -> None:
        async with documents_client(DocumentDAO(failing_sessions), document_storage) as api:
            response = await api.post(URL, files={"file": ("synthetic.pdf", CONTENT)})
            assert response.status_code == 500, response.text
            assert flushed == [True]
            assert commits == []
            async with document_sessions() as db:
                assert list((await db.scalars(select(Document))).all()) == []
            assert stored_files(document_storage) == []
            listing = await api.get(URL)
            assert listing.status_code == 200
            assert listing.json() == {"items": []}

    asyncio.run(scenario())


def test_upload_lost_ack_after_real_commit_preserves_download(
    document_sessions: async_sessionmaker[AsyncSession],
    document_storage: LocalFileStorage,
) -> None:
    committed = []

    class LoseCommitAcknowledgement(AsyncSession):
        async def commit(self):
            await super().commit()
            # Ошибка на границе commit, только ПОСЛЕ успешного PostgreSQL COMMIT.
            committed.append(True)
            raise ConnectionError("synthetic lost acknowledgement after real commit")

    uncertain_sessions = async_sessionmaker(
        document_sessions.kw["bind"], class_=LoseCommitAcknowledgement, expire_on_commit=False
    )

    async def scenario() -> None:
        async with documents_client(DocumentDAO(uncertain_sessions), document_storage) as api:
            metadata = await upload(api)
            assert committed == [True]
            row = await read_document(document_sessions, metadata["id"])
            assert row is not None
            assert await document_storage.read(row.storage_key) == CONTENT
        async with documents_client(
            DocumentDAO(document_sessions), LocalFileStorage(document_storage.base_dir)
        ) as api:
            listing = await api.get(URL)
            assert listing.status_code == 200
            assert listing.json() == {"items": [metadata]}
            download = await api.get(f"{URL}/{row.id}")
            assert download.status_code == 200
            assert download.content == CONTENT
            assert stored_files(document_storage) == [
                document_storage.resolve_path(row.storage_key)
            ]

    asyncio.run(scenario())
