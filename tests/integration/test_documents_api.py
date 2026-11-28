"""Documents API #52: HTTP, миграции PostgreSQL и реальные файлы, включая сбои."""

import asyncio
import errno
import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from edu_doc_agent.api.dependencies import get_document_dao, get_file_storage
from edu_doc_agent.api.documents_router import router
from edu_doc_agent.dao import Document, DocumentDAO, LocalFileStorage, Run, Session, SessionRunDAO
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


@pytest.mark.parametrize("run_status", ["queued", "running", "waiting_for_input"])
def test_logical_deletion_wins_before_new_run(
    document_sessions: async_sessionmaker[AsyncSession],
    document_storage: LocalFileStorage,
    monkeypatch: pytest.MonkeyPatch,
    run_status: str,
) -> None:
    async def scenario() -> None:
        dao = DocumentDAO(document_sessions)
        async with documents_client(dao, document_storage) as api:
            metadata = await upload(api)
            row = await read_document(document_sessions, metadata["id"])
            assert row is not None
            real_delete = document_storage.delete
            visible_during_cleanup = []

            async def start_run_then_unlink(key):
                # Storage I/O начинается после COMMIT логического удаления.
                run = await SessionRunDAO(document_sessions).create_queued_run(SESSION_ID)
                async with document_sessions() as db:
                    await db.execute(update(Run).where(Run.id == run.id).values(status=run_status))
                    await db.commit()
                listing = await api.get(URL)
                download = await api.get(f"{URL}/{row.id}")
                visible_during_cleanup.append((listing.json(), download.status_code))
                return await real_delete(key)

            monkeypatch.setattr(document_storage, "delete", start_run_then_unlink)
            response = await api.delete(f"{URL}/{row.id}")
            print(
                f"DELETE first, {run_status}: HTTP {response.status_code}; "
                f"during cleanup={visible_during_cleanup}; "
                f"file={await document_storage.exists(row.storage_key)}"
            )
            assert response.status_code == 204, response.text
            assert visible_during_cleanup == [({"items": []}, 404)]
            await assert_absent(
                api, document_sessions, document_storage, metadata["id"], row.storage_key
            )

    asyncio.run(scenario())


async def wait_for_session_blocker(engine, blocker_pid: int) -> None:
    """Наблюдаем настоящую блокировку, не предполагаем её после sleep."""
    async with engine.connect() as observer, asyncio.timeout(5):
        while True:
            # pg_stat_activity кэширует снимок до конца транзакции наблюдателя.
            await observer.commit()
            if await observer.scalar(
                text(
                    "SELECT EXISTS (SELECT FROM pg_stat_activity "
                    "WHERE :blocker = ANY(pg_blocking_pids(pid)))"
                ),
                {"blocker": blocker_pid},
            ):
                return
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("run_status", ["queued", "running", "waiting_for_input"])
def test_run_transaction_wins_before_delete(
    document_sessions, document_storage, run_status
) -> None:
    async def scenario() -> None:
        async with documents_client(DocumentDAO(document_sessions), document_storage) as api:
            metadata = await upload(api)
            row = await read_document(document_sessions, metadata["id"])
            async with document_sessions() as writer:
                # FK удерживает KEY SHARE на Session до COMMIT Run.
                pid = await writer.scalar(text("SELECT pg_backend_pid()"))
                await writer.execute(insert(Run).values(session_id=SESSION_ID, status=run_status))
                request = asyncio.create_task(api.delete(f"{URL}/{row.id}"))
                try:
                    await wait_for_session_blocker(document_sessions.kw["bind"], pid)
                    assert not request.done()
                    await writer.commit()
                    response = await request
                finally:
                    request.cancel()
                    await asyncio.gather(request, return_exceptions=True)
            assert response.status_code == 409, response.text
            preserved = await read_document(document_sessions, metadata["id"])
            assert preserved is not None and not preserved.deletion_pending
            assert await document_storage.read(row.storage_key) == CONTENT
            assert (await api.get(URL)).json() == {"items": [metadata]}
            download = await api.get(f"{URL}/{row.id}")
            assert download.status_code == 200 and download.content == CONTENT
            print(f"Run first ({run_status}): DELETE=409; pending=False; list visible; GET=200")

    asyncio.run(scenario())


def test_delete_transaction_wins_before_run_insert(
    document_sessions, document_storage, monkeypatch
) -> None:
    async def scenario() -> None:
        commit_entered = asyncio.Event()
        release_commit = asyncio.Event()
        cleanup_entered = asyncio.Event()
        release_cleanup = asyncio.Event()
        pids = []

        class PauseCommit(AsyncSession):
            async def commit(self):
                await self.flush()  # deletion_pending уже записан, Session заблокирована.
                pids.append(await self.scalar(text("SELECT pg_backend_pid()")))
                commit_entered.set()
                await release_commit.wait()
                await super().commit()

        engine = document_sessions.kw["bind"]
        paused_sessions = async_sessionmaker(engine, class_=PauseCommit, expire_on_commit=False)
        dao = DocumentDAO(paused_sessions)
        real_delete = document_storage.delete

        async def pause_cleanup(key):
            cleanup_entered.set()
            await release_cleanup.wait()
            return await real_delete(key)

        async with documents_client(DocumentDAO(document_sessions), document_storage) as api:
            metadata = await upload(api)
        async with documents_client(dao, document_storage) as api:
            monkeypatch.setattr(document_storage, "delete", pause_cleanup)
            request = asyncio.create_task(api.delete(f"{URL}/{metadata['id']}"))
            run_task = None
            try:
                await asyncio.wait_for(commit_entered.wait(), 5)
                run_task = asyncio.create_task(
                    SessionRunDAO(document_sessions).create_queued_run(SESSION_ID)
                )
                await wait_for_session_blocker(engine, pids[0])
                assert not run_task.done()
                release_commit.set()
                await asyncio.wait_for(run_task, 5)
                await asyncio.wait_for(cleanup_entered.wait(), 5)
                # Run стартовал, но удалённого документа уже нет в доступном пакете.
                assert (await api.get(URL)).json() == {"items": []}
                assert (await api.get(f"{URL}/{metadata['id']}")).status_code == 404
                row = await dao.get_document_for_cleanup(SESSION_ID, UUID(metadata["id"]))
                assert row is not None and await document_storage.exists(row.storage_key)
                # Файловый этап не удерживает даже блокировку Session.
                async with document_sessions() as observer:
                    await observer.execute(text("SET LOCAL lock_timeout = '500ms'"))
                    await observer.execute(
                        select(Session.id).where(Session.id == SESSION_ID).with_for_update()
                    )
                release_cleanup.set()
                assert (await asyncio.wait_for(request, 5)).status_code == 204
                await assert_absent(
                    api, document_sessions, document_storage, metadata["id"], row.storage_key
                )
            finally:
                release_commit.set()
                release_cleanup.set()
                tasks = [request] + ([run_task] if run_task is not None else [])
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("run_status", ["queued", "running", "waiting_for_input"])
@pytest.mark.parametrize("failure", ["storage", "cancel_after_unlink", "confirmation"])
def test_accepted_delete_retries_in_new_process_with_active_run(
    document_sessions, document_storage, monkeypatch, tmp_path, run_status, failure
) -> None:
    async def scenario() -> None:
        dao = DocumentDAO(document_sessions)
        async with documents_client(dao, document_storage) as api:
            metadata = await upload(api)
            row = await read_document(document_sessions, metadata["id"])
            real_delete = document_storage.delete

            async def failed_cleanup(key):
                run = await SessionRunDAO(document_sessions).create_queued_run(SESSION_ID)
                async with document_sessions() as db:
                    await db.execute(update(Run).where(Run.id == run.id).values(status=run_status))
                    await db.commit()
                if failure == "storage":
                    raise PermissionError("synthetic storage permission failure")
                await real_delete(key)
                if failure == "cancel_after_unlink":
                    raise asyncio.CancelledError()

            async def failed_confirmation(*args, **kwargs):
                async with document_sessions() as db:
                    await db.execute(Document.__table__.delete().where(Document.id == row.id))
                    # Настоящий DELETE БД откатывается при выходе, файл уже удалён.
                    raise RuntimeError("synthetic confirmation failure")

            monkeypatch.setattr(document_storage, "delete", failed_cleanup)
            if failure == "confirmation":
                monkeypatch.setattr(dao, "complete_document_deletion", failed_confirmation)
            if failure == "cancel_after_unlink":
                with pytest.raises(asyncio.CancelledError):
                    await api.delete(f"{URL}/{row.id}")
            else:
                assert (await api.delete(f"{URL}/{row.id}")).status_code == 500
            assert (await api.get(URL)).json() == {"items": []}
            assert (await api.get(f"{URL}/{row.id}")).status_code == 404
            pending = await dao.get_document_for_cleanup(SESSION_ID, row.id)
            assert pending is not None and pending.storage_key == row.storage_key
            assert await dao.get_document_for_cleanup("foreign-session", row.id) is None
            assert await document_storage.exists(row.storage_key) is (failure == "storage")
            # Контракт issue #52: ровно два аргумента; активный Run не мешает повтору.
            assert await dao.delete_document(SESSION_ID, row.id) == row.storage_key

        root = Path(__file__).resolve().parents[2]
        environment = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
        environment.update(
            DATABASE_URL=document_sessions.kw["bind"].url.render_as_string(hide_password=False),
            PYTHONPATH=os.pathsep.join((str(root), str(root / "src"))),
            PYTHONDONTWRITEBYTECODE="1",
        )
        code = """
import asyncio, os, sys
from pathlib import Path
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from edu_doc_agent.dao import DocumentDAO, LocalFileStorage
from tests.integration.test_documents_api import documents_client
async def main():
    engine = create_async_engine(os.environ['DATABASE_URL'], poolclass=NullPool)
    try:
        dao = DocumentDAO(async_sessionmaker(engine, expire_on_commit=False))
        async with documents_client(dao, LocalFileStorage(Path(sys.argv[1]))) as api:
            print((await api.delete(sys.argv[2])).status_code)
    finally:
        await engine.dispose()
asyncio.run(main())
"""
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", code, str(document_storage.base_dir), f"{URL}/{row.id}"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == "204", result.stdout + result.stderr
        async with documents_client(DocumentDAO(document_sessions), document_storage) as api:
            await assert_absent(
                api, document_sessions, document_storage, metadata["id"], row.storage_key
            )
            assert await dao.delete_document(SESSION_ID, row.id) is None
        print(f"{failure}, {run_status}: new-process retry=204; DB row absent; file absent")

    asyncio.run(scenario())


def test_document_dao_logical_delete_contract_and_session_isolation(
    document_sessions, document_storage
) -> None:
    async def scenario() -> None:
        dao = DocumentDAO(document_sessions)
        async with documents_client(dao, document_storage) as api:
            metadata = await upload(api)
            row = await read_document(document_sessions, metadata["id"])
            foreign_url = "/api/v1/sessions/foreign-session/documents"
            foreign = await api.post(foreign_url, files={"file": ("foreign.pdf", CONTENT)})
            assert foreign.status_code == 201
            assert await dao.delete_document("foreign-session", row.id) is None
            assert await dao.delete_document(SESSION_ID, uuid4()) is None

            # Issue #52: первый вызов с двумя позиционными аргументами даёт ключ.
            assert await dao.delete_document(SESSION_ID, row.id) == row.storage_key
            assert await dao.get_document(SESSION_ID, row.id) is None
            assert await dao.list_by_session(SESSION_ID) == []
            pending = await dao.get_document_for_cleanup(SESSION_ID, row.id)
            assert pending is not None and pending.storage_key == row.storage_key
            assert await dao.get_document_for_cleanup("foreign-session", row.id) is None
            await SessionRunDAO(document_sessions).create_queued_run(SESSION_ID)
            assert await dao.delete_document(SESSION_ID, row.id) == row.storage_key
            with pytest.raises(RuntimeError):
                await dao.complete_document_deletion(SESSION_ID, row.id, storage_key="wrong-key")
            await dao.complete_document_deletion(
                "foreign-session", row.id, storage_key=row.storage_key
            )
            assert await dao.get_document_for_cleanup(SESSION_ID, row.id) is not None
            await document_storage.delete(row.storage_key)
            for _ in range(2):
                await dao.complete_document_deletion(
                    SESSION_ID, row.id, storage_key=row.storage_key
                )
            assert await dao.get_document_for_cleanup(SESSION_ID, row.id) is None
            assert await dao.delete_document(SESSION_ID, row.id) is None
            assert (await api.get(f"{foreign_url}/{foreign.json()['id']}")).content == CONTENT

    asyncio.run(scenario())


def test_parallel_accepted_deletes_finish_with_new_run(
    document_sessions, document_storage, monkeypatch
) -> None:
    async def scenario() -> None:
        async with documents_client(DocumentDAO(document_sessions), document_storage) as api:
            metadata = await upload(api)
            row = await read_document(document_sessions, metadata["id"])
            both_entered = asyncio.Event()
            release = asyncio.Event()
            calls = []
            real_delete = document_storage.delete

            async def pause_cleanup(key):
                calls.append(key)
                if len(calls) == 2:
                    both_entered.set()
                await release.wait()
                return await real_delete(key)

            monkeypatch.setattr(document_storage, "delete", pause_cleanup)
            requests = [asyncio.create_task(api.delete(f"{URL}/{row.id}")) for _ in range(2)]
            try:
                await asyncio.wait_for(both_entered.wait(), 5)
                await SessionRunDAO(document_sessions).create_queued_run(SESSION_ID)
                assert (await api.get(URL)).json() == {"items": []}
                release.set()
                responses = await asyncio.wait_for(asyncio.gather(*requests), 5)
                assert [r.status_code for r in responses] == [204, 204]
                await assert_absent(
                    api, document_sessions, document_storage, metadata["id"], row.storage_key
                )
            finally:
                release.set()
                for request in requests:
                    request.cancel()
                await asyncio.gather(*requests, return_exceptions=True)

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
