"""Порядок DELETE/Run и восстановление POST через публичные контракты PR #53."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from edu_doc_agent.dao import Document, DocumentDAO, LocalFileStorage, Run, SessionRunDAO
from tests.integration.dao.support import Database
from tests.integration.test_documents_api import (
    CONTENT,
    SESSION_ID,
    URL,
    assert_absent,
    documents_client,
    read_document,
    stored_files,
    upload,
)
from tests.integration.test_documents_api import document_sessions as document_sessions
from tests.integration.test_documents_api import document_storage as document_storage

pytestmark = pytest.mark.integration


def test_delete_commit_precedes_run_and_hides_document_from_new_consumer(
    document_sessions: async_sessionmaker[AsyncSession],
    document_storage: LocalFileStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        cleanup_entered, release_cleanup = asyncio.Event(), asyncio.Event()
        async with documents_client(DocumentDAO(document_sessions), document_storage) as api:
            metadata = await upload(api)
            row = await read_document(document_sessions, metadata["id"])
            assert row is not None
            real_delete = document_storage.delete

            async def pause_physical_cleanup(key: str) -> bool:
                # HTTP уже зафиксировал логическое удаление, но unlink ещё не начат.
                assert key == row.storage_key
                cleanup_entered.set()
                await release_cleanup.wait()
                return await real_delete(key)

            with monkeypatch.context() as boundary:
                boundary.setattr(document_storage, "delete", pause_physical_cleanup)
                deleting = asyncio.create_task(api.delete(f"{URL}/{row.id}"))
                try:
                    await asyncio.wait_for(cleanup_entered.wait(), 5)
                    pending = await read_document(document_sessions, metadata["id"])
                    assert pending is not None and pending.deletion_pending
                    assert await document_storage.read(row.storage_key) == CONTENT
                    run = await asyncio.wait_for(
                        SessionRunDAO(document_sessions).create_queued_run(SESSION_ID), 5
                    )
                    async with document_sessions() as db:
                        assert (await db.get(Run, run.id)).status == "queued"

                    # Новый потребитель не использует экземпляры DAO/storage удаляющего API.
                    async with documents_client(
                        DocumentDAO(document_sessions), LocalFileStorage(document_storage.base_dir)
                    ) as consumer:
                        listing = await consumer.get(URL)
                        download = await consumer.get(f"{URL}/{row.id}")
                        assert listing.status_code == 200
                        assert listing.json() == {"items": []}
                        assert download.status_code == 404
                    assert not deleting.done()
                    release_cleanup.set()
                    response = await asyncio.wait_for(deleting, 5)
                    assert response.status_code == 204, response.text
                    await assert_absent(
                        api, document_sessions, document_storage, metadata["id"], row.storage_key
                    )
                    async with document_sessions() as db:
                        assert (await db.get(Run, run.id)).status == "queued"
                finally:
                    release_cleanup.set()
                    if not deleting.done():
                        deleting.cancel()
                    await asyncio.gather(deleting, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("run_status", ["queued", "running", "waiting_for_input"])
def test_committed_run_precedes_delete_and_preserves_original_document(
    document_sessions: async_sessionmaker[AsyncSession],
    document_storage: LocalFileStorage,
    run_status: str,
) -> None:
    async def scenario() -> None:
        async with documents_client(DocumentDAO(document_sessions), document_storage) as api:
            metadata = await upload(api)
            row = await read_document(document_sessions, metadata["id"])
            assert row is not None
            run = await SessionRunDAO(document_sessions).create_queued_run(SESSION_ID)
            if run_status != "queued":
                async with document_sessions() as db:
                    await db.execute(update(Run).where(Run.id == run.id).values(status=run_status))
                    await db.commit()
            response = await api.delete(f"{URL}/{row.id}")
            assert response.status_code == 409, response.text
            preserved = await read_document(document_sessions, metadata["id"])
            assert preserved is not None and not preserved.deletion_pending
            assert preserved.storage_key == row.storage_key
            assert stored_files(document_storage) == [
                document_storage.resolve_path(row.storage_key)
            ]
            assert await document_storage.read(row.storage_key) == CONTENT
            listing = await api.get(URL)
            assert listing.status_code == 200
            assert listing.json() == {"items": [metadata]}
            download = await api.get(f"{URL}/{row.id}")
            assert download.status_code == 200
            assert download.content == CONTENT
            async with document_sessions() as db:
                assert (await db.get(Run, run.id)).status == run_status

    asyncio.run(scenario())


def test_public_delete_dao_two_argument_contract_survives_recreation(
    document_sessions: async_sessionmaker[AsyncSession],
    document_storage: LocalFileStorage,
) -> None:
    async def scenario() -> None:
        async with documents_client(DocumentDAO(document_sessions), document_storage) as api:
            metadata = await upload(api)
            row = await read_document(document_sessions, metadata["id"])
            assert row is not None
            async with document_sessions() as db:
                assert (await db.scalars(select(Run))).all() == []
            # Ровно два позиционных аргумента; ни keyword storage_key, ни внутренний API.
            original_key = await DocumentDAO(document_sessions).delete_document(SESSION_ID, row.id)
            assert original_key == row.storage_key
            pending = await read_document(document_sessions, metadata["id"])
            assert pending is not None and pending.deletion_pending
            recreated = DocumentDAO(document_sessions)
            assert await recreated.delete_document(SESSION_ID, row.id) == original_key
            assert await document_storage.read(original_key) == CONTENT
            assert await recreated.delete_document(SESSION_ID, uuid4()) is None
            # Незавершённая очистка остаётся повторяемой через HTTP нового приложения.
        async with documents_client(recreated, LocalFileStorage(document_storage.base_dir)) as api:
            assert (await api.delete(f"{URL}/{row.id}")).status_code == 204
            await assert_absent(
                api, document_sessions, document_storage, metadata["id"], original_key
            )
            assert await recreated.delete_document(SESSION_ID, row.id) is None

    asyncio.run(scenario())


async def run_reconciliation_cli(
    database: Database, storage: LocalFileStorage, directory: Path, expected: dict[str, int]
) -> None:
    # CLI читает DATABASE_URL, включая search_path той же изолированной схемы.
    # cwd временный: рабочий .env не читается. Штатные зависимости не подменяются.
    environment = {
        name: os.environ[name] for name in ("PATH", "HOME", "SYSTEMROOT") if name in os.environ
    }
    environment.update(
        DATABASE_URL=database.url.render_as_string(hide_password=False),
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(directory),
    )
    result = await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            "-m",
            "edu_doc_agent.reconcile_uploads",
            "--storage-dir",
            str(storage.base_dir),
        ],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == expected


@pytest.mark.parametrize("order", ["reconcile-download-delete", "delete-reconcile"])
def test_committed_upload_recovers_in_new_app_with_cli_in_either_order(
    document_sessions: async_sessionmaker[AsyncSession],
    document_storage: LocalFileStorage,
    migrated_database: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    order: str,
) -> None:
    commits, selects = [], []

    class LoseCommitAcknowledgement(AsyncSession):
        async def commit(self):
            await super().commit()
            # Сбой только ПОСЛЕ настоящего успешного COMMIT PostgreSQL.
            commits.append(True)
            raise ConnectionError("synthetic lost commit acknowledgement")

    uncertain_sessions = async_sessionmaker(
        document_sessions.kw["bind"], class_=LoseCommitAcknowledgement, expire_on_commit=False
    )
    uncertain_dao = DocumentDAO(uncertain_sessions)
    real_select = uncertain_dao.get_document

    async def first_select_unavailable(*args, **kwargs):
        selects.append(True)
        if len(selects) == 1:
            raise ConnectionError("synthetic first reconciliation SELECT unavailable")
        return await real_select(*args, **kwargs)

    async def scenario() -> None:
        with monkeypatch.context() as boundary:
            boundary.setattr(uncertain_dao, "get_document", first_select_unavailable)
            async with documents_client(uncertain_dao, document_storage) as api:
                response = await api.post(URL, files={"file": ("synthetic.pdf", CONTENT)})
                assert response.status_code == 503, response.text
        assert commits == [True]
        assert selects == [True]
        async with document_sessions() as db:
            row = (await db.scalars(select(Document))).one()
        assert not row.deletion_pending
        assert await document_storage.read(row.storage_key) == CONTENT
        markers = await document_storage.list_upload_intents()
        assert len(markers) == 1
        assert len(stored_files(document_storage)) == 2

        recovered_storage = LocalFileStorage(document_storage.base_dir)
        async with documents_client(DocumentDAO(document_sessions), recovered_storage) as api:
            if order == "reconcile-download-delete":
                await run_reconciliation_cli(
                    migrated_database, recovered_storage, tmp_path, {"confirmed": 1}
                )
                assert await recovered_storage.list_upload_intents() == []
                assert stored_files(recovered_storage) == [
                    recovered_storage.resolve_path(row.storage_key)
                ]
                download = await api.get(f"{URL}/{row.id}")
                assert download.status_code == 200
                assert download.content == CONTENT
                assert await read_document(document_sessions, str(row.id)) is not None
                await run_reconciliation_cli(migrated_database, recovered_storage, tmp_path, {})
                assert await recovered_storage.read(row.storage_key) == CONTENT

            response = await api.delete(f"{URL}/{row.id}")
            assert response.status_code == 204, response.text
            await assert_absent(
                api, document_sessions, recovered_storage, str(row.id), row.storage_key
            )
            for _ in range(2):
                await run_reconciliation_cli(migrated_database, recovered_storage, tmp_path, {})
                assert stored_files(recovered_storage) == []
                assert await read_document(document_sessions, str(row.id)) is None

    asyncio.run(scenario())
