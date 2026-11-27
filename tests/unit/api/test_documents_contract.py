"""Публичный HTTP-контракт Documents API (#52, PR #53), без БД и диска."""

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import quote, unquote
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from edu_doc_agent.api import documents_router, multipart
from edu_doc_agent.api.dependencies import get_document_dao, get_file_storage
from edu_doc_agent.dao import ActiveRunConflictError, DocumentDAO, FileStorage
from edu_doc_agent.dao.documents import DocumentCreateRolledBackError
from edu_doc_agent.dao.storage import StorageFileNotFoundError
from edu_doc_agent.main import app

URL = "/api/v1/sessions/synthetic-session/documents"
CONTENT = b"synthetic document\x00\xff\r\n"
LIMIT = 128
FORMATS = [
    ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ("pdf", "application/pdf"),
]


@pytest.fixture
def documents_api(monkeypatch):
    """Stateful doubles implement storage/DAO boundaries, not HTTP responses."""
    state = SimpleNamespace(
        dao=AsyncMock(spec=DocumentDAO),
        storage=AsyncMock(spec=FileStorage),
        rows={},
        files={},
        active_run=None,
        events=[],
    )

    async def save(*, session_id, doc_id, filename, content):
        key = f"{session_id}/{doc_id}/{filename}"
        state.files[key] = content
        state.events.append(("save", key))
        return key

    async def read(key):
        if key not in state.files:
            raise StorageFileNotFoundError("synthetic missing file")
        return state.files[key]

    async def remove_file(key):
        state.events.append(("cleanup", key))
        return state.files.pop(key, None) is not None

    async def create(**values):
        values["id"] = values.pop("doc_id")
        doc = SimpleNamespace(
            **values, derived_from=None, created_at=datetime.now(UTC), deletion_pending=False
        )
        state.rows[doc.id] = doc
        state.events.append(("metadata", doc.storage_key))
        return doc

    async def get_document(session_id, doc_id):
        doc = state.rows.get(doc_id)
        return (
            doc
            if doc is not None and doc.session_id == session_id and not doc.deletion_pending
            else None
        )

    async def list_documents(session_id):
        return [
            doc
            for doc in state.rows.values()
            if doc.session_id == session_id and not doc.deletion_pending
        ]

    async def delete_document(session_id, doc_id):
        doc = state.rows.get(doc_id)
        if doc is None or doc.session_id != session_id:
            return None
        if doc.deletion_pending:
            return doc.storage_key
        # The real DAO owns Run lookup; model its conflict at the API boundary.
        if state.active_run in {"queued", "running", "waiting_for_input"}:
            raise ActiveRunConflictError(state.active_run)
        doc.deletion_pending = True
        return doc.storage_key

    async def complete_document_deletion(*, session_id, doc_id, storage_key):
        doc = state.rows.get(doc_id)
        assert doc is not None and doc.storage_key == storage_key
        assert doc.session_id == session_id and doc.deletion_pending
        assert storage_key not in state.files
        del state.rows[doc_id]

    state.storage.save.side_effect = save
    state.storage.read.side_effect = read
    state.storage.delete.side_effect = remove_file
    state.dao.create_document.side_effect = create
    state.dao.get_document.side_effect = get_document
    state.dao.list_by_session.side_effect = list_documents
    state.dao.delete_document.side_effect = delete_document
    state.dao.complete_document_deletion.side_effect = complete_document_deletion
    settings = SimpleNamespace(max_upload_file_bytes=LIMIT)
    monkeypatch.setattr(documents_router, "get_settings", lambda: settings)
    monkeypatch.setattr(multipart, "get_settings", lambda: settings)
    monkeypatch.setitem(app.dependency_overrides, get_document_dao, lambda: state.dao)
    monkeypatch.setitem(app.dependency_overrides, get_file_storage, lambda: state.storage)
    # No lifespan: these routes need neither Redis nor a database connection.
    state.client = TestClient(app, raise_server_exceptions=False)
    try:
        yield state
    finally:
        state.client.close()


def upload(api, filename="sample.docx", content=CONTENT, **kwargs):
    return api.client.post(
        URL, files={"file": (filename, content, "application/octet-stream")}, **kwargs
    )


def assert_no_writes(api):
    api.storage.save.assert_not_awaited()
    api.dao.create_document.assert_not_awaited()
    assert api.files == {}
    assert api.rows == {}


@pytest.mark.parametrize("extension,mime", FORMATS)
def test_post_accepts_formats_and_returns_canonical_mime(documents_api, extension, mime):
    api = documents_api
    response = upload(api, f"sample.{extension}")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["content_type"] == mime
    assert body["filename"] == f"sample.{extension}"
    assert body["session_id"] == "synthetic-session"
    assert body["size_bytes"] == len(CONTENT)
    assert "storage_key" not in body
    doc = api.rows[UUID(body["id"])]
    assert api.files[doc.storage_key] == CONTENT
    api.dao.create_document.assert_awaited_once()


@pytest.mark.parametrize(
    "filename,content,statuses",
    [("sample.docx", b"", {400, 422}), ("sample.txt", CONTENT, {415})],
)
def test_invalid_upload_has_no_persistence(documents_api, filename, content, statuses):
    response = upload(documents_api, filename, content)
    assert response.status_code in statuses, response.text
    assert_no_writes(documents_api)


@pytest.mark.parametrize("size,status", [(LIMIT, 201), (LIMIT + 1, 413)])
@pytest.mark.parametrize("content_length", [True, False], ids=["length", "stream"])
def test_post_file_limit_with_and_without_content_length(
    documents_api, size, status, content_length
):
    api = documents_api
    payload = b"x" * size
    if content_length:
        response = upload(api, content=payload)
        assert "content-length" in response.request.headers
    else:

        async def chunks():
            yield (
                b'--contract\r\nContent-Disposition: form-data; name="file"; '
                b'filename="sample.docx"\r\nContent-Type: application/octet-stream\r\n\r\n'
            )
            yield payload[:LIMIT]
            yield payload[LIMIT:]
            yield b"\r\n--contract--\r\n"

        async def scenario():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                request = client.build_request(
                    "POST",
                    URL,
                    content=chunks(),
                    headers={"Content-Type": "multipart/form-data; boundary=contract"},
                )
                assert "content-length" not in request.headers
                return await client.send(request)

        response = asyncio.run(scenario())
    assert response.status_code == status, response.text
    if status == 201:
        doc = next(iter(api.rows.values()))
        assert doc.size_bytes == size
        assert api.files[doc.storage_key] == payload
    else:
        assert_no_writes(api)


@pytest.mark.parametrize("session_id", ["%20%20", "%00", "bad%00session"])
@pytest.mark.parametrize(
    "method,item", [("POST", False), ("GET", False), ("GET", True), ("DELETE", True)]
)
def test_invalid_session_id_is_rejected_on_each_route(documents_api, session_id, method, item):
    api = documents_api
    url = f"/api/v1/sessions/{session_id}/documents" + (f"/{uuid4()}" if item else "")
    kwargs = {"files": {"file": ("sample.docx", CONTENT)}} if method == "POST" else {}
    response = api.client.request(method, url, **kwargs)
    assert 400 <= response.status_code < 500, response.text
    assert_no_writes(api)
    assert api.dao.mock_calls == []
    assert api.storage.mock_calls == []


def test_nul_doc_type_is_rejected_before_persistence(documents_api):
    response = upload(documents_api, data={"doc_type": "rpd\x00plan"})
    assert 400 <= response.status_code < 500, response.text
    assert_no_writes(documents_api)


def test_confirmed_database_rollback_cleans_up_saved_file(documents_api):
    api = documents_api

    async def rollback(**values):
        assert api.files[values["storage_key"]] == CONTENT
        api.events.append(("rollback", values["storage_key"]))
        raise DocumentCreateRolledBackError("synthetic confirmed rollback")

    api.dao.create_document.side_effect = rollback
    response = upload(api)
    assert 500 <= response.status_code < 600, response.text
    api.dao.create_document.assert_awaited_once()
    key = api.dao.create_document.call_args.kwargs["storage_key"]
    assert api.events.index(("save", key)) < api.events.index(("rollback", key))
    assert api.events.index(("rollback", key)) < api.events.index(("cleanup", key))
    assert api.files == {}
    assert api.rows == {}


def test_lost_commit_acknowledgement_preserves_downloadable_document(documents_api):
    api = documents_api
    create = api.dao.create_document.side_effect

    async def committed_without_reply(**values):
        await create(**values)
        raise ConnectionError("synthetic lost commit acknowledgement")

    api.dao.create_document.side_effect = committed_without_reply
    upload(api)
    api.dao.create_document.assert_awaited_once()
    doc = next(iter(api.rows.values()))
    assert all(call.args != (doc.storage_key,) for call in api.storage.delete.await_args_list)
    response = api.client.get(f"{URL}/{doc.id}")
    assert response.status_code == 200, response.text
    assert response.content == CONTENT


def test_cancellation_during_save_leaves_no_file_without_metadata(documents_api):
    api = documents_api
    save = api.storage.save.side_effect

    async def scenario():
        written = asyncio.Event()
        release = asyncio.Event()

        async def blocked_save(**values):
            key = await save(**values)
            if values["filename"] == "sample.docx":
                written.set()
                await release.wait()
            return key

        api.storage.save.side_effect = blocked_save
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            request = asyncio.create_task(
                client.post(URL, files={"file": ("sample.docx", CONTENT)})
            )
            try:
                await asyncio.wait_for(written.wait(), timeout=5)
                assert api.files and not api.rows
                request.cancel()
                # cancel is requested while save is blocked; release needs no sleep.
                release.set()
                with suppress(asyncio.CancelledError):
                    await asyncio.wait_for(request, timeout=5)
                # Metadata may finish or the saved file may be compensated.
                document_keys = {doc.storage_key for doc in api.rows.values()}
                assert set(api.files) <= document_keys
                for doc in api.rows.values():
                    response = await client.get(f"{URL}/{doc.id}")
                    assert response.status_code == 200
                    assert response.content == CONTENT
            finally:
                release.set()
                request.cancel()
                with suppress(asyncio.CancelledError):
                    await request

    asyncio.run(scenario())


def test_list_exposes_public_metadata_without_storage_key(documents_api):
    api = documents_api
    created = upload(api).json()
    response = api.client.get(URL)
    assert response.status_code == 200
    assert response.json() == {"items": [created]}
    assert "storage_key" not in response.text
    api.dao.list_by_session.assert_awaited_once_with("synthetic-session")


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_foreign_document_without_active_run_returns_404(documents_api, method):
    api = documents_api
    doc_id = UUID(upload(api).json()["id"])
    doc = api.rows[doc_id]
    api.storage.reset_mock()
    url = f"/api/v1/sessions/other-session/documents/{doc_id}"
    response = api.client.request(method, url)
    assert response.status_code == 404, response.text
    lookup = api.dao.get_document if method == "GET" else api.dao.delete_document
    lookup.assert_awaited_once_with(session_id="other-session", doc_id=doc_id)
    assert api.storage.mock_calls == []
    assert api.rows[doc_id] is doc
    assert api.files[doc.storage_key] == CONTENT


@pytest.mark.parametrize("extension,mime", FORMATS)
def test_download_returns_original_bytes_mime_and_filename(documents_api, extension, mime):
    api = documents_api
    filename = f"Учебный план.{extension}"
    doc_id = upload(api, filename).json()["id"]
    response = api.client.get(f"{URL}/{doc_id}")
    assert response.status_code == 200, response.text
    assert response.content == CONTENT
    assert response.headers["content-type"] == mime
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert unquote(disposition.split("filename*=UTF-8''", 1)[1]) == filename
    assert quote(filename, safe="") in disposition


def test_missing_file_with_existing_metadata_returns_500(documents_api):
    api = documents_api
    doc_id = upload(api).json()["id"]
    api.files.clear()
    response = api.client.get(f"{URL}/{doc_id}")
    assert response.status_code == 500, response.text
    assert UUID(doc_id) in api.rows


def test_openapi_download_describes_binary_for_all_supported_mimes(documents_api):
    response = documents_api.client.get("/openapi.json")
    assert response.status_code == 200
    path = "/api/v1/sessions/{session_id}/documents/{doc_id}"
    content = response.json()["paths"][path]["get"]["responses"]["200"]["content"]
    for _, mime in FORMATS:
        assert content[mime]["schema"] == {"type": "string", "format": "binary"}


@pytest.mark.parametrize("run_status", ["queued", "running", "waiting_for_input"])
def test_delete_with_active_run_returns_409(documents_api, run_status):
    api = documents_api
    doc_id = UUID(upload(api).json()["id"])
    api.active_run = run_status
    api.storage.reset_mock()
    response = api.client.delete(f"{URL}/{doc_id}")
    assert response.status_code == 409, response.text
    assert api.storage.mock_calls == []
    api.dao.delete_document.assert_awaited_once_with(session_id="synthetic-session", doc_id=doc_id)
    api.dao.complete_document_deletion.assert_not_awaited()
    assert doc_id in api.rows


def test_post_is_allowed_while_waiting_for_input(documents_api):
    api = documents_api
    api.active_run = "waiting_for_input"
    response = upload(api)
    assert response.status_code == 201, response.text
    api.dao.create_document.assert_awaited_once()
    api.dao.delete_document.assert_not_awaited()
    assert api.client.get(f"{URL}/{response.json()['id']}").content == CONTENT


def test_delete_cleanup_failure_is_reported_and_retry_completes(documents_api):
    api = documents_api
    doc_id = UUID(upload(api).json()["id"])
    key = api.rows[doc_id].storage_key
    delete = api.storage.delete.side_effect
    api.storage.delete.side_effect = OSError("synthetic cleanup unavailable")
    failed = api.client.delete(f"{URL}/{doc_id}")
    assert failed.status_code == 500, failed.text
    api.dao.complete_document_deletion.assert_not_awaited()
    assert doc_id in api.rows
    assert api.files[key] == CONTENT
    assert api.client.get(URL).json() == {"items": []}
    assert api.client.get(f"{URL}/{doc_id}").status_code == 404
    api.storage.delete.side_effect = delete
    retried = api.client.delete(f"{URL}/{doc_id}")
    assert retried.status_code == 204, retried.text
    assert retried.content == b""
    assert key not in api.files
    assert doc_id not in api.rows
    api.dao.complete_document_deletion.assert_awaited_once_with(
        session_id="synthetic-session", doc_id=doc_id, storage_key=key
    )
