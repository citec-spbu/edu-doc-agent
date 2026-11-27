"""Приём multipart: граница файла, остановка потока и текстовые поля."""

import asyncio
from datetime import UTC, datetime
from tempfile import SpooledTemporaryFile
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette import formparsers

from edu_doc_agent.api import documents_router, multipart
from edu_doc_agent.api.dependencies import get_document_dao, get_file_storage

LIMIT = 512
URL = "/api/v1/sessions/synthetic/documents"
BOUNDARY = b"synthetic-boundary"


@pytest.fixture
def upload_api(monkeypatch):
    settings = SimpleNamespace(max_upload_file_bytes=LIMIT)
    monkeypatch.setattr(multipart, "get_settings", lambda: settings)
    monkeypatch.setattr(documents_router, "get_settings", lambda: settings)
    dao, storage = AsyncMock(), AsyncMock()
    storage.save.return_value = "synthetic/key"

    def created(**values):
        values["id"] = values.pop("doc_id")
        return SimpleNamespace(**values, derived_from=None, created_at=datetime.now(UTC))

    dao.create_document.side_effect = created
    app = FastAPI()
    app.include_router(documents_router.router)
    app.dependency_overrides[get_document_dao] = lambda: dao
    app.dependency_overrides[get_file_storage] = lambda: storage
    return app, dao, storage


def assert_no_effects(dao, storage):
    assert dao.mock_calls == []
    assert storage.mock_calls == []


@pytest.mark.parametrize(
    "size,status", [(LIMIT - 1, 201), (LIMIT, 201), (LIMIT + 1, 413), (0, 400)]
)
def test_file_size_boundary(upload_api, size, status):
    app, dao, storage = upload_api
    with TestClient(app) as client:
        response = client.post(
            URL, files={"file": ("sample.docx", b"x" * size)}, data={"doc_type": "  любой тип  "}
        )
    assert response.status_code == status
    if status == 201:
        assert dao.create_document.call_args.kwargs["size_bytes"] == size
        assert dao.create_document.call_args.kwargs["doc_type"] == "любой тип"
        assert storage.save.call_args_list[0].kwargs["content"] == b"x" * size
    else:
        assert_no_effects(dao, storage)


class IncomingBody(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.received = []

    async def __aiter__(self):
        for chunk in self.chunks:
            self.received.append(chunk)
            yield chunk


@pytest.mark.parametrize("size", [LIMIT - 1, LIMIT, LIMIT + 1, 0])
@pytest.mark.parametrize("doc_type_first", [False, True])
def test_stream_without_content_length(upload_api, monkeypatch, tmp_path, size, doc_type_first):
    app, dao, storage = upload_api
    temporary_files = []

    class ObservedTemporaryFile(SpooledTemporaryFile):
        def __init__(self, **kwargs):
            super().__init__(max_size=1, dir=tmp_path)
            self.bytes_written = 0
            temporary_files.append(self)

        def write(self, data):
            self.bytes_written += len(data)
            return super().write(data)

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", ObservedTemporaryFile)
    doc_type = (
        b"--"
        + BOUNDARY
        + b'\r\nContent-Disposition: form-data; name="doc_type"\r\n\r\nfree text\r\n'
    )
    file_header = (
        b"--"
        + BOUNDARY
        + b'\r\nContent-Disposition: form-data; name="file"; filename="sample.docx"'
        + b"\r\nContent-Type: application/octet-stream\r\n\r\n"
    )
    header = (doc_type if doc_type_first else b"") + file_header
    tail = b"\r\n" + (b"" if doc_type_first else doc_type) + b"--" + BOUNDARY + b"--\r\n"
    body = IncomingBody(
        [header, b"x" * min(size, LIMIT - 1)] + [b"x"] * max(0, size - LIMIT + 1) + [tail]
    )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            request = client.build_request(
                "POST",
                URL,
                content=body,
                headers={"Content-Type": "multipart/form-data; boundary=" + BOUNDARY.decode()},
            )
            assert "content-length" not in request.headers
            return await client.send(request)

    response = asyncio.run(scenario())
    assert len(temporary_files) == 1
    assert all(file.closed for file in temporary_files)
    assert not list(tmp_path.iterdir())
    if size > LIMIT:
        assert response.status_code == 413
        # Вход остановился на первом байте сверх лимита: оставшийся multipart
        # (в том числе doc_type после файла) сервер даже не запросил.
        assert body.received == body.chunks[:-1]
        assert temporary_files[0].bytes_written == LIMIT
        assert_no_effects(dao, storage)
    elif size == 0:
        assert response.status_code == 400
        assert_no_effects(dao, storage)
    else:
        assert response.status_code == 201
        assert body.received == body.chunks
        assert temporary_files[0].bytes_written == size
        assert dao.create_document.call_args.kwargs["doc_type"] == "free text"


def test_doc_type_nul_rejected_before_storage(upload_api):
    app, dao, storage = upload_api
    with TestClient(app) as client:
        response = client.post(
            URL, files={"file": ("sample.docx", b"x")}, data={"doc_type": "a\x00b"}
        )
    assert response.status_code in {400, 422}
    assert_no_effects(dao, storage)


@pytest.mark.parametrize("session_id", ["%20%20", "%00", "a%00b"])
@pytest.mark.parametrize(
    "method,item", [("POST", False), ("GET", False), ("GET", True), ("DELETE", True)]
)
def test_session_id_validation_all_routes(upload_api, session_id, method, item):
    app, dao, storage = upload_api
    url = f"/api/v1/sessions/{session_id}/documents" + (f"/{uuid4()}" if item else "")
    kwargs = {"files": {"file": ("sample.docx", b"x")}} if method == "POST" else {}
    with TestClient(app) as client:
        response = client.request(method, url, **kwargs)
    assert response.status_code == 400
    assert_no_effects(dao, storage)


def test_swagger_multipart_contract(upload_api):
    app, _, _ = upload_api
    schema = app.openapi()
    body = schema["paths"]["/api/v1/sessions/{session_id}/documents"]["post"]["requestBody"]
    ref = body["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    fields = schema["components"]["schemas"][ref]
    assert fields["properties"]["file"]["contentMediaType"] == "application/octet-stream"
    assert "doc_type" in fields["properties"]
    assert "file" in fields["required"]


@pytest.mark.parametrize("value", ["", " ", "\x00", "a\x00b"])
def test_empty_and_invalid_session_id(value):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        documents_router._validate_session_id(value)
    assert error.value.status_code == 400


@pytest.mark.parametrize(
    "files,data",
    [
        (None, {"file": "text instead of file"}),
        ({"file": ("sample.docx", b"x"), "doc_type": ("type.txt", b"rpd")}, None),
    ],
)
def test_invalid_form_field_types(upload_api, files, data):
    app, dao, storage = upload_api
    with TestClient(app) as client:
        response = client.post(URL, files=files, data=data)
    assert response.status_code == 422
    assert_no_effects(dao, storage)


@pytest.mark.parametrize("filename", ["../sample.docx", "dir\\sample.docx", " "])
def test_filename_validation_preserved(upload_api, filename):
    app, dao, storage = upload_api
    with TestClient(app) as client:
        response = client.post(URL, files={"file": (filename, b"x")})
    assert response.status_code == 400
    assert_no_effects(dao, storage)


def test_raw_nul_filename_rejected(upload_api):
    app, dao, storage = upload_api
    # httpx кодирует NUL в filename как %00; здесь нужен настоящий байт NUL.
    body = (
        b'--test\r\nContent-Disposition: form-data; name="file"; filename="bad\x00.docx"'
        b"\r\n\r\nx\r\n--test--\r\n"
    )
    with TestClient(app) as client:
        response = client.post(
            URL, content=body, headers={"Content-Type": "multipart/form-data; boundary=test"}
        )
    assert response.status_code == 400
    assert_no_effects(dao, storage)
