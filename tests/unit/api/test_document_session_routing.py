"""HTTP-контракт пустого session_id в Documents API."""

from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edu_doc_agent.api.dependencies import get_document_dao, get_file_storage
from edu_doc_agent.api.documents_router import router

DOC_ID = UUID("10000000-0000-4000-8000-000000000001")


@pytest.fixture
def documents_api():
    app = FastAPI()
    app.include_router(router)
    dao, storage = AsyncMock(), AsyncMock()
    dao.get_document.return_value = None
    dao.delete_document.return_value = None
    dao_provider = Mock(return_value=dao)
    storage_provider = Mock(return_value=storage)
    app.dependency_overrides[get_document_dao] = lambda: dao_provider()
    app.dependency_overrides[get_file_storage] = lambda: storage_provider()
    with TestClient(app) as client:
        yield client, dao, storage, dao_provider, storage_provider


@pytest.mark.parametrize(
    "method,suffix",
    [("GET", ""), ("POST", ""), ("GET", f"/{DOC_ID}"), ("DELETE", f"/{DOC_ID}")],
)
def test_empty_session_id_returns_400_without_dependencies(documents_api, method, suffix):
    client, dao, storage, dao_provider, storage_provider = documents_api
    # Даже POST без тела должен отклоняться по session_id, а не по File/Form.
    response = client.request(method, f"/api/v1/sessions//documents{suffix}")
    assert response.status_code == 400
    assert response.json() == {"detail": "Некорректный идентификатор сессии."}
    assert not response.history
    assert dao.mock_calls == []
    assert storage.mock_calls == []
    dao_provider.assert_not_called()
    storage_provider.assert_not_called()


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/v1/sessions//unknown"),
        ("POST", "/api/v1/sessions//process"),
        ("GET", "/api/v1/sessions//documents/not-a-uuid"),
        ("DELETE", f"/api/v1/sessions//documents/{DOC_ID}/unknown"),
        ("GET", "/unknown"),
    ],
)
def test_unknown_route_remains_404(documents_api, method, path):
    client, dao, storage, dao_provider, storage_provider = documents_api
    response = client.request(method, path)
    assert response.status_code == 404
    assert dao.mock_calls == []
    assert storage.mock_calls == []
    dao_provider.assert_not_called()
    storage_provider.assert_not_called()


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_missing_document_with_valid_session_remains_404(documents_api, method):
    client, dao, storage, _, _ = documents_api
    response = client.request(method, f"/api/v1/sessions/synthetic/documents/{DOC_ID}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Документ не найден в рамках указанной сессии."}
    operation = dao.get_document if method == "GET" else dao.delete_document
    operation.assert_awaited_once_with(session_id="synthetic", doc_id=DOC_ID)
    assert storage.mock_calls == []


def test_swagger_contains_only_regular_document_routes(documents_api):
    client, _, _, _, _ = documents_api
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert set(paths) == {
        "/api/v1/sessions/{session_id}/documents",
        "/api/v1/sessions/{session_id}/documents/{doc_id}",
    }
    assert set(paths["/api/v1/sessions/{session_id}/documents"]) == {"get", "post"}
    assert set(paths["/api/v1/sessions/{session_id}/documents/{doc_id}"]) == {"get", "delete"}
