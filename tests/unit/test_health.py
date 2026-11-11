"""Тесты для /health — проверка, что приложение поднимается и отвечает."""

from fastapi.testclient import TestClient

from edu_doc_agent import __version__


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


def test_openapi_schema_available(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "edu-doc-agent"
    assert "/health" in schema["paths"]


def test_root_still_redirects_to_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
    assert "/" not in client.get("/openapi.json").json()["paths"]
