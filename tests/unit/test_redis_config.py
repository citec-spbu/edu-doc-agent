"""Конфигурация подключения к очереди без настоящего .env и Redis."""

import os

import pytest

from edu_doc_agent.config import Settings


def test_default_redis_url(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.lower() == "redis_url":
            monkeypatch.delenv(name)
    settings = Settings(_env_file=None)

    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.arq_redis_settings.host == "localhost"
    assert settings.arq_redis_settings.port == 6379
    assert settings.arq_redis_settings.database == 0


def test_redis_url_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://queue.example:6380/4")
    connection = Settings(_env_file=None).arq_redis_settings

    assert (connection.host, connection.port, connection.database) == ("queue.example", 6380, 4)


@pytest.mark.parametrize("scheme, ssl", [("redis", False), ("rediss", True)])
def test_redis_dsn_preserves_connection_fields(scheme: str, ssl: bool) -> None:
    settings = Settings(
        _env_file=None, redis_url=f"{scheme}://test-user:synthetic-password@queue.example:6381/3"
    )
    connection = settings.arq_redis_settings

    assert connection.host == "queue.example"
    assert connection.port == 6381
    assert connection.database == 3
    assert connection.username == "test-user"
    assert connection.password == "synthetic-password"
    assert connection.ssl is ssl


@pytest.mark.parametrize("dsn", ["http://queue.example:6379", "redis://localhost/not-a-db"])
def test_invalid_redis_dsn_is_rejected(dsn: str) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        _ = Settings(_env_file=None, redis_url=dsn).arq_redis_settings
