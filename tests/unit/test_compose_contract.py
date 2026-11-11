"""Контракты развёртывания очереди, проверяемые без Docker daemon."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def services() -> dict:
    # PyYAML уже входит в dev-окружение через pre-commit.
    return yaml.safe_load((ROOT / "compose.yaml").read_text())["services"]


def test_api_and_worker_share_redis_and_storage(services: dict) -> None:
    api, worker, redis = (services[name] for name in ("api", "worker", "redis"))

    assert api["environment"]["REDIS_URL"] == worker["environment"]["REDIS_URL"]
    assert api["environment"]["REDIS_URL"].startswith("redis://redis:6379/")
    for service in (api, worker):
        assert service["depends_on"]["redis"]["condition"] == "service_healthy"
        assert "./storage:/app/storage" in service["volumes"]
    assert redis["healthcheck"]["test"] == ["CMD", "redis-cli", "ping"]
    assert worker["command"] == ["arq", "edu_doc_agent.worker.settings.WorkerSettings"]


# PR #29 P1: https://github.com/citec-spbu/edu-doc-agent/pull/29#discussion_r4478019738
@pytest.mark.pr_review
def test_redis_is_not_published_on_public_interfaces(services: dict) -> None:
    for port in services["redis"].get("ports", []):
        if isinstance(port, dict):
            assert port.get("host_ip") in {"127.0.0.1", "::1"}
        else:
            assert str(port).startswith(("127.0.0.1:", "[::1]:"))
