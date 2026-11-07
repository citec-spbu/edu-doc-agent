"""Клиент OpenAI-совместимого API: запрос, повторы, лимиты. Сеть не используется."""

import json

import httpx
import pytest

from edu_doc_agent.config import Settings
from edu_doc_agent.llm import LLMError, OpenAICompatibleClient, openai_compat

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def _client(handler, **kwargs) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url="https://llm.example/v1",
        api_key="sk-test",
        model="m",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _ok(content: str = '{"ok": true}') -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(openai_compat.time, "sleep", slept.append)
    return slept


def test_request_has_schema_zero_temperature_and_no_thinking() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok()

    assert _client(handler).complete_json(system="s", user="u", schema=SCHEMA) == '{"ok": true}'
    (request,) = seen
    body = json.loads(request.content)
    assert str(request.url) == "https://llm.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert body["temperature"] == 0 and body["model"] == "m"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_rate_limit_waits_retry_after_then_succeeds(_no_sleep: list[float]) -> None:
    replies = [httpx.Response(429, headers={"retry-after": "19.5"}), _ok()]
    client = _client(lambda request: replies.pop(0))
    assert client.complete_json(system="s", user="u", schema=SCHEMA) == '{"ok": true}'
    assert 20.5 in _no_sleep


def test_client_error_is_not_retried() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="bad key")

    with pytest.raises(LLMError, match="401"):
        _client(handler).complete_json(system="s", user="u", schema=SCHEMA)
    assert len(calls) == 1


def test_gives_up_after_retries() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(LLMError, match="4 попыток"):
        _client(handler, retries=3).complete_json(system="s", user="u", schema=SCHEMA)
    assert len(calls) == 4


def test_requests_are_spaced_by_rpm_limit(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    now = [100.0]
    monkeypatch.setattr(openai_compat.time, "monotonic", lambda: now[0])
    client = _client(lambda request: _ok(), requests_per_minute=2)
    client.complete_json(system="s", user="u", schema=SCHEMA)
    now[0] += 5  # прошло 5 с из 30 допустимых между запросами
    client.complete_json(system="s", user="u", schema=SCHEMA)
    assert _no_sleep == [25.0]


def test_from_settings_requires_all_values() -> None:
    with pytest.raises(LLMError, match="LLM_BASE_URL"):
        OpenAICompatibleClient.from_settings(Settings(_env_file=None, llm_api_key="k"))
    client = OpenAICompatibleClient.from_settings(
        Settings(
            _env_file=None, llm_base_url="https://h/v1", llm_api_key="k", llm_model="m", llm_rpm=2
        )
    )
    assert client.model == "m" and client._min_interval == 30
