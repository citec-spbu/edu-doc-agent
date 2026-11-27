"""Повтор структурированного ответа через настоящий ChatOpenAI (HTTP подменён)."""

import asyncio
import json

import httpx
import openai
import pytest
from langchain_core.messages import HumanMessage

from edu_doc_agent.config import Settings
from edu_doc_agent.llm import get_chat_model
from edu_doc_agent.orchestration import structured
from edu_doc_agent.orchestration.agent import AgentLimitError
from edu_doc_agent.orchestration.orchestrator import OrchestratorDecision
from edu_doc_agent.orchestration.structured import (
    MAX_SCHEMA_RETRIES,
    StructuredOutputError,
    invoke_structured,
    user_error,
)

PROVIDER_ERROR = "422"  # ответ провайдера «Provider returned error» вместо сообщения
BUSY = "429"  # «Model busy, retry later»


def _model(contents: list[str], requests: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = contents.pop(0)
        if content == PROVIDER_ERROR:
            return httpx.Response(422, json={"error": {"message": "Provider returned error"}})
        if content == BUSY:
            return httpx.Response(429, json={"error": {"message": "Model busy, retry later"}})
        message = {"role": "assistant", "content": content}
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            },
        )

    settings = Settings(
        _env_file=None, llm_base_url="https://llm.example/v1", llm_api_key="k", llm_model="qwen3"
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return get_chat_model("orchestrator", settings, http_async_client=client)


def _decide(contents: list[str], requests: list[dict]) -> OrchestratorDecision:
    model = _model(contents, requests)
    return asyncio.run(invoke_structured(model, OrchestratorDecision, [HumanMessage("проверь")]))


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(json.dumps({"scope": {}}), id="validator"),
        pytest.param("не JSON", id="not-json"),
    ],
)
def test_retries_until_valid(bad: str) -> None:
    requests: list[dict] = []

    result = _decide([bad, json.dumps({"kind": "check"})], requests)

    assert result.kind == "check"
    assert len(requests) == 2
    assert requests[0]["response_format"]["json_schema"]["name"] == "OrchestratorDecision"
    assert "не соответствует схеме" in requests[1]["messages"][-1]["content"]


def test_gives_up() -> None:
    bad = json.dumps({"kind": "unknown"})

    with pytest.raises(StructuredOutputError):
        _decide([bad] * (MAX_SCHEMA_RETRIES + 1), [])


def test_provider_error_422_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(structured, "RETRY_DELAY", 0)
    requests: list[dict] = []

    result = _decide([PROVIDER_ERROR, json.dumps({"kind": "check"})], requests)

    assert result.kind == "check"
    assert len(requests) == 2  # клиент OpenAI сам 422 не повторяет


def test_user_error_hides_provider_details() -> None:
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    upstream = openai.InternalServerError(
        "Upstream error from Darkbloom", response=httpx.Response(502, request=request), body=None
    )

    assert user_error(upstream) == "модель временно недоступна"
    assert user_error(openai.APITimeoutError(request)) == "модель не ответила вовремя"
    assert user_error(AgentLimitError("10 шагов")) == "агент не дал ответ за отведённое число шагов"


def test_model_busy_429_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(structured, "RETRY_DELAY", 0)
    requests: list[dict] = []
    model = _model([BUSY, json.dumps({"kind": "check"})], requests)
    model.max_retries = 0  # повторы клиента выключены: повторяет call_provider

    result = asyncio.run(invoke_structured(model, OrchestratorDecision, [HumanMessage("x")]))

    assert result.kind == "check"
    assert len(requests) == 2
