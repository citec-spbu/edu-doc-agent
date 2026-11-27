"""Чат-модель для агентов: настройки, запрос с инструментами. Сеть не используется."""

import asyncio
import json

import httpx
import pytest
from langchain_core.tools import tool

from edu_doc_agent.config import Settings
from edu_doc_agent.llm import LLMError, get_chat_model


def _settings(**overrides) -> Settings:
    values = {
        "llm_base_url": "https://llm.example/v1",
        "llm_api_key": "sk-test",
        "llm_model": "qwen3",
        "llm_rpm": None,
    }
    return Settings(_env_file=None, **(values | overrides))


@tool
def get_findings(rule_id: str) -> str:
    """Замечания по правилу."""
    return ""


def test_requires_llm_settings() -> None:
    with pytest.raises(LLMError, match="LLM_BASE_URL"):
        get_chat_model("orchestrator", _settings(llm_api_key=None))


def test_uses_llm_settings() -> None:
    model = get_chat_model("checker", _settings(llm_rpm=30))

    assert model.model_name == "qwen3"
    assert model.openai_api_base == "https://llm.example/v1"
    assert model.temperature == 0
    assert model.max_retries == 4
    assert model.rate_limiter is not None
    assert model.tags == ["role:checker"]


def test_rate_limiter_is_shared_between_agents() -> None:
    settings = _settings(llm_rpm=30)

    a = get_chat_model("checker", settings)
    b = get_chat_model("reviewer", settings)

    assert a.rate_limiter is b.rate_limiter


def test_tool_call_request() -> None:
    """Запрос уходит в chat/completions с инструментами и без рассуждения Qwen3."""
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append({"url": str(request.url), "body": json.loads(request.content)})
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_findings", "arguments": '{"rule_id": "r1"}'},
        }
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call],
                        },
                    }
                ],
            },
        )

    model = get_chat_model(
        "orchestrator",
        _settings(),
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    reply = asyncio.run(model.bind_tools([get_findings]).ainvoke("какие замечания по r1?"))

    assert reply.tool_calls[0]["name"] == "get_findings"
    assert reply.tool_calls[0]["args"] == {"rule_id": "r1"}
    (request,) = sent
    assert request["url"] == "https://llm.example/v1/chat/completions"
    assert request["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["body"]["tools"][0]["function"]["name"] == "get_findings"


def test_openrouter_disables_reasoning_its_way() -> None:
    """OpenRouter не передаёт шаблону chat_template_kwargs: рассуждение выключается его полем."""
    settings = _settings().model_copy(update={"llm_provider": "openrouter"})

    model = get_chat_model("checker", settings)

    assert model.extra_body == {
        "reasoning": {"enabled": False},
        "provider": {"require_parameters": True},
    }
