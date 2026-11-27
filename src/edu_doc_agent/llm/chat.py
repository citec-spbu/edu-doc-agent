"""Чат-модель LangChain с tool calling для агентов графа (ADR-0003, «Интерфейс с воркером»).

Ходит в тот же OpenAI-совместимый эндпоинт, что и `OpenAICompatibleClient`, с теми же
настройками: модель, ключ, лимит запросов в минуту, рассуждение Qwen3 выключено.
"""

from functools import cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from edu_doc_agent.config import Settings, get_settings
from edu_doc_agent.llm.openai_compat import LLMError

MAX_RETRIES = 4  # временные ошибки (таймаут, 429, 5xx): до 4 повторов, ADR-0003
TIMEOUT = 180.0


@cache
def _rate_limiter(requests_per_minute: int) -> InMemoryRateLimiter:
    """Один лимитер на процесс: лимит относится к ключу, а не к отдельному агенту."""
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60, check_every_n_seconds=0.1
    )


def _extra_body(s: Settings) -> dict[str, Any]:
    """Выключение рассуждения. OpenRouter не передаёт шаблону `chat_template_kwargs`; у него же
    выбираем только провайдеров, которые поддерживают все параметры запроса (инструменты)."""
    if s.llm_provider == "openrouter":
        return {
            "reasoning": {"enabled": False},
            "provider": {"require_parameters": True},
        }
    return {"chat_template_kwargs": {"enable_thinking": False}}


def get_chat_model(role: str, settings: Settings | None = None, **kwargs: Any) -> BaseChatModel:
    """Чат-модель для агента `role` (`orchestrator`, `checker`, ...).

    Роль пока не меняет настройки, она попадает в теги вызовов, чтобы в трассировке было
    видно, какой агент обращался к модели. `kwargs` перекрывают параметры `ChatOpenAI`
    (в тестах так подменяется HTTP-клиент).
    """
    s = settings or get_settings()
    if not (s.llm_base_url and s.llm_api_key and s.llm_model):
        raise LLMError("задайте LLM_BASE_URL, LLM_API_KEY и LLM_MODEL (см. .env.example)")
    params: dict[str, Any] = {
        "model": s.llm_model,
        "base_url": s.llm_base_url,
        "api_key": s.llm_api_key,
        "temperature": 0,
        "timeout": TIMEOUT,
        "max_retries": MAX_RETRIES,
        "rate_limiter": _rate_limiter(s.llm_rpm) if s.llm_rpm else None,
        # у Qwen3 рассуждение до ответа удлиняет вызов в разы
        "extra_body": _extra_body(s),
        "tags": [f"role:{role}"],
    }
    return ChatOpenAI(**(params | kwargs))
