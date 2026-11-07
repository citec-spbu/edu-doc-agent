"""Клиент OpenAI-совместимого API (LiteLLM/vLLM) со структурированным выводом по JSON-схеме."""

import time
from typing import Any

import httpx

from edu_doc_agent.config import Settings, get_settings

_RETRY_STATUSES = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Сбой вызова модели после всех повторов."""


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 180.0,
        retries: int = 3,
        thinking: bool = False,
        requests_per_minute: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.retries = retries
        self.thinking = thinking
        self._min_interval = 60 / requests_per_minute if requests_per_minute else 0.0
        self._last_request = 0.0
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, **kwargs: Any
    ) -> "OpenAICompatibleClient":
        s = settings or get_settings()
        if not (s.llm_base_url and s.llm_api_key and s.llm_model):
            raise LLMError("задайте LLM_BASE_URL, LLM_API_KEY и LLM_MODEL (см. .env.example)")
        kwargs.setdefault("requests_per_minute", s.llm_rpm)
        return cls(base_url=s.llm_base_url, api_key=s.llm_api_key, model=s.llm_model, **kwargs)

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": schema},
            },
            # у Qwen3 рассуждение до ответа удлиняет вызов в разы; для извлечения оно не нужно
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        last = ""
        for attempt in range(self.retries + 1):
            self._wait_for_slot()
            delay = min(2**attempt, 10)
            try:
                resp = self._http.post("chat/completions", json=payload)
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"] or ""
                last = f"HTTP {resp.status_code}"
                if resp.status_code not in _RETRY_STATUSES:
                    raise LLMError(f"{last}: {resp.text[:300]}")
                if (retry_after := resp.headers.get("retry-after")) and retry_after.replace(
                    ".", "", 1
                ).isdigit():
                    delay = float(retry_after) + 1
            time.sleep(delay)
        raise LLMError(f"модель недоступна после {self.retries + 1} попыток ({last})")

    def _wait_for_slot(self) -> None:
        """Выдерживает интервал между запросами, чтобы не упереться в RPM-лимит ключа."""
        pause = self._last_request + self._min_interval - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        self._last_request = time.monotonic()
