"""Эмбеддинги через Ollama (ADR-0005, решение 2): единый батчевый метод для RAG.

И загрузка нормативов, и поиск (запрос — батч из одной строки) ходят сюда; другие модули к
Ollama напрямую не обращаются. Модель по умолчанию — BGE-M3, вектор 1024.
"""

import time
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from edu_doc_agent.config import Settings, get_settings
from edu_doc_agent.llm.openai_compat import LLMError

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "bge-m3"
DEFAULT_DIM = 1024
NUM_CTX = 8192  # максимальный вход BGE-M3; без явного num_ctx Ollama берёт меньший контекст
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Векторы для строк в том же порядке; каждый длины `dim`."""
        ...


class OllamaEmbedder:
    """Клиент `POST /api/embed` Ollama: пачки, повторы, проверка размерности.

    `truncate: false` — слишком длинный вход даёт ошибку, а не молча теряет конец пункта.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        batch_size: int = 16,
        timeout: float = 300.0,
        retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size должен быть положительным")
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.retries = retries
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/", timeout=timeout, transport=transport
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None, **kwargs: Any) -> "OllamaEmbedder":
        s = settings or get_settings()
        return cls(
            base_url=s.embedding_base_url or DEFAULT_BASE_URL,
            model=s.embedding_model or DEFAULT_MODEL,
            **kwargs,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors += self._batch(list(texts[start : start + self.batch_size]))
        return vectors

    def _batch(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": texts,
            "truncate": False,
            "options": {"num_ctx": NUM_CTX},
        }
        data = self._post("api/embed", payload)
        vectors = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise LLMError("некорректный ответ Ollama: число векторов не совпадает с числом строк")
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self.dim:
                size = len(vector) if isinstance(vector, list) else "?"
                raise LLMError(
                    f"размерность эмбеддинга {size}, ожидалась {self.dim} (модель {self.model})"
                )
        return vectors

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        last = ""
        for attempt in range(self.retries + 1):
            try:
                resp = self._http.post(path, json=payload)
            except httpx.TransportError as exc:
                last = type(exc).__name__
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError:
                        raise LLMError("ответ Ollama — не JSON") from None
                last = f"HTTP {resp.status_code}: {_error_text(resp)}"
                if resp.status_code == 404:
                    last += f" (скачайте модель: ollama pull {self.model})"
                if resp.status_code not in _RETRY_STATUSES:
                    # 400 — в том числе вход длиннее контекста при truncate: false; 404 — модель
                    # не скачана (`ollama pull bge-m3`). Повтор не поможет.
                    raise LLMError(f"Ollama: {last}")
            time.sleep(min(2**attempt, 10))
        raise LLMError(f"Ollama недоступна после {self.retries + 1} попыток ({last})")


def _error_text(resp: httpx.Response) -> str:
    try:
        error = resp.json().get("error")
    except (ValueError, AttributeError):
        error = None
    return str(error) if error else resp.text[:200]
