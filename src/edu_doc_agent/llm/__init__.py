"""Единый шлюз к LLM: клиент, кэш, ретраи, подмена в тестах."""

from edu_doc_agent.llm.base import LLMClient
from edu_doc_agent.llm.embeddings import Embedder, OllamaEmbedder
from edu_doc_agent.llm.fake import FakeEmbedder, FakeLLMClient, LLMCall
from edu_doc_agent.llm.openai_compat import LLMError, OpenAICompatibleClient

__all__ = [
    "Embedder",
    "FakeEmbedder",
    "FakeLLMClient",
    "LLMCall",
    "LLMClient",
    "LLMError",
    "OllamaEmbedder",
    "OpenAICompatibleClient",
]
