"""Единый шлюз к LLM: клиент, кэш, ретраи, подмена в тестах."""

from edu_doc_agent.llm.base import LLMClient
from edu_doc_agent.llm.chat import get_chat_model
from edu_doc_agent.llm.fake import FakeChatModel, FakeLLMClient, LLMCall
from edu_doc_agent.llm.openai_compat import LLMError, OpenAICompatibleClient

__all__ = [
    "FakeChatModel",
    "FakeLLMClient",
    "LLMCall",
    "LLMClient",
    "LLMError",
    "OpenAICompatibleClient",
    "get_chat_model",
]
