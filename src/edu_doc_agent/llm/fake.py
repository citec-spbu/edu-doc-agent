"""Подмена LLM для тестов и для проверки замера без настоящей модели."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field


class FakeChatModel(GenericFakeChatModel):
    """Чат-модель для тестов графа: отвечает сообщениями из `messages` по очереди.

    В отличие от `GenericFakeChatModel` принимает инструменты и структурированный вывод
    (ответ со structured output задаётся как `AIMessage` с `tool_calls`) и ведёт журнал
    запросов в `calls`.
    """

    calls: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@dataclass
class LLMCall:
    system: str
    user: str
    schema: dict[str, Any]


@dataclass
class FakeLLMClient:
    """Отвечает функцией `respond(call)` или по очереди из `replies`; ведёт журнал вызовов."""

    respond: Callable[[LLMCall], str] | None = None
    replies: list[str] = field(default_factory=list)
    calls: list[LLMCall] = field(default_factory=list)

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        call = LLMCall(system=system, user=user, schema=schema)
        self.calls.append(call)
        if self.respond is not None:
            return self.respond(call)
        if not self.replies:
            raise AssertionError("FakeLLMClient: ответы закончились")
        return self.replies.pop(0)
