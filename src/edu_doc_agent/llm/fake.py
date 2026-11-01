"""Подмена LLM для тестов и для проверки замера без настоящей модели."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


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
