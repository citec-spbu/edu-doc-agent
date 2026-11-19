"""Подмена LLM для тестов и для проверки замера без настоящей модели."""

import hashlib
import math
import re
from collections.abc import Callable, Sequence
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


@dataclass
class FakeEmbedder:
    """Детерминированные эмбеддинги без модели: мешок слов, свёрнутый хешем в `dim` измерений.

    Тексты с общими словами получают близкие векторы, поэтому поиск в тестах ведёт себя
    осмысленно. Журнал `calls` — список пачек, с которыми вызывали `embed`.
    """

    dim: int = 1024
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        vector[0] = 1e-3  # пустой текст не даёт нулевой вектор: косинус с ним не определён
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]
