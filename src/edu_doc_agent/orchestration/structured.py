"""Вызов LLM со структурированным ответом и повтором при ответе не по схеме (ADR-0003).

Если ответ не проходит схему Pydantic, тот же вызов повторяется с текстом ошибки,
до `MAX_SCHEMA_RETRIES` раз. Временные ошибки сети повторяет сама модель (`llm/`), сбои
провайдера, которые клиент не повторяет, — `call_provider`.

Модель получает JSON-схему словарём, а проверка по Pydantic идёт здесь: иначе клиент
OpenAI проверяет ответ сам и при ошибке бросает исключение до того, как можно повторить.
"""

import asyncio
import functools
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

MAX_SCHEMA_RETRIES = 4
PROVIDER_RETRIES = 4  # повторов при временном сбое провайдера
RETRY_DELAY = 1.0  # секунд перед первым повтором, дальше больше
BUSY_FACTOR = (
    5  # «модель занята» (429): ждать в 5 раз дольше — короткие повторы клиента не помогают
)


async def call_provider[T](call: Callable[[], Awaitable[T]], failed: Callable[[T], bool]) -> T:
    """Вызов LLM с повтором временных сбоев провайдера: ответ с finish_reason «error»,
    ошибка 422 «Provider returned error» (OpenRouter), которую клиент OpenAI не повторяет, и
    429 «модель занята» (DeepInfra), которую клиент повторяет слишком быстро."""
    for attempt in range(PROVIDER_RETRIES + 1):
        last = attempt == PROVIDER_RETRIES
        delay = RETRY_DELAY * (attempt + 1)
        try:
            result = await call()
        except (openai.UnprocessableEntityError, openai.RateLimitError) as error:
            if last:
                raise
            busy = isinstance(error, openai.RateLimitError)
            delay *= BUSY_FACTOR if busy else 1
            logger.warning(
                "сбой провайдера LLM (%s), повтор через %.0f с", error.status_code, delay
            )
        else:
            if last or not failed(result):
                return result
            logger.warning("сбой провайдера LLM в ответе, повтор")
        await asyncio.sleep(delay)
    raise AssertionError("недостижимо")


def decode_json_strings(value: Any) -> Any:
    """Объекты и списки, которые модель передала строкой JSON («"{\"courses\": []}"»)."""
    if isinstance(value, dict):
        return {k: decode_json_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_json_strings(v) for v in value]
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return decode_json_strings(json.loads(value))
        except json.JSONDecodeError:
            return value
    return value


class StructuredOutputError(RuntimeError):
    """Ответ LLM не прошёл схему после всех повторов."""


def user_error(error: BaseException) -> str:
    """Причина сбоя LLM для пользователя: без кодов и названий провайдеров. Исходная ошибка —
    в логе."""
    logger.warning("сбой LLM: %r", error)
    if isinstance(error, openai.APITimeoutError | TimeoutError):
        return "модель не ответила вовремя"
    if isinstance(error, openai.RateLimitError):
        return "превышен лимит запросов к модели"
    if isinstance(error, openai.APIError):
        return "модель временно недоступна"
    if isinstance(error, StructuredOutputError):
        return "модель отвечала не по формату"
    if type(error).__name__ == "AgentLimitError":  # agent.py импортирует этот модуль
        return "агент не дал ответ за отведённое число шагов"
    return "внутренняя ошибка"


async def invoke_structured[T: BaseModel](
    model: BaseChatModel, schema: type[T], messages: Sequence[BaseMessage]
) -> T:
    runnable = model.with_structured_output(
        schema.model_json_schema(), method="json_schema", include_raw=True
    )
    history = list(messages)
    error: BaseException | None = None
    for _ in range(MAX_SCHEMA_RETRIES + 1):
        result = await call_provider(
            functools.partial(runnable.ainvoke, history),
            lambda r: r["raw"].response_metadata.get("finish_reason") == "error",
        )
        error = result["parsing_error"]
        if result["parsed"] is not None:
            try:
                return schema.model_validate(decode_json_strings(result["parsed"]))
            except ValidationError as invalid:
                error = invalid
        history += [
            result["raw"],
            HumanMessage(f"Ответ не соответствует схеме: {error}. Ответь ещё раз по схеме."),
        ]
    raise StructuredOutputError(
        f"{schema.__name__}: ответ не по схеме после {MAX_SCHEMA_RETRIES + 1} попыток: {error}"
    )
