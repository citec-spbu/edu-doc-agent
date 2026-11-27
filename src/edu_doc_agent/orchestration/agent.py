"""Цикл агента с инструментами: модель вызывает инструменты, пока не вызовет итоговый.

Итог агента — вызов инструмента-схемы (Pydantic-модель). Агент, которому можно спрашивать
пользователя, получает инструмент `ask_user`: на нём работа приостанавливается
(`AgentTurn.ask`), и ответ возвращается тому же агенту результатом вызова. Итог не по схеме
или не принятый кодом возвращается модели с ошибками, до `MAX_SCHEMA_RETRIES` раз; после
`max_steps` вызовов модели — `AgentLimitError` (ADR-0003, «Повторы и восстановление»).
"""

import functools
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from edu_doc_agent.orchestration.structured import (
    MAX_SCHEMA_RETRIES,
    call_provider,
    decode_json_strings,
)

logger = logging.getLogger(__name__)

MAX_STEPS = 10
MAX_TOOL_RESULT = 12_000  # символов результата инструмента в контексте модели


class AgentLimitError(RuntimeError):
    """Агент не дал итог за отведённое число шагов или ответов не по схеме."""


def to_json(value: Any) -> str:
    def plain(v: Any) -> Any:
        if isinstance(v, BaseModel):
            return v.model_dump(mode="json", exclude_none=True)
        if isinstance(v, list | tuple):
            return [plain(i) for i in v]
        if isinstance(v, dict):
            return {k: plain(i) for k, i in v.items()}
        return v

    text = json.dumps(plain(value), ensure_ascii=False)
    if len(text) > MAX_TOOL_RESULT:
        text = text[:MAX_TOOL_RESULT] + " …(результат обрезан, уточните запрос)"
    return text


async def _invoke(bound: Any, history: list[BaseMessage]) -> BaseMessage:
    """Вызов модели; временный сбой провайдера повторяется (`call_provider`)."""
    return await call_provider(
        functools.partial(bound.ainvoke, history),
        lambda ai: ai.response_metadata.get("finish_reason") == "error",
    )


# итоговый инструмент: схема и проверка кодом; непустой список ошибок возвращается модели.
# `history` — работа агента до этого шага (без системного промпта): по ней код проверяет,
# например, что текст правки взят из ответа пользователя
Validate = Callable[[str, BaseModel, list[BaseMessage]], Awaitable[list[str]]]

ASK_USER = "ask_user"
ACCEPT_DRAFT = "Принять текст"  # кнопка к тексту `AskUser.draft`, её добавляет код


class Question(BaseModel):
    """Вопрос пользователю, без ответа на который работу не продолжить. Работа
    приостанавливается до ответа; ответ придёт результатом этого вызова (с его answer_id), и
    работа продолжится."""

    question: str = Field(description="вопрос пользователю по-русски, коротко")
    options: list[str] = Field(
        default_factory=list,
        description="варианты ответа — кнопки, пользователь может и написать свой ответ",
    )
    comment: str | None = Field(
        None,
        description="ответ на последнюю реплику пользователя, если она не ответ на вопрос "
        "(он спросил о чём-то сам) или ответ не подошёл; показывается перед вопросом",
    )

    def text(self) -> str:
        return "\n\n".join(p for p in (self.comment, self.question) if p)

    def choices(self) -> list[str]:
        return self.options


class AskUser(Question):
    """Вопрос пользователю, без ответа на который работу не продолжить; можно предложить
    текст для документа. Работа приостанавливается до ответа; ответ придёт результатом этого
    вызова (с его answer_id), и работа продолжится."""

    draft: str | None = Field(
        None,
        description="текст для документа, который ты предлагаешь (цели, миссия, аннотация, "
        "результат обучения), целиком; пользователь примет его кнопкой «Принять текст» или "
        "напишет свой",
    )

    def text(self) -> str:
        return "\n\n".join(
            p for p in (super().text(), f"«{self.draft}»" if self.draft else None) if p
        )

    def choices(self) -> list[str]:
        if not self.draft:
            return self.options
        return [ACCEPT_DRAFT, *(o for o in self.options if o != ACCEPT_DRAFT)]


@dataclass
class AgentTurn:
    """Итог шага работы агента: итоговый вызов или вопрос пользователю.

    `messages` — вся работа агента без системного промпта; с ней (и ответом пользователя)
    агент продолжает работу после паузы. Итоговый вызов в ней уже с результатом."""

    messages: list[BaseMessage]
    name: str | None = None
    result: BaseModel | None = None
    ask: Question | None = None
    call_id: str | None = None  # вызов ask_user, ответ на который ждёт агент


def answer_message(
    call_id: str, answer: str, options: Sequence[str], draft: str | None = None
) -> ToolMessage:
    """Ответ пользователя — результат вызова ask_user. Нажатую кнопку отмечает код
    (`artifact`); принятый кнопкой текст `draft` — это и есть ответ."""
    if draft and answer == ACCEPT_DRAFT:
        how, answer = "принят предложенный текст", draft
    else:
        how = "выбран вариант" if answer in options else "свой текст"
    return ToolMessage(
        f"Ответ пользователя (answer_id: {call_id}, {how}): {answer}",
        tool_call_id=call_id,
        name=ASK_USER,
        artifact={"answer": answer, "option": how != "свой текст"},
    )


def user_answer(history: Sequence[BaseMessage], call_id: str | None) -> dict | None:
    """Ответ пользователя на вопрос ask_user `call_id`: {"answer", "option"} или None."""
    for m in history:
        if isinstance(m, ToolMessage) and m.tool_call_id == call_id and m.name == ASK_USER:
            return m.artifact if isinstance(m.artifact, dict) else None
    return None


def _final_spec(name: str, schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (schema.__doc__ or name).strip(),
            "parameters": schema.model_json_schema(),
        },
    }


def _call_key(call: dict[str, Any]) -> str:
    return f"{call['name']}{json.dumps(call['args'], sort_keys=True)}"


def _seen_calls(history: Sequence[BaseMessage], skip: set[str]) -> set[str]:
    """Вызовы инструментов, уже сделанные в прежней работе агента."""
    return {
        _call_key(c)
        for m in history
        if isinstance(m, AIMessage)
        for c in m.tool_calls
        if c["name"] not in skip
    }


async def _tool_reply(call: dict[str, Any], tools: dict[str, BaseTool], seen: set[str]) -> str:
    """Результат инструмента; повтор вызова с теми же аргументами не выполняется."""
    tool = tools.get(call["name"])
    if tool is None:
        return f"Нет инструмента {call['name']}. Доступны: {', '.join(tools)} и итоговые."
    key = _call_key(call)
    if key in seen:
        return (
            "Этот вызов с теми же аргументами уже был, его результат — выше. Не повторяй "
            "его: используй результат, вызови другой инструмент или дай итог."
        )
    seen.add(key)
    try:
        return to_json(await tool.ainvoke(call["args"]))
    except Exception as error:  # ошибку инструмента видит модель и исправляет вызов
        logger.info("инструмент %s: %s", call["name"], error)
        return f"Ошибка инструмента: {error}"


def _log_step(finals: dict[str, type[BaseModel]], step: int, started: float, ai: Any) -> None:
    calls = [
        f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)[:120]})"
        for c in getattr(ai, "tool_calls", [])
    ]
    logger.info(
        "агент %s, шаг %d (%.0f с): %s", "/".join(finals), step, time.monotonic() - started, calls
    )


async def run_agent_turn(
    model: BaseChatModel,
    *,
    system: str,
    messages: Sequence[BaseMessage],
    tools: Sequence[BaseTool],
    finals: dict[str, type[BaseModel]],
    validate: Validate | None = None,
    max_steps: int = MAX_STEPS,
    ask: type[Question] | None = None,
) -> AgentTurn:
    """Работа агента до итога или (`ask`) до вопроса пользователю.

    `messages` — начало работы или прежняя работа агента с ответом пользователя в конце:
    агент продолжает с полным контекстом. Ответ не по схеме или не прошедший `validate`
    возвращается модели с ошибками, всего не более `MAX_SCHEMA_RETRIES` раз."""
    by_name = {t.name: t for t in tools}
    specs = [_final_spec(n, s) for n, s in finals.items()]
    if ask is not None:
        specs.append(_final_spec(ASK_USER, ask))
    bound = model.bind_tools([*tools, *specs], tool_choice="required")
    # последний шаг — только итоговые инструменты: агент отвечает по тому, что уже узнал
    last = model.bind_tools(specs, tool_choice="required")
    history: list[BaseMessage] = list(messages)
    rejected = 0
    # вызовы инструментов: повтор с теми же аргументами не выполняется
    seen = _seen_calls(history, {*finals, ASK_USER})
    ending = " или ".join([*finals, *([ASK_USER] if ask is not None else [])])
    # после последнего шага — только итог; отклонённый итог исправляется ещё до
    # MAX_SCHEMA_RETRIES раз, а не обрывает работу на последнем шаге
    for step in range(max_steps + MAX_SCHEMA_RETRIES):
        started = time.monotonic()
        final_only = step >= max_steps - 1
        if step == max_steps - 1:
            history.append(HumanMessage(f"Шаги закончились: дай итог ({ending})."))
        ai = await _invoke(last if final_only else bound, [SystemMessage(system), *history])
        _log_step(finals, step + 1, started, ai)
        if not isinstance(ai, AIMessage) or not ai.tool_calls:
            logger.info(
                "агент %s: ответ без вызова инструмента (%s): %.300s",
                "/".join(finals),
                ai.response_metadata.get("finish_reason"),
                ai.content,
            )
            if ai.content:  # пустой ответ модели провайдер не принимает в истории
                history.append(ai)
            history.append(HumanMessage(f"Ответь вызовом инструмента; итог — {ending}."))
            continue
        history.append(ai)
        replies: list[ToolMessage] = []
        asked = (
            next((c for c in ai.tool_calls if c["name"] == ASK_USER), None)
            if ask is not None
            else None
        )
        done: tuple[str, BaseModel, str] | None = None
        for call in ai.tool_calls:
            name = call["name"]
            if call is asked:
                continue  # ответ на вопрос придёт после паузы
            if name == ASK_USER and ask is not None:
                content = "Задавай по одному вопросу: этот не задан, дождись ответа на первый."
            elif asked is not None and name in finals:
                content = "Итог не принят: сначала дождись ответа пользователя на вопрос."
            elif name in finals:
                try:
                    result = finals[name].model_validate(decode_json_strings(call["args"]))
                    problems = await validate(name, result, history) if validate else []
                except ValidationError as error:
                    problems = [f"ответ не соответствует схеме: {error}"]
                if not problems and done is None:
                    done = (name, result, call["id"])
                    continue
                if problems:
                    rejected += 1
                    if rejected > MAX_SCHEMA_RETRIES:
                        raise AgentLimitError(f"{name}: не принят после исправлений: {problems}")
                    content = (
                        "Не принято:\n- " + "\n- ".join(problems) + "\nИсправь и вызови ещё раз."
                    )
                else:
                    content = "Итог уже принят, этот вызов лишний."
            else:
                content = await _tool_reply(call, by_name, seen)
            replies.append(ToolMessage(content, tool_call_id=call["id"], name=name))
        if asked is not None:
            try:
                question = ask.model_validate(decode_json_strings(asked["args"]))
                problems = await validate(ASK_USER, question, history) if validate else []
            except ValidationError as error:
                problems = [str(error)]
            if problems:  # вопрос не задан: модель исправляет его по замечаниям кода
                replies.append(
                    ToolMessage(
                        "Вопрос не задан: " + "; ".join(problems),
                        tool_call_id=asked["id"],
                        name=ASK_USER,
                    )
                )
                history += replies
                continue
            history += replies
            return AgentTurn(history, ask=question, call_id=asked["id"])
        if done is not None:
            name, result, call_id = done
            # у итогового вызова — результат: работу можно продолжить (ответ пользователя)
            replies.append(ToolMessage("Принято.", tool_call_id=call_id, name=name))
            history += replies
            return AgentTurn(history, name=name, result=result)
        history += replies
    raise AgentLimitError(f"{' / '.join(finals)}: нет итога за {max_steps} шагов")


async def run_agent[T: BaseModel](
    model: BaseChatModel,
    *,
    system: str,
    messages: Sequence[BaseMessage],
    tools: Sequence[BaseTool],
    final: type[T],
    validate: Validate | None = None,
    max_steps: int = MAX_STEPS,
) -> T:
    """Итог агента по схеме `final` без вопросов пользователю."""
    turn = await run_agent_turn(
        model,
        system=system,
        messages=messages,
        tools=tools,
        finals={final.__name__: final},
        validate=validate,
        max_steps=max_steps,
    )
    assert isinstance(turn.result, final)
    return turn.result
