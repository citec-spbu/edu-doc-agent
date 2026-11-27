"""Диалог сессии с пользователем: сообщения в состоянии графа и краткое содержание ранних.

В диалоге только реплики пользователя и системы: запрос, вопросы с ответами, итоговый ответ;
работа агентов (вызовы инструментов) в него не попадает. У реплики системы — имя агента
(`name`): без него агент принимает чужой ответ пользователю за свой. Сверх `HISTORY_CHARS`
ранние сообщения сворачиваются в краткое содержание и удаляются (`RemoveMessage`, шаблон
LangGraph «summarize conversation history»).
"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from pydantic import BaseModel

from edu_doc_agent.orchestration.structured import invoke_structured

HISTORY_CHARS = 8000  # символов диалога дословно, дальше — краткое содержание
KEEP_MESSAGES = 4  # последних сообщений всегда дословно
MAX_SHOWN_CHARS = 4000  # длинное сообщение (протокол проверки) агенту показывается началом

SUMMARY_PROMPT = """\
Ты ведёшь краткое содержание диалога пользователя с системой проверки документов \
образовательной программы СПбГУ. Объедини прежнее краткое содержание и новые реплики в одно \
краткое содержание: о чём спрашивал пользователь, какие программы, документы и дисциплины \
обсуждались, что система ответила, проверила или изменила, что осталось нерешённым. Не больше \
10 предложений, по-русски, только факты из реплик."""

_ROLE = {"human": "Пользователь", "ai": "Система"}


class Summary(BaseModel):
    summary: str


def question_turn(
    question: str, options: Sequence[str], answer: str, author: str
) -> list[BaseMessage]:
    """Вопрос агента `author` и ответ пользователя — реплики диалога."""
    text = question + (f"\nВарианты: {'; '.join(options)}" if options else "")
    return [AIMessage(text, name=author), HumanMessage(answer)]


def dialogue(messages: Sequence[BaseMessage], summary: str | None) -> list[BaseMessage]:
    """Диалог для агента: краткое содержание ранних реплик и последние сообщения."""
    shown: list[BaseMessage] = []
    if summary:
        shown.append(HumanMessage(f"Краткое содержание более раннего диалога: {summary}"))
    for m in messages:
        text = m.content if isinstance(m.content, str) else str(m.content)
        if len(text) > MAX_SHOWN_CHARS:
            text = text[:MAX_SHOWN_CHARS] + " …(сокращено)"
        shown.append(type(m)(text, name=m.name))
    return shown


async def remember(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    summary: str | None,
    reply: str | None,
    author: str,
) -> dict:
    """Обновление состояния: ответ запуска (агента `author`) в диалог; лишнее сверх
    `HISTORY_CHARS` — в краткое содержание."""
    new = [AIMessage(reply, name=author)] if reply else []
    everything = [*messages, *new]
    size = sum(len(str(m.content)) for m in everything)
    if size <= HISTORY_CHARS or len(everything) <= KEEP_MESSAGES:
        return {"messages": new}
    old = everything[:-KEEP_MESSAGES]
    lines = [f"Прежнее краткое содержание: {summary}"] if summary else []
    lines += [f"{_ROLE.get(m.type, m.type)}: {m.content}" for m in old]
    try:
        result = await invoke_structured(
            model, Summary, [SystemMessage(SUMMARY_PROMPT), HumanMessage("\n".join(lines))]
        )
    except Exception:  # LLM недоступна: сообщения не теряются, свернуть — в следующий раз
        return {"messages": new}
    removed = [RemoveMessage(id=m.id) for m in old if m.id]
    return {"messages": [*new, *removed], "summary": result.summary}
