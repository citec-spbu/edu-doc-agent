"""Цикл агента: повтор вызова инструмента и исправление итога после последнего шага."""

import asyncio

from graph_helpers import call
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from edu_doc_agent.llm import FakeChatModel
from edu_doc_agent.orchestration.agent import AskUser, answer_message, run_agent, run_agent_turn


class Answer(BaseModel):
    value: int


def test_repeated_tool_call_is_not_executed() -> None:
    executed: list[str] = []

    @tool
    async def lookup(code: str) -> str:
        """Поиск по коду."""
        executed.append(code)
        return f"данные {code}"

    model = FakeChatModel(
        messages=iter(
            [call("lookup", code="УК-1"), call("lookup", code="УК-1"), call("Answer", value=1)]
        )
    )

    asyncio.run(
        run_agent(model, system="", messages=[HumanMessage("?")], tools=[lookup], final=Answer)
    )

    assert executed == ["УК-1"]
    repeat = model.calls[2][-1]
    assert isinstance(repeat, ToolMessage) and "уже был" in repeat.content


def test_final_can_be_fixed_after_last_step() -> None:
    async def validate(name, result, history):
        return [] if result.value == 2 else ["нужно 2"]

    model = FakeChatModel(messages=iter([call("Answer", value=1), call("Answer", value=2)]))

    result = asyncio.run(
        run_agent(
            model,
            system="",
            messages=[HumanMessage("?")],
            tools=[],
            final=Answer,
            validate=validate,
            max_steps=1,
        )
    )

    assert result.value == 2  # отклонённый на последнем шаге итог исправлен


def test_ask_user_pauses_and_the_same_agent_continues() -> None:
    executed: list[str] = []

    @tool
    async def lookup(code: str) -> str:
        """Поиск по коду."""
        executed.append(code)
        return f"данные {code}"

    model = FakeChatModel(
        messages=iter(
            [
                call("lookup", code="УК-1"),
                call("ask_user", question="Какой код?", options=["УК-1", "УК-2"]),
                call("lookup", code="УК-1"),  # повтор после паузы не выполняется
                call("Answer", value=2),
            ]
        )
    )

    paused = asyncio.run(
        run_agent_turn(
            model,
            system="",
            messages=[HumanMessage("?")],
            tools=[lookup],
            finals={"Answer": Answer},
            ask=AskUser,
        )
    )

    assert paused.ask is not None and paused.ask.options == ["УК-1", "УК-2"]
    assert paused.result is None
    work = [*paused.messages, answer_message(paused.call_id, "УК-2", paused.ask.options)]

    done = asyncio.run(
        run_agent_turn(
            model,
            system="",
            messages=work,
            tools=[lookup],
            finals={"Answer": Answer},
            ask=AskUser,
        )
    )

    assert done.result == Answer(value=2)
    assert executed == ["УК-1"]
    # продолжение видит всю прежнюю работу и ответ результатом вызова ask_user
    resumed = model.calls[2]
    assert resumed[-1].content.endswith("выбран вариант): УК-2")
    assert sum(isinstance(m, ToolMessage) for m in resumed) == 2
    # у итогового вызова есть результат: работу можно продолжить
    assert isinstance(done.messages[-1], ToolMessage) and done.messages[-1].content == "Принято."


def test_question_schema_without_draft_and_single_accept_button() -> None:
    from edu_doc_agent.orchestration.agent import ACCEPT_DRAFT, Question

    assert "draft" not in Question.model_json_schema()["properties"]
    ask = AskUser(question="Подойдёт?", options=[ACCEPT_DRAFT, "Изменить"], draft="Цель — …")
    assert ask.choices() == [ACCEPT_DRAFT, "Изменить"]
