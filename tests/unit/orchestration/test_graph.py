"""Граф оркестрации: маршрутизация, оркестратор-агент, пауза и возобновление, код
направления, сброс запуска, повтор после сбоя.

LLM подменяется `FakeChatModel` по ролям агентов, состояние хранит `InMemorySaver`.
"""

import asyncio

import pytest
from graph_helpers import Ingestor, Models, ask, call, config, decision, make_graph, rpd, run
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from edu_doc_agent.orchestration import history as history_module
from edu_doc_agent.orchestration import make_input
from edu_doc_agent.orchestration.agent import AgentLimitError
from edu_doc_agent.orchestration.orchestrator import unknown_courses
from edu_doc_agent.orchestration.structured import MAX_SCHEMA_RETRIES

RPD_FILES = {"a.md": ("rpd", rpd(title="Физика"))}


def test_question_goes_to_qa_agent() -> None:
    models = Models(
        orchestrator=[decision(kind="question")],
        qa=[call("QAAnswer", answer="Ответ Q&A-агента")],
    )
    graph, _ = make_graph(models)

    state, nodes = run(graph, make_input("r1", "что такое УК-1?"))

    assert nodes == ["start", "orchestrator", "qa", "remember"]
    assert state.values["reply"] == "Ответ Q&A-агента"
    assert state.next == ()


def test_check_route_and_reporter() -> None:
    models = Models(
        orchestrator=[decision(kind="check")],
        reporter=[call("ReporterAnswer", message="Коротко о проверке.")],
    )
    graph, _ = make_graph(models, RPD_FILES)

    state, nodes = run(graph, make_input("r1", "проверь", file_paths=["a.md"]))

    assert nodes == ["start", "ingest", "orchestrator", "check", "reporter", "remember"]
    assert state.values["direction_code"] == "02.03.02"
    reply = state.values["reply"]
    assert reply.startswith("Коротко о проверке.\n\nПроверка завершена.")
    # репортер видит только протокол: на вопросы пользователя отвечает Q&A-агент
    (reporter_input,) = models.calls("reporter")[0][1:]
    assert reporter_input.content.startswith("Проверка завершена.")
    assert "проверь" not in reporter_input.content


def test_check_and_fix_without_fixable_findings() -> None:
    graph, _ = make_graph(Models(orchestrator=[decision(kind="check_and_fix")]), RPD_FILES)

    state, nodes = run(graph, make_input("r1", "проверь и исправь", file_paths=["a.md"]))

    assert nodes[-4:] == ["check", "generate", "reporter", "remember"]
    assert "Изменений нет" in state.values["reply"]


def test_action_routes_without_llm() -> None:
    models = Models()
    graph, _ = make_graph(models, RPD_FILES)

    state, nodes = run(graph, make_input("r1", "проверь", action="check", file_paths=["a.md"]))

    assert nodes == ["start", "ingest", "orchestrator", "check", "reporter", "remember"]
    assert state.values["kind"] == "check"
    assert models.calls("orchestrator") == []


def test_orchestrator_reads_findings_with_tools() -> None:
    """Оркестратор находит ID замечаний инструментом и передаёт их в Scope."""
    fid = "cross-check-rpd-credits-vs-workload-table:doc:a.md"
    models = Models(
        orchestrator=[
            call("get_findings", course_id="doc:a.md"),
            decision(kind="change", scope={"finding_ids": [fid]}),
        ],
        generator=[ask("Сколько з.е. оставить?")],
    )
    graph, _ = make_graph(models, {"a.md": ("rpd", rpd(title="Физика", credits=6))})
    run(graph, make_input("r1", "проверь", action="check", file_paths=["a.md"]))

    state, nodes = run(graph, make_input("r2", "исправь трудоёмкость физики"))

    assert state.values["scope"]["finding_ids"] == [fid]
    tool_reply = models.calls("orchestrator")[1][-1]
    assert isinstance(tool_reply, ToolMessage)
    assert fid in tool_reply.content
    # генератор спросил пользователя: граф на паузе
    assert nodes[-2:] == ["orchestrator", "generate"]
    assert state.next == ("ask_user",)
    assert state.interrupts[0].value["type"] == "missing_data"


def test_scope_from_decision() -> None:
    scope = {"courses": ["Физика"], "rule_sets": ["cross_check"]}
    graph, _ = make_graph(Models(orchestrator=[decision(kind="check", scope=scope)]), RPD_FILES)

    state, _ = run(graph, make_input("r1", "сверь физику", file_paths=["a.md"]))

    assert state.values["scope"] == {
        "courses": ["Физика"],
        "rule_sets": ["cross_check"],
        "finding_ids": [],
        "change_goal": None,
    }
    assert "ФГОС" not in state.values["reply"]


def test_clarification_pauses_and_resumes() -> None:
    unclear = decision(
        kind="unclear", question="Какие дисциплины проверить?", options=["Все", "Только физику"]
    )
    models = Models(orchestrator=[unclear, decision(kind="check")])
    graph, _ = make_graph(models, RPD_FILES)

    paused, nodes = run(graph, make_input("r1", "проверь", file_paths=["a.md"]))

    assert nodes == ["start", "ingest", "orchestrator"]
    assert paused.next == ("ask_user",)
    assert paused.interrupts[0].value == {
        "type": "clarification",
        "question": "Какие дисциплины проверить?",
        "options": ["Все", "Только физику"],
    }

    done, nodes = run(graph, Command(resume="все, кроме практики"))

    assert nodes == ["ask_user", "orchestrator", "check", "reporter", "remember"]
    # оркестратор решает заново по диалогу: в нём его вопрос и ответ пользователя
    second = models.calls("orchestrator")[1]
    assert [(type(m).__name__, m.content) for m in second[-2:]] == [
        ("AIMessage", "Какие дисциплины проверить?\nВарианты: Все; Только физику"),
        ("HumanMessage", "все, кроме практики"),
    ]
    dialog = [(type(m).__name__, m.content) for m in done.values["messages"]]
    assert dialog[0] == ("HumanMessage", "проверь")
    assert "Проверка завершена" in dialog[3][1]


def test_button_answer_is_marked_by_code() -> None:
    models = Models(
        orchestrator=[decision(kind="change", scope={"change_goal": "трудоёмкость"})],
        generator=[ask("Сколько з.е.?", ["6", "8"]), ask("Точно?", ["Да"])],
    )
    graph, _ = make_graph(models, RPD_FILES)
    run(graph, make_input("r1", "измени трудоёмкость", file_paths=["a.md"]))

    run(graph, Command(resume="6"))

    # ответ — результат вызова ask_user того же агента; кнопку отмечает код
    reply = models.calls("generator")[1][-1]
    assert isinstance(reply, ToolMessage) and reply.name == "ask_user"
    assert reply.artifact == {"answer": "6", "option": True}
    assert "выбран вариант): 6" in reply.content
    assert len(models.calls("generator")) == 2  # первый вызов не повторяется


def test_question_comment_is_shown_before_question() -> None:
    models = Models(
        orchestrator=[decision(kind="change", scope={"change_goal": "трудоёмкость"})],
        generator=[
            ask("Сколько з.е.?", ["6", "8"]),
            ask("Сколько з.е.?", ["6", "8"], comment="В таблице итог 6, в тексте 8."),
        ],
    )
    graph, _ = make_graph(models, RPD_FILES)
    run(graph, make_input("r1", "измени трудоёмкость", file_paths=["a.md"]))

    again, _ = run(graph, Command(resume="а почему спрашиваете?"))

    assert again.interrupts[0].value["question"] == "В таблице итог 6, в тексте 8.\n\nСколько з.е.?"


@pytest.mark.parametrize(
    ("answer", "choice", "code", "fgos_checked"),
    [
        ("02.03.02", {"direction_code": "02.03.02"}, "02.03.02", True),
        ("не надо ФГОС", {"skip_fgos": True}, "", False),
    ],
)
def test_asks_direction_code_when_missing(
    answer: str, choice: dict, code: str, fgos_checked: bool
) -> None:
    models = Models(
        orchestrator=[
            decision(kind="unclear", question="Укажите код направления.", options=["02.03.02"]),
            decision(kind="check", **choice),
        ]
    )
    graph, _ = make_graph(models, {"a.md": ("rpd", rpd(direction_code=None))})

    paused, nodes = run(graph, make_input("r1", "проверь", file_paths=["a.md"]))

    assert nodes[-1] == "orchestrator"
    assert paused.interrupts[0].value["type"] == "clarification"
    note = models.calls("orchestrator")[0][0].content  # системный промпт
    assert "Код направления в документах не найден; правила ФГОС есть для: 02.03.02." in note

    done, nodes = run(graph, Command(resume=answer))

    assert nodes == ["ask_user", "orchestrator", "check", "reporter", "remember"]
    assert models.calls("orchestrator")[1][-1].content == answer  # ответ — реплика диалога
    assert done.values["direction_code"] == code
    reply = done.values["reply"]
    assert ("Правила ФГОС не применялись" in reply) is not fgos_checked


def test_check_without_code_is_refused_until_asked() -> None:
    models = Models(
        orchestrator=[
            decision(kind="check"),  # кода нет ни в документах, ни в итоге
            decision(
                kind="unclear",
                question="Укажите код направления.",
                options=["02.03.02"],  # коды кнопками не показываются
                about="direction_code",
            ),
        ]
    )
    graph, _ = make_graph(models, {"a.md": ("rpd", rpd(direction_code=None))})

    paused, _ = run(graph, make_input("r1", "проверь", file_paths=["a.md"]))

    refused = models.calls("orchestrator")[1][-1]
    assert isinstance(refused, ToolMessage)
    assert "нужен код направления: правила ФГОС в каталоге есть для 02.03.02" in refused.content
    assert paused.interrupts[0].value["options"] == ["Проверить без ФГОС"]


def test_entered_direction_code_must_have_rules() -> None:
    models = Models(
        orchestrator=[
            decision(kind="unclear", question="Укажите код направления."),
            decision(kind="check", direction_code="03.03.02"),  # опечатка: правил для кода нет
            decision(kind="unclear", question="Для 03.03.02 правил нет.", options=["02.03.02"]),
            decision(kind="check", direction_code="02.03.02"),
        ]
    )
    graph, _ = make_graph(models, {"a.md": ("rpd", rpd(direction_code=None))})
    run(graph, make_input("r1", "проверь", action="check", file_paths=["a.md"]))

    again, _ = run(graph, Command(resume="03.03.02"))

    refused = models.calls("orchestrator")[2][-1]
    assert isinstance(refused, ToolMessage)
    # код пользователя без правил: другое направление вместо него не предлагается
    assert "для направления 03.03.02 правил ФГОС в каталоге нет" in refused.content
    assert "Не предлагай вместо него другое направление" in refused.content
    assert again.interrupts[0].value["options"] == ["02.03.02"]
    done, _ = run(graph, Command(resume="02.03.02"))
    assert done.values["direction_code"] == "02.03.02"
    assert "Правила ФГОС не применялись" not in done.values["reply"]


def test_ingest_only_with_files() -> None:
    ingestor = Ingestor(RPD_FILES)
    graph, _ = make_graph(Models(qa=[call("QAAnswer", answer="ok")]), ingestor=ingestor)

    _, nodes = run(graph, make_input("r1", "?", action="qa"))
    assert "ingest" not in nodes

    run(graph, make_input("r2", "проверь", action="check", file_paths=["a.md"]))
    assert ingestor.paths == ["a.md"]


def test_unknown_document_is_reported() -> None:
    class Failing(Ingestor):
        async def ingest(self, session_id, path):
            raise ValueError(f"{path}: не удалось определить тип документа по титулу")

    graph, _ = make_graph(Models(), ingestor=Failing({}))

    state, _ = run(graph, make_input("r1", "проверь", action="check", file_paths=["x.docx"]))

    assert "x.docx: не удалось определить тип документа" in state.values["reply"]


def test_run_fields_reset_and_session_data_kept() -> None:
    models = Models(
        orchestrator=[decision(kind="check"), decision(kind="question")],
        qa=[call("QAAnswer", answer="ответ")],
    )
    graph, _ = make_graph(models, RPD_FILES)
    first, _ = run(graph, make_input("r1", "проверь", file_paths=["a.md"]))

    second, nodes = run(graph, make_input("r2", "вопрос"))

    assert "ingest" not in nodes
    assert second.values["check_summary"] is None
    assert second.values["document_ids"] == ["doc:a.md"]
    assert second.values["finding_ids"] == first.values["finding_ids"]
    assert second.values["direction_code"] == "02.03.02"


def test_action_not_inherited_by_next_run() -> None:
    models = Models(orchestrator=[decision(kind="question")], qa=[call("QAAnswer", answer="ok")])
    graph, _ = make_graph(models, RPD_FILES)
    run(graph, make_input("r1", "проверь", action="check", file_paths=["a.md"]))

    state, _ = run(graph, make_input("r2", "вопрос"))

    assert state.values["kind"] == "question"
    assert len(models.calls("orchestrator")) == 1


def test_retry_after_failure_continues_from_last_step() -> None:
    ingestor = Ingestor(RPD_FILES, fail_times=1)
    graph, _ = make_graph(Models(), ingestor=ingestor)

    with pytest.raises(RuntimeError, match="сбой извлечения"):
        run(graph, make_input("r1", "проверь", action="check", file_paths=["a.md"]))
    failed = asyncio.run(graph.aget_state(config()))
    assert failed.next == ("ingest",)
    assert not failed.interrupts

    state, nodes = run(graph, None)

    assert nodes[0] == "ingest"
    assert state.values["reply"]
    assert ingestor.paths == ["a.md", "a.md"]


def test_sessions_are_isolated() -> None:
    graph, _ = make_graph(Models(), RPD_FILES)
    run(graph, make_input("r1", "проверь", action="check", file_paths=["a.md"]), session="s1")

    state, _ = run(graph, make_input("r1", "проверь", action="check"), session="s2")

    assert state.values["document_ids"] == []
    assert "Нет загруженных документов" in state.values["reply"]


def test_invalid_decision_is_retried_with_error() -> None:
    models = Models(orchestrator=[decision(scope={}), decision(kind="check")])
    graph, _ = make_graph(models, RPD_FILES)

    state, _ = run(graph, make_input("r1", "проверь", file_paths=["a.md"]))

    assert state.values["kind"] == "check"
    retry = models.calls("orchestrator")[1][-1]
    assert isinstance(retry, ToolMessage)
    assert "ответ не соответствует схеме" in retry.content and "kind" in retry.content


def test_invalid_decision_gives_up() -> None:
    bad = [decision(scope={}) for _ in range(MAX_SCHEMA_RETRIES + 1)]
    graph, _ = make_graph(Models(orchestrator=bad))

    with pytest.raises(AgentLimitError):
        run(graph, make_input("r1", "проверь"))


def test_scope_passed_as_json_string_is_decoded() -> None:
    scope = '{"courses": ["Физика"], "rule_sets": ["cross_check"]}'
    graph, _ = make_graph(Models(orchestrator=[decision(kind="check", scope=scope)]), RPD_FILES)

    state, _ = run(graph, make_input("r1", "сверь физику", file_paths=["a.md"]))

    assert state.values["scope"]["courses"] == ["Физика"]


def test_scope_courses_must_be_session_documents() -> None:
    """Код не читает запрос: он проверяет итог оркестратора по загруженным документам."""
    program = {"courses": ["Программирование и информационные технологии"]}
    models = Models(
        orchestrator=[
            decision(kind="question", scope=program),  # название программы — не дисциплина
            decision(kind="question", scope={"missing_courses": ["Химия"]}),
        ],
        qa=[call("QAAnswer", answer="ok")],
    )
    graph, _ = make_graph(models, RPD_FILES)

    state, _ = run(graph, make_input("r1", "что в программе?", file_paths=["a.md"]))

    refused = models.calls("orchestrator")[1][-1]
    assert "среди документов нет дисциплин: Программирование и информационные" in (refused.content)
    assert "Загружены: «Физика»" in refused.content
    assert state.values["scope"]["courses"] == ["Химия"]


def test_unknown_courses() -> None:
    titles = ["Физика", "Машинное обучение"]

    assert unknown_courses(["машинное обучение", "Физика", "Химия"], titles) == ["Химия"]


# --- диалог сессии ---------------------------------------------------------------------------


def test_follow_up_sees_previous_turns() -> None:
    models = Models(
        orchestrator=[decision(kind="question"), decision(kind="question")],
        qa=[
            call("QAAnswer", answer="Уточните код направления программы."),
            call("QAAnswer", answer="Для 02.04.02 объём — 120 з.е."),
        ],
    )
    graph, _ = make_graph(models)
    run(graph, make_input("r1", "какие курсы в магистратуре ИИ?"))

    state, _ = run(graph, make_input("r2", "02.04.02"))

    # агенты получают диалог сообщениями, последнее — новый запрос
    qa = [(type(m).__name__, m.content) for m in models.calls("qa")[1][1:]]
    assert qa == [
        ("HumanMessage", "какие курсы в магистратуре ИИ?"),
        ("AIMessage", "Уточните код направления программы."),
        ("HumanMessage", "02.04.02"),
    ]
    assert [m.content for m in models.calls("orchestrator")[1][1:]] == [
        "какие курсы в магистратуре ИИ?",
        "Уточните код направления программы.",
        "02.04.02",
    ]
    assert state.values["messages"][-1].content == "Для 02.04.02 объём — 120 з.е."


def test_long_history_is_summarized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(history_module, "HISTORY_CHARS", 60)
    monkeypatch.setattr(history_module, "KEEP_MESSAGES", 2)
    models = Models(
        orchestrator=[decision(kind="question") for _ in range(3)],
        qa=[call("QAAnswer", answer=f"ответ {i} " + "x" * 20) for i in range(3)],
        summarizer=[
            call("Summary", summary="Спрашивали о вопросе 1."),
            call("Summary", summary="Спрашивали о вопросах 1 и 2."),
        ],
    )
    graph, _ = make_graph(models)
    run(graph, make_input("r1", "вопрос 1"))
    run(graph, make_input("r2", "вопрос 2"))

    state, _ = run(graph, make_input("r3", "вопрос 3"))

    assert state.values["summary"] == "Спрашивали о вопросах 1 и 2."
    # свёрнутые сообщения удалены из состояния (RemoveMessage)
    assert [m.content for m in state.values["messages"]] == ["вопрос 3", "ответ 2 " + "x" * 20]
    assert "Пользователь: вопрос 1" in models.calls("summarizer")[0][-1].content
    assert "Прежнее краткое содержание: Спрашивали о вопросе 1." in (
        models.calls("summarizer")[1][-1].content
    )
    # краткое содержание идёт агенту вместе с последними сообщениями
    request = [m.content for m in models.calls("qa")[2][1:]]
    assert request == [
        "Краткое содержание более раннего диалога: Спрашивали о вопросе 1.",
        "вопрос 2",
        "ответ 1 " + "x" * 20,
        "вопрос 3",
    ]


def test_empty_session_answers_without_session_tools() -> None:
    models = Models(
        qa=[call("find_course", query="программа"), call("QAAnswer", answer="Нет документов.")]
    )
    graph, _ = make_graph(models)

    run(graph, make_input("r1", "какие курсы в программе?", action="qa"))

    system = models.calls("qa")[0][0].content
    assert "Документы сессии: не загружено ни одного документа." in system
    tool_reply = models.calls("qa")[1][-1]
    assert (
        isinstance(tool_reply, ToolMessage) and "Нет инструмента find_course" in tool_reply.content
    )


def test_document_code_without_rules_is_reported() -> None:
    graph, _ = make_graph(Models(), {"a.md": ("rpd", rpd(direction_code="09.03.04"))})

    state, _ = run(graph, make_input("r1", "проверь", action="check", file_paths=["a.md"]))

    reply = state.values["reply"]
    assert "Правила ФГОС не применялись: для направления 09.03.04 их нет в каталоге." in reply
    assert "Итог: проверка неполная." in reply or "Итог: есть нарушения." in reply


def test_code_not_named_by_user_is_refused() -> None:
    models = Models(
        orchestrator=[
            decision(kind="check", direction_code="02.03.02"),  # единственный код каталога
            decision(kind="unclear", question="Укажите код направления.", options=["02.03.02"]),
            decision(kind="check", direction_code="02.03.02"),
        ]
    )
    graph, _ = make_graph(models, {"a.md": ("rpd", rpd(direction_code=None))})

    paused, _ = run(graph, make_input("r1", "проверь", file_paths=["a.md"]))

    refused = models.calls("orchestrator")[1][-1]
    assert "кода 02.03.02 нет ни в документах, ни в репликах пользователя" in refused.content
    done, _ = run(graph, Command(resume="02.03.02"))  # код назвал пользователь
    assert done.values["direction_code"] == "02.03.02"


def test_orchestrator_has_no_ask_user() -> None:
    models = Models(
        orchestrator=[ask("Что именно вас интересует?", ["Зачёты"]), decision(kind="question")],
        qa=[call("QAAnswer", answer="Документов нет.")],
    )
    graph, _ = make_graph(models)

    state, nodes = run(graph, make_input("r1", "сколько зачётов по физике?"))

    refused = models.calls("orchestrator")[1][-1]
    assert isinstance(refused, ToolMessage) and "Нет инструмента ask_user" in refused.content
    assert nodes[-2:] == ["qa", "remember"]


def test_unclear_kind_is_asked_and_decided_again() -> None:
    unclear = decision(kind="unclear", question="Проверить или исправить?", options=["Проверить"])
    models = Models(orchestrator=[unclear, decision(kind="check")])
    graph, _ = make_graph(models, RPD_FILES)

    paused, _ = run(graph, make_input("r1", "сделай с физикой", file_paths=["a.md"]))

    assert paused.interrupts[0].value["question"] == "Проверить или исправить?"
    done, nodes = run(graph, Command(resume="Проверить"))
    assert nodes == ["ask_user", "orchestrator", "check", "reporter", "remember"]
    # ответ пришёл в диалог: оркестратор решает заново по нему
    assert models.calls("orchestrator")[1][-1].content == "Проверить"


def test_declined_fgos_for_code_without_rules_is_reported() -> None:
    models = Models(
        orchestrator=[
            decision(kind="unclear", question="Укажите код направления.", options=["02.03.02"]),
            decision(kind="unclear", question="Для 03.03.02 правил нет. Без ФГОС?"),
            # «без ФГОС»: код пользователя сохраняется, наборы правил сузила модель
            decision(
                kind="check",
                scope={"rule_sets": ["spbu", "cross_check"]},
                skip_fgos=True,
                direction_code="03.03.02",
            ),
        ]
    )
    graph, _ = make_graph(models, {"a.md": ("rpd", rpd(direction_code=None))})
    run(graph, make_input("r1", "проверь", file_paths=["a.md"]))
    run(graph, Command(resume="03.03.02"))

    done, _ = run(graph, Command(resume="Без ФГОС"))

    assert done.values["direction_code"] == "03.03.02"
    reply = done.values["reply"]
    assert "Правила ФГОС не применялись: для направления 03.03.02 их нет в каталоге." in reply


def test_declined_fgos_without_code_is_reported() -> None:
    models = Models(
        orchestrator=[
            decision(kind="unclear", question="Укажите код направления.", options=["02.03.02"]),
            decision(kind="check", skip_fgos=True),
        ]
    )
    graph, _ = make_graph(models, {"a.md": ("rpd", rpd(direction_code=None))})
    run(graph, make_input("r1", "проверь", file_paths=["a.md"]))

    done, _ = run(graph, Command(resume="не надо ФГОС"))

    assert done.values["direction_code"] == ""
    assert "проверка по ФГОС отменена пользователем" in done.values["reply"]
