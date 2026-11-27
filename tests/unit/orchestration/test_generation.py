"""Подграф «Генерация»: правки агента-генератора, проверка черновика кодом, подтверждение."""

import asyncio
import io

import pytest
from docx import Document
from graph_helpers import (
    RULES_DIR,
    Models,
    answer_id,
    ask,
    call,
    config,
    decision,
    entry,
    make_graph,
    plan,
    rpd,
    run,
)
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from edu_doc_agent.dao.storage import LocalFileStorage
from edu_doc_agent.extraction.models import LearningOutcome
from edu_doc_agent.extraction.source import SourceRef
from edu_doc_agent.orchestration import make_input
from edu_doc_agent.orchestration.agent import answer_message
from edu_doc_agent.orchestration.check import select_rules
from edu_doc_agent.orchestration.corrected import _document_patches, _patches
from edu_doc_agent.orchestration.draft import review_draft, unsupported_edits
from edu_doc_agent.orchestration.edits import apply_edits, edit_text, get_value
from edu_doc_agent.orchestration.generation import _checked_documents, fix_groups
from edu_doc_agent.orchestration.state import empty_scope
from edu_doc_agent.orchestration.stores import ChangeEdit, StoredDocument
from edu_doc_agent.validation.catalog import load_rules
from edu_doc_agent.validation.findings import Finding

FID = "cross-check-rpd-credits-vs-workload-table:doc:a.md"


def propose(value, **kw):
    edit = {"course_id": "doc:a.md", "path": "credits", "value": value}
    return call("propose_change", edits=[edit], rationale="как в таблице", **kw)


def setup(*generator, orchestrator=None):
    """Сессия с проверенной РПД: 6 з.е. на титуле при 4 з.е. в таблице трудоёмкости."""
    models = Models(
        orchestrator=orchestrator or [decision(kind="change", scope={"finding_ids": [FID]})],
        generator=list(generator),
    )
    graph, deps = make_graph(models, {"a.md": ("rpd", rpd(title="Физика", credits=6))})
    run(graph, make_input("r1", "проверь", action="check", file_paths=["a.md"]))
    return graph, deps, models


def document(deps):
    return asyncio.run(deps.stores.get_document("s1", "doc:a.md"))


def changes(graph, deps):
    ids = asyncio.run(graph.aget_state(config())).values["change_ids"]
    return [asyncio.run(deps.stores.get_change("s1", i)) for i in ids]


def test_get_and_apply_edits() -> None:
    doc = StoredDocument("d1", "s1", "rpd", rpd(), "a.md")
    assert get_value(doc.data.model_dump(mode="json"), "workload_table/2/credits") == 4

    (changed,), errors = apply_edits(
        [doc], [ChangeEdit(document_id="d1", path="workload_table/2/credits", new=5)]
    )
    assert errors == []
    assert changed.data.workload_table[2].credits == 5

    _, errors = apply_edits([doc], [ChangeEdit(document_id="d1", path="credits", new="много")])
    assert "значение не подходит по типу" in errors[0]
    _, errors = apply_edits([doc], [ChangeEdit(document_id="d1", path="nope/1", new=1)])
    assert errors == ["a.md: нет значения по пути «nope/1»"]


def test_change_is_checked_confirmed_and_applied() -> None:
    graph, deps, models = setup(propose(7), propose(4, finding_ids=[FID]))

    paused, nodes = run(graph, make_input("r2", "исправь трудоёмкость физики"))

    assert nodes == ["start", "orchestrator", "generate", "confirm"]
    # первая правка не исправила замечание: код вернул её агенту
    rejected = models.calls("generator")[1][-1]
    assert isinstance(rejected, ToolMessage)
    assert f"замечание {FID} осталось" in rejected.content
    pending = paused.interrupts[0].value
    assert pending["type"] == "confirmation"
    assert pending["options"] == ["Применить", "Отклонить"]
    assert "Физика, «credits»: было «6», стало «4»" in pending["question"]
    assert (
        "будет снято замечание «Трудоёмкость дисциплины совпадает с итогом" in pending["question"]
    )

    done, nodes = run(graph, Command(resume="Применить"))

    assert nodes == ["ask_user", "apply_change", "reporter", "remember"]
    assert document(deps).data.credits == 4
    assert FID not in done.values["finding_ids"]
    assert "Изменение: внесено в данные." in done.values["reply"]


def test_rejected_change_keeps_data() -> None:
    graph, deps, _ = setup(propose(4))
    run(graph, make_input("r2", "исправь"))

    done, _ = run(graph, Command(resume="Отклонить"))

    assert document(deps).data.credits == 6
    assert FID in done.values["finding_ids"]
    assert "Изменение: отклонено пользователем." in done.values["reply"]


def test_free_text_at_confirmation_goes_to_the_generator() -> None:
    graph, deps, models = setup(propose(4), propose(4))
    run(graph, make_input("r2", "исправь"))

    again, nodes = run(graph, Command(resume="добавь обоснование со ссылкой на таблицу"))

    assert nodes == ["ask_user", "apply_change", "generate", "confirm"]
    # тот же генератор продолжает работу: прежний черновик и ответ пользователя — в ней
    work = models.calls("generator")[1]
    assert any(getattr(m, "tool_calls", None) for m in work)
    assert work[-1].content == (
        "Ответ пользователя на черновик (не кнопка): добавь обоснование со ссылкой на таблицу"
    )
    assert again.interrupts[0].value["type"] == "confirmation"
    statuses = sorted(c.status for c in changes(graph, deps))
    assert statuses == ["draft", "superseded"]


def test_question_at_confirmation_is_explained_and_asked_again() -> None:
    explain = call("explain_draft", answer="В таблице трудоёмкости итог — 4 з.е.")
    graph, deps, models = setup(propose(4, finding_ids=[FID]), explain)
    run(graph, make_input("r2", "исправь"))

    again, nodes = run(graph, Command(resume="объясни расчёт"))

    assert nodes == ["ask_user", "apply_change", "generate", "confirm"]
    pending = again.interrupts[0].value
    assert pending["type"] == "confirmation"
    assert pending["question"].startswith("В таблице трудоёмкости итог — 4 з.е.\n\n")
    assert "Физика, «credits»: было «6», стало «4»" in pending["question"]
    assert [c.status for c in changes(graph, deps)] == ["draft"]  # черновик тот же
    run(graph, Command(resume="Применить"))
    assert document(deps).data.credits == 4


def test_refusal_in_text_rejects_the_draft() -> None:
    graph, deps, _ = setup(propose(4), call("reject_draft", reason="пользователь против"))
    run(graph, make_input("r2", "исправь"))

    done, nodes = run(graph, Command(resume="не надо, пусть останется 6"))

    assert nodes == ["ask_user", "apply_change", "generate", "reporter", "remember"]
    assert document(deps).data.credits == 6
    assert "Изменение: отклонено пользователем." in done.values["reply"]


def test_generator_asks_user_for_missing_data() -> None:
    graph, _, models = setup(ask("Какая трудоёмкость верная?", ["4", "6"]), propose(4))

    paused, _ = run(graph, make_input("r2", "исправь"))

    assert paused.interrupts[0].value == {
        "type": "missing_data",
        "question": "Какая трудоёмкость верная?",
        "options": ["4", "6"],
    }
    confirm, nodes = run(graph, Command(resume="4"))

    assert nodes == ["ask_user", "generate", "confirm"]
    assert models.calls("generator")[1][-1].content.endswith("выбран вариант): 4")
    assert confirm.interrupts[0].value["type"] == "confirmation"


def test_check_and_fix_targets_fixable_findings() -> None:
    models = Models(orchestrator=[decision(kind="check_and_fix")], generator=[propose(4)])
    graph, _ = make_graph(models, {"a.md": ("rpd", rpd(title="Физика", credits=6))})

    paused, nodes = run(graph, make_input("r1", "проверь и исправь", file_paths=["a.md"]))

    assert nodes[-3:] == ["check", "generate", "confirm"]
    goal = models.calls("generator")[0][-1].content
    assert f"- {FID}: Трудоёмкость дисциплины совпадает с итогом" in goal


def test_remove_and_append_list_items() -> None:
    doc = StoredDocument("d1", "s1", "rpd", rpd(competences=["УК-1", "УК-2", "УК-3"]), "a.md")
    edits = [
        ChangeEdit(document_id="d1", path="competences/0", op="remove"),
        ChangeEdit(document_id="d1", path="competences/2", op="remove"),
        ChangeEdit(
            document_id="d1",
            path="competences",
            op="append",
            new={"code": "ОПК-1", "source": {"file": "a.md"}},
        ),
    ]

    (changed,), errors = apply_edits([doc], edits)

    assert errors == []
    assert [c.code for c in changed.data.competences] == ["УК-2", "ОПК-1"]
    _, errors = apply_edits(
        [doc], [ChangeEdit(document_id="d1", path="competences/7", op="remove")]
    )
    assert errors == ["a.md: нет элемента списка «competences/7»"]


@pytest.mark.parametrize("bad_path", ["credit", "workload_table/9/credits"])
def test_wrong_path_returned_to_generator(bad_path: str) -> None:
    wrong = call(
        "propose_change",
        edits=[{"course_id": "doc:a.md", "path": bad_path, "value": 4}],
        rationale="",
    )
    graph, _, models = setup(wrong, propose(4))

    run(graph, make_input("r2", "исправь"))

    assert f"нет значения по пути «{bad_path}»" in models.calls("generator")[1][-1].content


def test_whole_workload_table_edit_is_reviewed() -> None:
    """Правка всей таблицы трудоёмкости, а не одной строки, проверяется без сбоя."""
    _, deps, _ = setup()
    documents = asyncio.run(deps.stores.list_documents("s1"))
    table = documents[0].data.model_dump(mode="json")["workload_table"]
    edit = ChangeEdit(document_id="doc:a.md", path="workload_table", new=table)
    rules = select_rules(deps.rules_dir, "02.03.02", empty_scope())

    review = asyncio.run(review_draft(deps, "s1", "r2", rules, documents, [edit], []))

    assert review.problems == []
    assert not any("всего часов" in c for c in review.consequences)


# --- проверка с исправлением по документам и исправленный файл ------------------------------


def fix_two_documents(*answers: str):
    edit = lambda doc: {"course_id": doc, "path": "credits", "value": 4}  # noqa: E731
    models = Models(
        orchestrator=[decision(kind="check_and_fix")],
        generator=[
            call("propose_change", edits=[edit("doc:a.md")], rationale="как в таблице"),
            call("propose_change", edits=[edit("doc:b.md")], rationale="как в таблице"),
        ],
    )
    documents = {
        "a.md": ("rpd", rpd(title="Физика", credits=6)),
        "b.md": ("rpd", rpd(title="Химия", credits=6)),
    }
    graph, deps = make_graph(models, documents)
    state, _ = run(graph, make_input("r1", "проверь и исправь", file_paths=["a.md", "b.md"]))
    steps = [state]
    for answer in answers:
        state, _ = run(graph, Command(resume=answer))
        steps.append(state)
    return steps, models


def test_check_and_fix_confirms_each_document() -> None:
    (first, second, done), models = fix_two_documents("Применить", "Отклонить")

    # одно изменение на документ: сначала «Физика», потом «Химия»
    pending = first.interrupts[0].value
    assert "Физика, «credits»: было «6», стало «4»" in pending["question"]
    assert "После этого документа к исправлению ещё: 1." in pending["question"]
    assert pending["options"] == ["Применить", "Отклонить", "Завершить исправления"]
    pending = second.interrupts[0].value
    assert "Химия, «credits»" in pending["question"]
    assert pending["options"] == ["Применить", "Отклонить"]  # документов больше нет
    reply = done.values["reply"]
    assert "Изменение: внесено в данные." in reply
    assert "Изменение: отклонено пользователем." in reply
    goal = models.calls("generator")[1][-1].content
    assert "Химия" in goal and "Физика" not in goal  # цель — замечания одного документа
    # следующий документ генератор начинает с чистой работы, а не с работы над прежним
    assert not any(getattr(m, "tool_calls", None) for m in models.calls("generator")[1])


def test_stop_finishes_check_and_fix() -> None:
    (_, done), models = fix_two_documents("Завершить исправления")

    assert done.next == ()
    assert len(models.calls("generator")) == 1  # «Химию» не исправляли
    assert "Изменение: отклонено пользователем." in done.values["reply"]


def test_applied_change_produces_corrected_document(tmp_path) -> None:
    path = tmp_path / "a.docx"
    original = Document()
    original.add_paragraph("Трудоемкость в зачетных единицах: 6")
    original.save(path)
    title = SourceRef(file="a.docx", line=1, quote="Трудоемкость в зачетных единицах: 6")
    course = rpd(title="Физика", credits=6, sources={"credits": title})
    fid = "cross-check-rpd-credits-vs-workload-table:doc:a.docx"
    edit = {"course_id": "doc:a.docx", "path": "credits", "value": 4}
    models = Models(
        orchestrator=[decision(kind="change", scope={"finding_ids": [fid]})],
        generator=[call("propose_change", edits=[edit], rationale="как в таблице")],
    )
    storage = LocalFileStorage(tmp_path / "storage")
    graph, _ = make_graph(models, {str(path): ("rpd", course)}, storage=storage)
    run(graph, make_input("r1", "проверь", action="check", file_paths=[str(path)]))
    run(graph, make_input("r2", "исправь трудоёмкость"))

    done, _ = run(graph, Command(resume="Применить"))

    (corrected,) = done.values["files"]
    assert corrected["kind"] == "corrected_document"
    assert corrected["filename"] == "a (исправлено).docx"
    assert corrected["derived_from"] == "a.docx"
    content = asyncio.run(storage.read(corrected["storage_key"]))
    assert Document(io.BytesIO(content)).paragraphs[0].text == "Трудоемкость в зачетных единицах: 4"
    assert "Документ: a (исправлено).docx: перенесено правок 1 из 1" in done.values["reply"]


def test_whole_item_edit_shows_and_transfers_changed_fields() -> None:
    old = {"code": "УК-12", "name": "способен", "source": {"file": "a.docx"}}
    edit = ChangeEdit(document_id="d1", path="competences/0", old=old, new={**old, "code": "УК-1"})

    assert edit_text(edit, {"d1": "Физика"}) == "Физика, «competences/0»: code: «УК-12» → «УК-1»"
    src = SourceRef(file="a.docx", line=1, quote="| УК-12 способен")
    course = rpd(competences=["УК-12"])
    course.competences[0].source = src
    doc = StoredDocument("d1", "s1", "rpd", course, "a.docx")
    (patch,) = _patches(doc, edit)
    assert (patch.label, patch.old, patch.new, patch.quote) == (
        "competences/0/code",
        "УК-12",
        "УК-1",
        "| УК-12 способен",
    )


def test_removals_never_lose_document_content() -> None:
    row = SourceRef(file="a.docx", line=1, quote="| УК-12 способен | знает: поиск |")
    course = rpd(competences=["УК-12"])
    course.competences[0].source = row
    course.learning_outcomes = [
        LearningOutcome(kind="знает", text="поиск", source=row),
        LearningOutcome(kind="умеет", text="анализ", source=row),
    ]
    doc = StoredDocument("d1", "s1", "rpd", course, "a.docx")

    def remove(path: str) -> ChangeEdit:
        return ChangeEdit(document_id="d1", path=path, op="remove")

    # замена компетенции удалением и добавлением; её результаты — в той же строке
    replace_edits = [
        remove("competences/0"),
        ChangeEdit(document_id="d1", path="competences", op="append", new={"code": "УК-1"}),
        remove("learning_outcomes/0"),
        remove("learning_outcomes/1"),
    ]
    reasons = {p.label: p.reason for p in _document_patches(doc, replace_edits)}
    assert reasons["competences/0"].startswith("элемент заменён другим")
    assert reasons["learning_outcomes/0"].startswith("место в документе общее")
    assert reasons["learning_outcomes/1"].startswith("место в документе общее")

    # удаление одного результата: строку делят другие — вручную
    (patch,) = _document_patches(doc, [remove("learning_outcomes/0")])
    assert patch.reason and patch.reason.startswith("место в документе общее")


def test_program_findings_are_fixed_last_and_name_their_documents() -> None:
    def finding(fid: str, course: str | None) -> Finding:
        return Finding(
            id=fid,
            session_id="s1",
            run_id="r1",
            rule_id=fid,
            rule_title=fid,
            rule_set="cross_check",
            severity="error",
            fixable=True,
            document_id=course and f"doc:{course}",
            course=course,
            status="violation",
            message="нарушено",
        )

    groups = fix_groups([finding("p1", None), finding("p2", None), finding("f1", "Физика")])

    assert [[f.id for f in g] for g in groups] == [["f1"], ["p1", "p2"]]
    documents = [
        StoredDocument("c1", "s1", "characteristic", rpd(), "char.docx"),
        StoredDocument("c2", "s1", "characteristic", rpd(), "char_new.docx"),
        StoredDocument("r1", "s1", "rpd", rpd(), "a.docx"),
    ]
    # при двух характеристиках замечания программы — о последней загруженной
    assert _checked_documents(documents) == (
        "Замечания уровня программы относятся к документам: char_new.docx (course_id c2)."
    )


def test_new_goals_text_comes_from_the_user() -> None:
    draft = "Цель — изучить методы машинного обучения."
    goals = {"course_id": "doc:a.md", "path": "goals", "value": draft}
    asking = ask("Какие цели записать?", [draft])
    models = Models(
        orchestrator=[decision(kind="change", scope={"change_goal": "заменить цели физики"})],
        generator=[
            call("propose_change", edits=[goals], rationale="цели о борще"),  # придумал сам
            asking,
            call(
                "propose_change",
                edits=[{**goals, "answer_id": answer_id(asking)}],
                rationale="цели, которые выбрал пользователь",
            ),
        ],
    )
    graph, _ = make_graph(models, {"a.md": ("rpd", rpd(title="Физика", goals="Борщ."))})
    run(graph, make_input("r1", "проверь", action="check", file_paths=["a.md"]))

    asked, _ = run(graph, make_input("r2", "замени цели физики"))

    rejected = models.calls("generator")[1][-1]
    assert isinstance(rejected, ToolMessage)
    assert "goals: новый текст не придумывай" in rejected.content
    assert asked.interrupts[0].value["options"] == [draft]
    confirm, _ = run(graph, Command(resume=draft))  # пользователь выбрал вариант генератора
    assert confirm.interrupts[0].value["type"] == "confirmation"
    assert f"стало «{draft}»" in confirm.interrupts[0].value["question"]


# --- правки без опоры: проверка кодом по данным и ответам ask_user ---------------------------


def _support_case():
    course = rpd(title="Физика", discipline_code="001863", competences=["УК-12"])
    # формулировка в документе — УК-1 по ФГОС: код ошибочный, компетенция та же
    course.competences[0].name = (
        "способен осуществлять поиск, критический анализ и синтез информации, применять "
        "системный подход для решения поставленных задач"
    )
    documents = {"d1": StoredDocument("d1", "s1", "rpd", course, "a.docx")}
    curriculum = StoredDocument(
        "p1", "s1", "plan", plan(entry("Физика", "001863", 6, ["ОПК-1"])), "plan.docx"
    )
    rules = select_rules(RULES_DIR, "02.03.02", empty_scope())
    history = [
        answer_message("q1", "УК-12", []),
        answer_message("q2", "УК-3", ["УК-3", "УК-4"]),
        answer_message("q3", "Цель — изучить физику.", []),
    ]
    return documents, curriculum, rules, history


def _problems(path, value, *, op="set", answer=None, old="УК-12", with_plan=False):
    documents, curriculum, rules, history = _support_case()
    if with_plan:
        documents = {**documents, "p1": curriculum}
    edit = ChangeEdit(document_id="d1", path=path, op=op, old=old, new=value)
    return unsupported_edits([edit], [answer], documents, rules, history)


def test_new_competence_code_needs_grounding() -> None:
    code = "competences/0/code"
    # формулировка компетенции совпадает с ФГОС для УК-1
    assert _problems(code, "УК-1") == []
    # УК-8 подобран по смыслу: ни в плане, ни по формулировке
    (problem,) = _problems(code, "УК-8")
    assert "код УК-8 не подтверждён (учебного плана нет" in problem
    (problem,) = _problems(code, "УК-8", with_plan=True)
    assert "в учебном плане у дисциплины его нет" in problem
    assert _problems(code, "ОПК-1", with_plan=True) == []  # код из строки учебного плана
    # код из ответа пользователя: выбранный вариант; «УК-12» — не «УК-1»
    assert _problems(code, "УК-3", answer="q2") == []
    assert _problems(code, "УК-8", answer="q1") != []
    assert _problems(code, "УК-8", answer="нет-такого") != []


def test_competence_is_removed_only_with_user_consent() -> None:
    (problem,) = _problems("competences/0", None, op="remove", old=None)
    assert "компетенцию не удаляй без согласия пользователя" in problem
    assert _problems("competences/0", None, op="remove", old=None, answer="q1") == []


def test_authored_text_must_be_the_user_answer() -> None:
    (problem,) = _problems("goals", "Цель — изучить физику.", old="Борщ.")
    assert "goals: новый текст не придумывай" in problem
    assert _problems("goals", "Цель — изучить физику.", old="Борщ.", answer="q3") == []
    (problem,) = _problems("goals", "Цель — изучить физику и химию.", old="Борщ.", answer="q3")
    assert "goals: текст не совпадает с ответом q3" in problem


def test_user_can_keep_a_target_finding_unfixed() -> None:
    keep = call(
        "propose_change",
        edits=[{"course_id": "doc:a.md", "path": "title", "value": "Физика (общая)"}],
        rationale="без трудоёмкости",
        skipped_finding_ids=[FID],
    )
    graph, deps, _ = setup(propose(4), keep)
    run(graph, make_input("r2", "исправь"))

    again, _ = run(graph, Command(resume="оставь 6, это верно"))

    # черновик без правки трудоёмкости принят: пропущенное замечание его не держит
    draft = [c for c in changes(graph, deps) if c.status == "draft"]
    assert [e.path for e in draft[0].edits] == ["title"]
    assert FID not in draft[0].finding_ids
    assert again.interrupts[0].value["type"] == "confirmation"


def two_edit_draft():
    edits = [
        {"course_id": "doc:a.md", "path": "credits", "value": 4},
        {"course_id": "doc:a.md", "path": "title", "value": "Физика (общая)"},
    ]
    return call("propose_change", edits=edits, rationale="как в таблице", finding_ids=[FID])


def test_dropping_part_of_the_draft_keeps_the_rest() -> None:
    drop = call("drop_edits", numbers=[1], reason="пользователь оставляет 6 з.е.")
    graph, deps, models = setup(two_edit_draft(), drop)
    run(graph, make_input("r2", "исправь"))

    again, nodes = run(graph, Command(resume="оставь 6"))

    assert nodes == ["ask_user", "apply_change", "generate", "confirm"]
    old, new = changes(graph, deps)
    assert old.status == "superseded"
    assert [e.path for e in new.edits] == ["title"]
    assert new.skipped_finding_ids == [FID] and new.finding_ids == []
    question = again.interrupts[0].value["question"]
    assert "1. Физика, «title»" in question and "«credits»" not in question
    assert "останется замечание «Трудоёмкость дисциплины совпадает" in question
    run(graph, Command(resume="Применить"))
    assert (document(deps).data.credits, document(deps).data.title) == (6, "Физика (общая)")


def test_declined_finding_is_shown_to_the_next_generator() -> None:
    drop = call("drop_edits", numbers=[1], reason="оставить 6")
    graph, _, models = setup(
        two_edit_draft(),
        drop,
        call(
            "propose_change",
            edits=[{"course_id": "doc:a.md", "path": "title", "value": "Физика (общая)"}],
            rationale="трудоёмкость пользователь оставил",
            skipped_finding_ids=[FID],
        ),
        orchestrator=[decision(kind="change", scope={"finding_ids": [FID]}) for _ in range(2)],
    )
    run(graph, make_input("r2", "исправь"))
    run(graph, Command(resume="оставь 6"))
    run(graph, Command(resume="Отклонить"))

    run(graph, make_input("r3", "Применить"))

    request = models.calls("generator")[2][-1].content
    assert f"Пользователь раньше просил не исправлять замечания: {FID}" in request


def test_explanation_without_draft_goes_to_the_question_comment() -> None:
    graph, _, models = setup(
        ask("Какая трудоёмкость верная?", ["4", "6"]),
        call("explain_draft", answer="Нужно знать трудоёмкость."),
        call("ask_user", question="Какая верная?", options=["4", "6"], comment="Из таблицы."),
    )
    run(graph, make_input("r2", "исправь"))

    again, _ = run(graph, Command(resume="зачем вы спрашиваете?"))

    refused = models.calls("generator")[2][-1]
    assert isinstance(refused, ToolMessage) and "comment вызова ask_user" in refused.content
    assert again.interrupts[0].value["question"] == "Из таблицы.\n\nКакая верная?"


GOALS_RULE = """\
- id: cross-check-goals
  title: Цели согласуются с результатами
  rule_set: cross_check
  applies_to: course
  doc_kinds: [discipline]
  method: llm
  criterion: Цели и результаты обучения об одной дисциплине.
  fields: [goals]
  severity: warning
  fixable: true
"""


@pytest.mark.parametrize("from_user", [True, False])
def test_llm_violation_of_user_text_is_a_warning(tmp_path, from_user: bool) -> None:
    (tmp_path / "cross_check.yaml").write_text(GOALS_RULE, encoding="utf-8")
    quote = [{"document": "a.md", "quote": "научить готовить борщ"}]
    models = Models(
        checker=[call("CheckerAnswer", satisfied=False, explanation="о кулинарии", quotes=quote)],
        reviewer=[call("ReviewerVerdict", confirm=True, reason="верно")],
    )
    _, deps = make_graph(models, rules_dir=tmp_path)
    doc = StoredDocument(
        "d1", "s1", "rpd", rpd(goals="Борщ."), "a.md", text="Цель — научить готовить борщ."
    )
    edit = ChangeEdit(document_id="d1", path="goals", old="Борщ.", new="Суп.")
    paths = ["goals"] if from_user else []

    review = asyncio.run(
        review_draft(
            deps, "s1", "r1", load_rules(tmp_path, []), [doc], [edit], [], user_paths=paths
        )
    )

    warning = "после правки будет нарушено «Цели согласуются с результатами»"
    if from_user:
        assert review.problems == [] and any(warning in c for c in review.consequences)
    else:
        assert review.problems and "после правки нарушено" in review.problems[0]


def test_draft_text_is_accepted_by_the_code_button() -> None:
    draft = "Цель — изучить методы машинного обучения."
    asking = call("ask_user", question="Подходит такой текст целей?", draft=draft)
    goals = {"course_id": "doc:a.md", "path": "goals", "value": draft}
    models = Models(
        orchestrator=[decision(kind="change", scope={"change_goal": "заменить цели физики"})],
        generator=[
            asking,
            call(
                "propose_change",
                edits=[{**goals, "answer_id": answer_id(asking)}],
                rationale="цели, которые принял пользователь",
            ),
        ],
    )
    graph, _ = make_graph(models, {"a.md": ("rpd", rpd(title="Физика", goals="Борщ."))})
    run(graph, make_input("r1", "проверь", action="check", file_paths=["a.md"]))

    asked, _ = run(graph, make_input("r2", "возьми цели из РПД по ML"))

    pending = asked.interrupts[0].value
    assert pending["options"] == ["Принять текст"]
    assert pending["question"] == f"Подходит такой текст целей?\n\n«{draft}»"
    confirm, _ = run(graph, Command(resume="Принять текст"))
    reply = models.calls("generator")[1][-1]
    assert reply.artifact == {"answer": draft, "option": True}
    assert confirm.interrupts[0].value["type"] == "confirmation"


def test_plan_is_not_changed_without_user() -> None:
    from graph_helpers import entry, plan

    from edu_doc_agent.orchestration.draft import unsupported_edits

    curriculum = StoredDocument(
        "p1", "s1", "plan", plan(entry("Физика", "001863", 6, ["УК-1"])), "plan.docx"
    )
    edit = ChangeEdit(document_id="p1", path="entries/0/credits", old=6, new=8)

    (problem,) = unsupported_edits([edit], [None], {"p1": curriculum}, [], [])

    assert "учебный план — источник истины, его не меняй" in problem


def test_internal_names_in_user_texts() -> None:
    from edu_doc_agent.orchestration.draft import internal_names
    from edu_doc_agent.validation.catalog import load_rule_file

    rules = load_rule_file(RULES_DIR / "fgos" / "02.03.02.yaml")
    documents = [StoredDocument("74a84e29051e3008", "s1", "rpd", rpd(title="Физика"), "a.md")]
    text = (
        "Правило fgos-02.03.02-uk-codes: итоговая строка (is_total: true) 6 з.е., "
        "в поле «credits» — 8; документ 74a84e29051e3008."
    )

    found = internal_names([text], rules, documents)

    assert found == ["74a84e29051e3008", "credits", "fgos-02.03.02-uk-codes", "is_total"]
    assert (
        internal_names(["Итог таблицы трудоёмкости — 6 з.е., в разделе 1.3 — 8."], rules, documents)
        == []
    )
