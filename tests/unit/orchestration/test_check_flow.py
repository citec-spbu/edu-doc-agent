"""Подграф «Проверка»: правила каталога из rules/, проверка кодом, агент-чекер и агент-ревьюер."""

import asyncio
import io
from pathlib import Path

import pytest
from docx import Document
from graph_helpers import (
    Models,
    call,
    decision,
    entry,
    findings,
    make_graph,
    plan,
    row,
    rpd,
    run,
)

from edu_doc_agent.dao.storage import LocalFileStorage
from edu_doc_agent.orchestration import check as check_module
from edu_doc_agent.orchestration import make_input
from edu_doc_agent.orchestration.check import course_matches
from edu_doc_agent.orchestration.report import MAX_CHAT_FINDINGS, _unchecked_lines, check_report
from edu_doc_agent.validation.findings import Finding


def broken_rpd():
    return rpd(
        title="Физика",
        credits=6,
        competences=["УК-1", "УК-12", "ОПК-9"],
        workload_table=[row(1, 2, 36, 36), row(2, 2, 36, 36), row(None, 5, 72, 72)],
    )


def check(graph, run_id: str, files: list[str]) -> dict:
    state, _ = run(graph, make_input(run_id, "проверь", action="check", file_paths=files))
    return state.values


def test_check_reports_violations_with_quote() -> None:
    graph, _ = make_graph(Models(), {"ok.md": ("rpd", rpd()), "bad.md": ("rpd", broken_rpd())})

    reply = check(graph, "r1", ["ok.md", "bad.md"])["reply"]

    assert "нарушений: 5" in reply
    assert "Итог: есть нарушения." in reply
    assert "итого (очная): 144 ч при 5 з.е., нужно 180 ч" in reply  # 1 з.е. = 36 ак. ч (СПбГУ)
    assert "Физика. Универсальные компетенции дисциплины есть во ФГОС (ошибка)\n" in reply
    assert "коды не из перечня: УК-12" in reply
    assert "коды не из перечня: ОПК-9" in reply
    assert (
        "Основание: fgos-02.03.02:3.2 (808 ред. от 08.02.2021): "
        "«Программа бакалавриата должна устанавливать следующие универсальные компетенции:»"
    ) in reply
    assert "итого (очная): 5 з.е., по семестрам 4 з.е." in reply
    assert "указано 6 з.е., в таблице (итого (очная)) 5 з.е." in reply
    assert "Математический анализ. Универсальные" not in reply
    assert (
        "Редакции нормативов: fgos-02.03.02: 808 ред. от 08.02.2021; "
        "spbu-os: 11144/1 от 24.08.2026."
    ) in reply


def test_recheck_replaces_findings() -> None:
    documents = {"bad.md": ("rpd", broken_rpd())}
    graph, deps = make_graph(Models(), documents)
    check(graph, "r1", ["bad.md"])

    # исправленная РПД: новая загрузка файла с тем же именем, ID документа тот же
    documents["v2/bad.md"] = ("rpd", rpd(title="Физика"))
    values = check(graph, "r2", ["bad.md", "v2/bad.md"])

    assert "нарушений: 0" in values["reply"]
    violations = [f for f in findings(deps, values["finding_ids"]) if f.status == "violation"]
    assert violations == []
    assert findings(deps, ["fgos-02.03.02-uk-codes:doc:bad.md"]) == []


def test_findings_are_not_duplicated_on_repeat() -> None:
    graph, _ = make_graph(Models(), {"bad.md": ("rpd", broken_rpd())})

    first = check(graph, "r1", ["bad.md"])["finding_ids"]
    second = check(graph, "r2", [])["finding_ids"]

    assert sorted(second) == sorted(first)
    assert len(set(second)) == len(second)


def test_documents_and_program_rules_with_plan() -> None:
    course = rpd(title="Физика", credits=4, discipline_code="001863", competences=["УК-1", "ОПК-2"])
    curriculum = plan(
        entry("Физика", "001863", 3, ["УК-1"], semester=3),
        entry("Учебная практика", "064682", 10, ["УК-1"], block="Блок.2..прки", semester=2),
        entry("Философия", "000001", 3, ["УК-5"]),
        gia=12,
    )
    graph, _ = make_graph(Models(), {"a.md": ("rpd", course), "plan.md": ("plan", curriculum)})

    reply = check(graph, "r1", ["a.md", "plan.md"])["reply"]

    assert "в РПД 4 з.е., в учебном плане 3 з.е." in reply
    assert "только в РПД: ОПК-2" in reply
    assert "ГИА 12 з.е., нужно 6–9 з.е." in reply
    assert "план: 10 з.е., нужно не менее 20" in reply
    assert "нет дисциплин: история, иностранный язык, безопасность жизнедеятельности" in reply


def test_documents_rule_without_plan_is_not_checked() -> None:
    graph, deps = make_graph(Models(), {"a.md": ("rpd", rpd())})

    values = check(graph, "r1", ["a.md"])

    unchecked = [f for f in findings(deps, values["finding_ids"]) if f.status == "not_checked"]
    # непроверенные правила одной причины — одной строкой
    line = next(
        x
        for x in _unchecked_lines(unchecked)
        if x.startswith("- Математический анализ: нет учебного плана")
    )
    assert "Трудоёмкость дисциплины совпадает в учебном плане и РПД" in line
    # в чате — первые строки, остальное в полном протоколе
    assert "…и ещё строк:" in values["reply"]


def test_scope_limits_courses_and_rule_sets() -> None:
    scope = {"courses": ["Физика"], "rule_sets": ["cross_check"]}
    models = Models(orchestrator=[decision(kind="check", scope=scope)])
    graph, _ = make_graph(models, {"ok.md": ("rpd", rpd()), "bad.md": ("rpd", broken_rpd())})

    state, _ = run(
        graph, make_input("r1", "сверь трудоёмкость физики", file_paths=["ok.md", "bad.md"])
    )

    reply = state.values["reply"]
    assert "нарушений: 2" in reply
    assert "ФГОС" not in reply.split("Каталог правил")[0]


# --- агент-чекер и агент-ревьюер -------------------------------------------------------------

GOALS = "Цель дисциплины — научить готовить борщ."
LLM_RULES = """\
- id: cross-check-goals
  title: Цели согласуются с результатами
  rule_set: cross_check
  applies_to: course
  doc_kinds: [discipline]
  method: llm
  criterion: Цели и результаты обучения об одной дисциплине.
  fields: [title, goals]
  severity: warning
  fixable: true
"""


@pytest.fixture
def llm_rules(tmp_path: Path) -> Path:
    (tmp_path / "cross_check.yaml").write_text(LLM_RULES, encoding="utf-8")
    return tmp_path


def answer(satisfied: bool, quote: str | None = None, explanation: str = "объяснение"):
    quotes = [{"document": "a.md", "quote": quote}] if quote else []
    return call("CheckerAnswer", satisfied=satisfied, explanation=explanation, quotes=quotes)


def run_checker(llm_rules: Path, models: Models):
    graph, deps = make_graph(models, {"a.md": ("rpd", rpd(goals=GOALS))}, rules_dir=llm_rules)
    values = check(graph, "r1", ["a.md"])
    return findings(deps, values["finding_ids"]), values["reply"]


def test_checker_satisfied(llm_rules: Path) -> None:
    models = Models(checker=[answer(True)])

    found, _ = run_checker(llm_rules, models)

    assert found == []
    assert models.calls("reviewer") == []


def test_checker_quote_fixed_then_reviewer_confirms(llm_rules: Path) -> None:
    models = Models(
        checker=[answer(False, "научить варить суп"), answer(False, "научить готовить борщ")],
        reviewer=[call("ReviewerVerdict", confirm=True, reason="цели о кулинарии")],
    )

    found, reply = run_checker(llm_rules, models)

    (f,) = [f for f in found if f.rule_id == "cross-check-goals"]
    assert f.status == "violation"
    assert f.method == "llm"
    assert f.evidence[0].quote == "научить готовить борщ"
    fix_request = models.calls("checker")[1][-1].content
    assert "цитаты «научить варить суп» нет в документе «a.md» дословно" in fix_request
    assert "Цитата (a.md): «научить готовить борщ»" in reply
    review = models.calls("reviewer")[0][-1].content
    assert "Данные документов (JSON)" in review and GOALS in review  # те же данные, что у чекера


def test_checker_withdraws_after_review(llm_rules: Path) -> None:
    fake = [{"document": "a.md", "quote": "выдумка"}]
    models = Models(
        checker=[
            answer(False, "научить готовить борщ"),
            call("CheckerReply", keep=False, explanation="согласен"),
        ],
        reviewer=[call("ReviewerVerdict", confirm=False, reason="это пример", quotes=fake)],
    )

    found, _ = run_checker(llm_rules, models)

    assert [f for f in found if f.rule_id == "cross-check-goals"] == []
    # возражение ревьюера уходит чекеру; цитату, которой нет в документе, код помечает
    objection = models.calls("checker")[1][-1].content
    assert "Агент-ревьюер не согласен: это пример" in objection
    assert "[a.md] «выдумка» (нет в документе)" in objection


def test_disagreement_needs_human(llm_rules: Path) -> None:
    keep = [call("CheckerReply", keep=True, explanation="нарушение есть") for _ in range(3)]
    dispute = [call("ReviewerVerdict", confirm=False, reason="нарушения нет") for _ in range(3)]
    models = Models(checker=[answer(False, "научить готовить борщ"), *keep], reviewer=dispute)

    found, reply = run_checker(llm_rules, models)

    (f,) = [f for f in found if f.rule_id == "cross-check-goals"]
    assert f.needs_human
    assert f.opinions == ["Агент-чекер: нарушение есть", "Агент-ревьюер: нарушения нет"]
    assert "требует проверки человеком" in reply
    assert "Чекер оставляет нарушение: нарушение есть" in models.calls("reviewer")[1][-1].content


def test_checker_gets_program_context(llm_rules: Path) -> None:
    course = rpd(title="Физика", goals=GOALS, discipline_code="001863")
    curriculum = plan(entry("Физика", "001863", 4, ["УК-1"], semester=3), gia=6)
    models = Models(checker=[answer(True)])
    graph, _ = make_graph(
        models, {"a.md": ("rpd", course), "plan.md": ("plan", curriculum)}, rules_dir=llm_rules
    )

    check(graph, "r1", ["a.md", "plan.md"])

    request = models.calls("checker")[0][1].content
    assert "Контекст программы:" in request
    assert "Учебный план: Физика, семестр 3, 4 з.е." in request
    assert "Компетенции, закреплённые за дисциплиной в учебном плане:\n- УК-1" in request


def test_checker_without_valid_quotes_is_not_checked(llm_rules: Path) -> None:
    models = Models(checker=[answer(False, "выдумка") for _ in range(5)])

    found, _ = run_checker(llm_rules, models)

    (f,) = [f for f in found if f.rule_id == "cross-check-goals"]
    assert f.status == "not_checked"
    assert "за 4 исправления" in f.message


def test_quotes_of_satisfied_answer_are_not_verified(llm_rules: Path) -> None:
    models = Models(checker=[answer(True, "знает: выдумка")])

    found, _ = run_checker(llm_rules, models)

    assert found == []
    assert len(models.calls("checker")) == 1


def test_outcome_kind_label_is_removed_from_quote(llm_rules: Path) -> None:
    models = Models(
        checker=[answer(False, "знает: научить готовить борщ")],
        reviewer=[call("ReviewerVerdict", confirm=True, reason="цели о кулинарии")],
    )

    found, _ = run_checker(llm_rules, models)

    (f,) = [f for f in found if f.rule_id == "cross-check-goals"]
    assert f.evidence[0].quote == "научить готовить борщ"
    assert len(models.calls("checker")) == 1


def test_fix_request_suggests_similar_fragment(llm_rules: Path) -> None:
    models = Models(
        checker=[answer(False, "научить готовит борщ"), answer(False, "научить готовить борщ")],
        reviewer=[call("ReviewerVerdict", confirm=True, reason="цели о кулинарии")],
    )

    run_checker(llm_rules, models)

    fix_request = models.calls("checker")[1][-1].content
    assert "похожий фрагмент документа: «научить готовить борщ.»" in fix_request


def test_missing_field_is_not_checked_without_llm(llm_rules: Path) -> None:
    models = Models()
    graph, deps = make_graph(models, {"a.md": ("rpd", rpd(goals=None))}, rules_dir=llm_rules)

    values = check(graph, "r1", ["a.md"])

    (f,) = [f for f in findings(deps, values["finding_ids"]) if f.rule_id == "cross-check-goals"]
    assert f.status == "not_checked"
    assert f.message == "в документе не найдены: цели дисциплины"
    assert models.calls("checker") == []


def test_hanging_llm_rule_is_not_checked(llm_rules: Path, monkeypatch) -> None:
    async def hang(*args, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(check_module, "LLM_RULE_TIMEOUT", 0.01)
    monkeypatch.setattr(check_module, "check_rule", hang)

    found, _ = run_checker(llm_rules, Models())

    (f,) = [f for f in found if f.rule_id == "cross-check-goals"]
    assert f.status == "not_checked"
    assert f.message.startswith("LLM не ответила за")


def test_ingested_files_are_skipped() -> None:
    graph, deps = make_graph(Models(), {"a.md": ("rpd", rpd())})
    check(graph, "r1", ["a.md"])

    _, nodes = run(graph, make_input("r2", "проверь", action="check", file_paths=["a.md"]))

    assert "ingest" not in nodes
    assert deps.ingestor.paths == ["a.md"]


def test_course_names_match_by_word_stems() -> None:
    assert course_matches("Машинное обучение", "РПД по машинному обучению")
    assert course_matches("Математический анализ I", "матанализ") is False
    assert course_matches("Физика", "физика")


def test_rule_set_without_rules_is_reported(llm_rules: Path) -> None:
    scope = {"rule_sets": ["spbu"]}  # в каталоге llm_rules только cross_check
    graph, _ = make_graph(
        Models(orchestrator=[decision(kind="check", scope=scope)]),
        {"a.md": ("rpd", rpd())},
        rules_dir=llm_rules,
    )

    state, _ = run(graph, make_input("r1", "проверь по нормам СПбГУ", file_paths=["a.md"]))

    reply = state.values["reply"]
    assert "Нет правил проверки для выбранных наборов (spbu)." in reply
    assert "Нет загруженных документов" not in reply


def test_unknown_course_is_reported() -> None:
    scope = {"missing_courses": ["Химия"]}  # дисциплина из запроса, которой нет в документах
    graph, _ = make_graph(
        Models(orchestrator=[decision(kind="check", scope=scope)]), {"a.md": ("rpd", rpd())}
    )

    state, _ = run(graph, make_input("r1", "проверь химию", file_paths=["a.md"]))

    assert "Не найдены дисциплины: Химия." in state.values["reply"]


# --- протокол проверки файлом --------------------------------------------------------------


def test_protocol_file_is_saved_and_returned(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    graph, _ = make_graph(
        Models(), {"ok.md": ("rpd", rpd()), "bad.md": ("rpd", broken_rpd())}, storage=storage
    )

    values = check(graph, "r1", ["ok.md", "bad.md"])

    (file,) = values["files"]
    assert file["kind"] == "check_protocol"
    assert (
        f"Полный протокол со всеми нарушениями и цитатами: «{file['filename']}»" in values["reply"]
    )
    text = [
        p.text
        for p in Document(io.BytesIO(asyncio.run(storage.read(file["storage_key"])))).paragraphs
    ]
    assert "Итог: есть нарушения. Без замечаний: 9, нарушений: 5, не проверено: 34." in text
    assert "Физика (ошибок: 5)" in text  # нарушения по документам
    assert "коды не из перечня: УК-12" in text


def test_chat_shows_most_important_findings_first() -> None:
    def finding(i: int, severity: str) -> Finding:
        return Finding(
            id=f"f{i}",
            session_id="s1",
            run_id="r1",
            rule_id=f"rule{i}",
            rule_title=f"Правило {i}",
            rule_set="cross_check",
            severity=severity,
            fixable=True,
            document_id="d1",
            course="Физика",
            status="violation",
            message="нарушено",
        )

    findings = [finding(i, "warning") for i in range(8)] + [
        finding(i, "error") for i in range(8, 16)
    ]
    summary = {
        "rules": 16,
        "rule_sets": [],
        "passed": 0,
        "finding_ids": [],
        "missing": [],
        "unknown_courses": [],
        "catalog_version": "v1",
        "editions": [],
    }

    reply = check_report(summary, findings, "Протокол.docx")

    assert "Самые важные нарушения:\n1. Физика. Правило 10 (ошибка)" in reply  # ошибки первыми
    assert f"…и ещё нарушений: {len(findings) - MAX_CHAT_FINDINGS}." in reply
    assert "Правило 7 (предупреждение)" not in reply

    long = findings[8].model_copy(update={"message": "\n".join(f"- строка {i}" for i in range(9))})
    reply = check_report(summary, [long])
    assert "- строка 4" in reply and "- строка 5" not in reply
    assert "…и ещё строк: 4 (в полном протоколе)" in reply


def test_document_rule_without_plan_is_not_checked() -> None:
    from edu_doc_agent.orchestration.checker import check_rule
    from edu_doc_agent.validation.catalog import load_rule_file
    from edu_doc_agent.validation.units import CourseDocuments, Unit

    (rule,) = [
        r
        for r in load_rule_file(Path(__file__).parents[3] / "rules" / "cross_check.yaml")
        if r.id == "cross-check-contradictions"
    ]
    unit = Unit("d1", "Физика", CourseDocuments(rpd(), None, (), ""), ("a.md",))

    verdict = asyncio.run(check_rule(None, None, rule, unit, {}))

    assert verdict.outcome.status == "not_checked"
    assert verdict.outcome.message == "нет учебного плана"
