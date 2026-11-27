"""Функции проверки для правил method: code."""

from pathlib import Path

import pytest
from rpd_samples import row, rpd

from edu_doc_agent.extraction.curriculum import CurriculumPlan, FinalAttestation, PlanEntry
from edu_doc_agent.extraction.models import PracticeInfo
from edu_doc_agent.extraction.source import SourceRef
from edu_doc_agent.validation.catalog import load_catalog
from edu_doc_agent.validation.checks import CHECKS, run_check
from edu_doc_agent.validation.checks.competences import codes_in_list
from edu_doc_agent.validation.checks.documents import competences_match_plan, credits_match_plan
from edu_doc_agent.validation.checks.practice import practice_type_allowed
from edu_doc_agent.validation.checks.program import (
    block_credits_at_least,
    disciplines_present,
    final_attestation_credits_in_range,
)
from edu_doc_agent.validation.checks.workload import (
    credits_match_workload_table,
    hours_match_credits,
    semesters_sum_to_total,
)
from edu_doc_agent.validation.units import CourseDocuments, ProgramDocuments

RULES_DIR = Path(__file__).parents[3] / "rules"


def test_hours_match_credits() -> None:
    assert hours_match_credits(rpd(), hours_per_credit=36).status == "passed"

    doc = rpd(workload_table=[row(1, 3, 36, 36), row(None, 3, 36, 36)])
    outcome = hours_match_credits(doc, hours_per_credit=36)

    assert outcome.status == "violation"
    assert outcome.message == (
        "семестр 1 (очная): 72 ч при 3 з.е., нужно 108 ч; "
        "итого (очная): 72 ч при 3 з.е., нужно 108 ч"
    )
    assert len(outcome.sources) == 2


def test_hours_match_credits_without_table() -> None:
    outcome = hours_match_credits(rpd(workload_table=[]), hours_per_credit=36)

    assert outcome.status == "not_checked"
    assert "не удалось прочитать" in outcome.message


def test_semesters_sum_to_total() -> None:
    assert semesters_sum_to_total(rpd()).status == "passed"

    doc = rpd(workload_table=[row(1, 2, 36, 36), row(2, 1, 18, 18), row(None, 4, 72, 72)])
    outcome = semesters_sum_to_total(doc)

    assert outcome.status == "violation"
    assert outcome.message == (
        "итого (очная): 144 ч, по семестрам 108 ч; итого (очная): 4 з.е., по семестрам 3 з.е."
    )


def test_semesters_sum_per_study_form() -> None:
    """У каждой формы обучения своя строка ИТОГО."""
    table = [
        row(1, 4, 72, 72),
        row(None, 4, 72, 72),
        row(1, 4, 20, 124, study_form="заочная"),
        row(None, 4, 20, 124, study_form="заочная"),
    ]

    assert semesters_sum_to_total(rpd(workload_table=table)).status == "passed"


def test_semesters_sum_needs_total_row() -> None:
    two = semesters_sum_to_total(rpd(workload_table=[row(1, 4, 72, 72), row(2, 2, 36, 36)]))
    single = semesters_sum_to_total(rpd(workload_table=[row(1, 4, 72, 72)]))

    assert two.status == "not_checked"
    assert single.status == "passed"  # один семестр без ИТОГО: складывать нечего


def test_credits_match_single_semester_without_total() -> None:
    table = [row(5, 4, 72, 72)]

    assert credits_match_workload_table(rpd(credits=4, workload_table=table)).status == "passed"
    assert credits_match_workload_table(rpd(credits=3, workload_table=table)).status == "violation"


def test_credits_match_workload_table() -> None:
    assert credits_match_workload_table(rpd()).status == "passed"

    outcome = credits_match_workload_table(rpd(credits=5))

    assert outcome.status == "violation"
    assert outcome.message == "указано 5 з.е., в таблице (итого (очная)) 4 з.е."
    assert credits_match_workload_table(rpd(credits=None)).status == "not_checked"


def test_codes_in_list() -> None:
    params = {"prefixes": ["УК"], "codes": ["УК-1", "УК-2"]}

    assert codes_in_list(rpd(competences=["УК-1", "ПКП-3"]), **params).status == "passed"

    outcome = codes_in_list(rpd(competences=["УК-1", "УК-12", "УК-12"]), **params)

    assert outcome.status == "violation"
    assert outcome.message == "коды не из перечня: УК-12"


def test_codes_in_list_canonical_form() -> None:
    outcome = codes_in_list(rpd(competences=["УК - 01"]), prefixes=["УК"], codes=["УК-1"])

    assert outcome.status == "passed"


def test_codes_in_list_without_competences() -> None:
    outcome = codes_in_list(rpd(competences=[]), prefixes=["УК"], codes=["УК-1"])

    assert outcome.status == "not_checked"


def test_unknown_check() -> None:
    outcome = run_check("no_such_check", rpd(), {})

    assert outcome.status == "not_checked"


def test_repository_rules_load_and_use_known_checks() -> None:
    rules = load_catalog(RULES_DIR)

    assert rules
    for rule in rules:
        if rule.method == "code":
            assert rule.check in CHECKS, rule.id


# --- сверка с учебным планом и проверки программы -------------------------------------------

SRC = SourceRef(file="plan.md", line=1)


def entry(code: str, credits: float, competences: list[str], **kw) -> PlanEntry:
    return PlanEntry(
        name=kw.pop("name", "Физика"),
        code=code,
        credits=credits,
        competences=competences,
        source=SRC,
        **kw,
    )


def course_docs(course, *entries: PlanEntry) -> CourseDocuments:
    plan = CurriculumPlan(file="plan.md", entries=list(entries))
    return CourseDocuments(course, plan, tuple(entries))


def test_credits_match_plan_per_profile() -> None:
    course = rpd(credits=6)
    split = course_docs(course, entry("1", 3, [], semester=1), entry("1", 3, [], semester=2))
    profiles = course_docs(
        course,
        entry("1", 4, [], profile="Профиль ТОП"),
        entry("1", 6, [], profile="Профиль СТАНДАРТ"),
    )

    assert credits_match_plan(split).status == "passed"
    assert credits_match_plan(profiles).status == "passed"
    outcome = credits_match_plan(course_docs(rpd(credits=5), entry("1", 4, [])))
    assert outcome.message == "в РПД 5 з.е., в учебном плане 4 з.е."
    assert credits_match_plan(CourseDocuments(course, None, ())).message == "нет учебного плана"


def test_competences_match_closest_profile() -> None:
    course = rpd(competences=["УК-1", "ОПК-2"])
    unit = course_docs(
        course,
        entry("1", 4, ["УК-1"], profile="Профиль ТОП"),
        entry("1", 4, ["УК-1", "ОПК-2"], profile="Профиль СТАНДАРТ"),
    )

    assert competences_match_plan(unit).status == "passed"
    outcome = competences_match_plan(course_docs(course, entry("1", 4, ["УК-1", "УК-2"])))
    assert outcome.message == "только в РПД: ОПК-2; только в учебном плане: УК-2"


def program(**plan_fields) -> ProgramDocuments:
    return ProgramDocuments(plan=CurriculumPlan(file="plan.md", **plan_fields))


def test_final_attestation_and_block_credits() -> None:
    gia = [FinalAttestation(credits=12, name="ВКР", source=SRC)]
    practice = [
        entry("2", 10, [], block="Блок.2..прки"),
        entry("3", 12, [], block="Блок.2..прки", profile="Профиль ТОП"),
        entry("4", 5, [], block="Блок.2..прки", profile="Профиль СТАНДАРТ"),
    ]

    outcome = final_attestation_credits_in_range(
        program(final_attestation=gia), minimum=6, maximum=9
    )
    assert outcome.message == "ГИА 12 з.е., нужно 6–9 з.е."
    outcome = block_credits_at_least(program(entries=practice), block="Блок.2", minimum=20)
    assert outcome.message == "Профиль СТАНДАРТ: 15 з.е., нужно не менее 20"
    assert block_credits_at_least(ProgramDocuments(), block="Блок.2", minimum=1).status == (
        "not_checked"
    )


def test_disciplines_present() -> None:
    entries = [
        entry("1", 3, [], name="Философия (онлайн-курс)", block="Блок.1.дисц"),
        entry("2", 3, [], name="Английский язык", block="Блок.1.дисц"),
    ]
    params = {
        "block": "Блок.1",
        "disciplines": {
            "философия": "философ",
            "иностранный язык": "английск",
            "история": "истори",
        },
    }

    outcome = disciplines_present(program(entries=entries), **params)

    assert outcome.message == "в блоке Блок.1 нет дисциплин: история"


PRACTICE_TYPES = {
    "проектно-технологическая": "технологическ|проектн",
    "научно-исследовательская работа": "научно-исследовательск",
}


def practice(type_: str | None) -> object:
    info = PracticeInfo(
        kind="Учебная", type=type_, source=SourceRef(file="rpd.md", section="1.2", line=1)
    )
    return rpd(doc_kind="practice", practice=info)


@pytest.mark.parametrize(
    ("type_", "status"),
    [
        ("проектная", "passed"),
        ("научно-исследовательская практика, проектная практика", "passed"),
        ("технологическая (проектно-технологическая)", "passed"),
        ("кулинарная", "violation"),
        ("проектная, кулинарная", "violation"),
        (None, "not_checked"),
    ],
)
def test_practice_type_allowed(type_: str | None, status: str) -> None:
    outcome = practice_type_allowed(practice(type_), types=PRACTICE_TYPES)

    assert outcome.status == status
    if status == "violation":
        assert "«кулинарная»" in outcome.message
