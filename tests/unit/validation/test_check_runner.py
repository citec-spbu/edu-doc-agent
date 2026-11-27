"""Проверка кодом по единицам проверки: дисциплина, дисциплина с планом, программа."""

from rpd_samples import rpd

from edu_doc_agent.extraction.curriculum import CurriculumPlan, PlanEntry
from edu_doc_agent.extraction.source import SourceRef
from edu_doc_agent.validation.catalog import Rule
from edu_doc_agent.validation.runner import check_with_code
from edu_doc_agent.validation.units import CourseDoc, ProgramDocuments, plan_entries_for, units

SRC = SourceRef(file="plan.md", line=1)


def rule(**overrides) -> Rule:
    values = {
        "id": "credits",
        "title": "З.е. совпадают с таблицей",
        "rule_set": "cross_check",
        "applies_to": "course",
        "doc_kinds": ["discipline"],
        "method": "code",
        "check": "credits_match_workload_table",
        "severity": "error",
        "fixable": True,
    }
    return Rule.model_validate(values | overrides)


def program(*courses: CourseDoc, plan: CurriculumPlan | None = None) -> ProgramDocuments:
    return ProgramDocuments(courses=courses, plan=plan)


def course(document_id: str, **kwargs) -> CourseDoc:
    return CourseDoc(document_id, rpd(**kwargs), "rpd.md")


def run(rules, prog):
    return check_with_code(rules, prog, session_id="s1", run_id="r1")


def test_violation_becomes_finding() -> None:
    results = run([rule()], program(course("d1"), course("d2", title="Физика", credits=5)))

    assert results.passed_ids == ["credits:d1"]
    (finding,) = results.findings
    assert finding.id == "credits:d2"
    assert finding.course == "Физика"
    assert finding.status == "violation"
    assert finding.run_id == "r1"
    assert finding.message == "указано 5 з.е., в таблице (итого (очная)) 4 з.е."


def test_rule_applies_only_to_its_doc_kinds() -> None:
    results = run([rule()], program(course("d1", doc_kind="practice", credits=5)))

    assert results.results == []


def test_unknown_doc_kind_is_checked() -> None:
    results = run([rule()], program(course("d1", doc_kind=None, credits=5)))

    assert [f.id for f in results.findings] == ["credits:d1"]


def test_llm_rules_are_skipped_by_code() -> None:
    llm_rule = rule(method="llm", check=None, criterion="c", fields=["goals"])

    assert run([llm_rule], program(course("d1"))).results == []


def test_program_rule_has_one_unit() -> None:
    gia = rule(
        applies_to="program",
        doc_kinds=["curriculum"],
        check="final_attestation_credits_in_range",
        params={"minimum": 6, "maximum": 9},
    )

    (finding,) = run([gia], program(course("d1"))).findings

    assert finding.id == "credits:program"
    assert finding.document_id is None
    assert finding.status == "not_checked"
    assert finding.message == "нет учебного плана"


def test_only_selected_documents() -> None:
    prog = program(course("d1", credits=5), course("d2", credits=5))

    results = check_with_code([rule()], prog, session_id="s1", run_id="r1", document_ids={"d2"})

    assert [f.id for f in results.findings] == ["credits:d2"]


def test_plan_entries_by_code_then_title() -> None:
    plan = CurriculumPlan(
        file="plan.md",
        entries=[
            PlanEntry(name="Физика", code="001863", source=SRC),
            PlanEntry(name="Математический анализ I", code="001856", source=SRC),
        ],
    )

    by_code = plan_entries_for(rpd(title="Другое", discipline_code="001863"), plan)
    by_title = plan_entries_for(rpd(title="Математический анализ"), plan)

    assert [e.name for e in by_code] == ["Физика"]
    assert [e.name for e in by_title] == ["Математический анализ I"]


def test_documents_units_carry_plan_file() -> None:
    plan = CurriculumPlan(file="plan.md")
    (unit,) = units("documents", ["discipline"], program(course("d1"), plan=plan))

    assert unit.files == ("rpd.md", "plan.md")
    assert unit.data.plan is plan
