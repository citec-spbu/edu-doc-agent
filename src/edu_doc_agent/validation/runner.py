"""Проверка кодом: правила `method: code` по единицам проверки и замечания по результатам."""

from dataclasses import dataclass, field

from edu_doc_agent.validation.catalog import Rule
from edu_doc_agent.validation.checks import CheckOutcome, run_check
from edu_doc_agent.validation.findings import Evidence, Finding, finding_id
from edu_doc_agent.validation.units import ProgramDocuments, Unit, units


@dataclass
class RuleResult:
    """Результат правила по одной единице проверки."""

    rule_id: str
    unit_id: str  # ID замечания: правило и единица
    finding: Finding | None  # None: правило выполнено


@dataclass
class CheckResults:
    results: list[RuleResult] = field(default_factory=list)

    @property
    def findings(self) -> list[Finding]:
        return [r.finding for r in self.results if r.finding is not None]

    @property
    def passed_ids(self) -> list[str]:
        return [r.unit_id for r in self.results if r.finding is None]


def make_result(
    rule: Rule,
    unit: Unit,
    outcome: CheckOutcome,
    *,
    session_id: str,
    run_id: str,
    evidence: list[Evidence] | None = None,
    needs_human: bool = False,
    opinions: list[str] | None = None,
) -> RuleResult:
    fid = finding_id(rule.id, unit.document_id)
    if outcome.status == "passed":
        return RuleResult(rule.id, fid, None)
    finding = Finding(
        id=fid,
        session_id=session_id,
        run_id=run_id,
        rule_id=rule.id,
        rule_title=rule.title,
        rule_set=rule.rule_set,
        severity=rule.severity,
        fixable=rule.fixable,
        method=rule.method,
        document_id=unit.document_id,
        course=unit.course,
        status=outcome.status,
        message=outcome.message,
        sources=list(outcome.sources),
        evidence=evidence or [],
        requirement=rule.requirement,
        needs_human=needs_human,
        opinions=opinions or [],
    )
    return RuleResult(rule.id, fid, finding)


def check_with_code(
    rules: list[Rule],
    program: ProgramDocuments,
    *,
    session_id: str,
    run_id: str,
    document_ids: set[str] | None = None,
) -> CheckResults:
    """Правила `method: code` по всем единицам; при `document_ids` — только по этим документам
    (правила программы проверяются всегда)."""
    results = CheckResults()
    for rule in rules:
        if rule.method != "code":
            continue
        assert rule.check is not None
        for unit in units(rule.applies_to, rule.doc_kinds, program):
            if document_ids is not None and unit.document_id not in (None, *document_ids):
                continue
            outcome = run_check(rule.check, unit.data, rule.params)
            results.results.append(
                make_result(rule, unit, outcome, session_id=session_id, run_id=run_id)
            )
    return results
