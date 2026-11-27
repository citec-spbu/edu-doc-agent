"""Проверки программы целиком (`applies_to: program`): учебный план и документы ОП."""

import re
from collections import defaultdict

from edu_doc_agent.extraction.codes import check_competence_codes
from edu_doc_agent.extraction.crossdoc import check_program_consistency
from edu_doc_agent.extraction.curriculum import CurriculumPlan, PlanEntry
from edu_doc_agent.validation.checks import (
    CheckOutcome,
    not_checked,
    passed,
    register,
    violation,
)
from edu_doc_agent.validation.units import ProgramDocuments

_EPS = 0.01


def _fmt(value: float) -> str:
    return f"{value:g}"


def _plan(program: ProgramDocuments) -> CurriculumPlan | CheckOutcome:
    return program.plan if program.plan is not None else not_checked("нет учебного плана")


@register("final_attestation_credits_in_range", fields=("final_attestation",))
def final_attestation_credits_in_range(
    program: ProgramDocuments, *, minimum: float, maximum: float
) -> CheckOutcome:
    """Объём ГИА (раздел 3 учебного плана) в пределах `minimum`–`maximum` з.е."""
    plan = _plan(program)
    if isinstance(plan, CheckOutcome):
        return plan
    rows = [a for a in plan.final_attestation if a.credits is not None]
    if not rows:
        return not_checked("не удалось прочитать объём ГИА в учебном плане")
    total = sum(a.credits for a in rows)
    if minimum - _EPS <= total <= maximum + _EPS:
        return passed()
    return violation(
        f"ГИА {_fmt(total)} з.е., нужно {_fmt(minimum)}–{_fmt(maximum)} з.е.",
        [a.source for a in rows],
    )


def _by_profile(entries: list[PlanEntry]) -> dict[str, list[PlanEntry]]:
    common = [e for e in entries if e.profile is None]
    profiles: dict[str, list[PlanEntry]] = defaultdict(list)
    for e in entries:
        if e.profile is not None:
            profiles[e.profile].append(e)
    return {p: common + rows for p, rows in profiles.items()} or {"": common}


@register("block_credits_at_least", fields=("entries",))
def block_credits_at_least(
    program: ProgramDocuments, *, block: str, minimum: float
) -> CheckOutcome:
    """Объём блока плана (`block` — начало шифра, «Блок.2») не меньше `minimum` з.е. в каждом
    профиле. Общие строки плана входят в каждый профиль."""
    plan = _plan(program)
    if isinstance(plan, CheckOutcome):
        return plan
    entries = [e for e in plan.entries if e.block and e.block.startswith(block)]
    if not entries:
        return not_checked(f"в учебном плане нет строк блока {block}")
    wrong = []
    for profile, rows in _by_profile(entries).items():
        total = sum(e.credits or 0 for e in rows)
        if total < minimum - _EPS:
            wrong.append((profile, total, rows))
    if not wrong:
        return passed()
    message = "; ".join(
        f"{profile or 'план'}: {_fmt(total)} з.е., нужно не менее {_fmt(minimum)}"
        for profile, total, _ in wrong
    )
    return violation(message, [e.source for _, _, rows in wrong for e in rows])


@register("disciplines_present", fields=("entries",))
def disciplines_present(
    program: ProgramDocuments, *, block: str, disciplines: dict[str, str]
) -> CheckOutcome:
    """В блоке плана есть дисциплины по каждой теме: `disciplines` — тема → регулярное
    выражение для названия (без учёта регистра)."""
    plan = _plan(program)
    if isinstance(plan, CheckOutcome):
        return plan
    names = [
        f"{e.name} {e.group or ''}" for e in plan.entries if e.block and e.block.startswith(block)
    ]
    if not names:
        return not_checked(f"в учебном плане нет строк блока {block}")
    missing = [
        topic
        for topic, pattern in disciplines.items()
        if not any(re.search(pattern, n, re.IGNORECASE) for n in names)
    ]
    if not missing:
        return passed()
    return violation(f"в блоке {block} нет дисциплин: {', '.join(missing)}", [])


@register("plan_matches_characteristic", fields=("direction", "level", "study_form", "program"))
def plan_matches_characteristic(program: ProgramDocuments) -> CheckOutcome:
    """Реквизиты программы (направление, уровень, форма, срок, название) совпадают в учебном
    плане и общей характеристике. Сверку выполняет модуль извлечения (`extraction/crossdoc.py`)."""
    plan = _plan(program)
    if isinstance(plan, CheckOutcome):
        return plan
    if program.characteristic is None:
        return not_checked("нет общей характеристики")
    differences = check_program_consistency(plan, program.characteristic)
    if not differences:
        return passed()
    return violation("; ".join(differences), [])


@register("competence_names_match_plan", fields=("competences",))
def competence_names_match_plan(program: ProgramDocuments) -> CheckOutcome:
    """Коды и формулировки компетенций в характеристике, ЦКМВ и РПД совпадают с таблицей 1.1
    учебного плана. Сверку выполняет модуль извлечения (`extraction/codes.py`)."""
    plan = _plan(program)
    if isinstance(plan, CheckOutcome):
        return plan
    differences = check_competence_codes(
        plan,
        program.ckmv,
        [c.data for c in program.courses],
        program.characteristic,
    )
    if not differences:
        return passed()
    return violation("\n".join(f"- {d}" for d in differences), [])
