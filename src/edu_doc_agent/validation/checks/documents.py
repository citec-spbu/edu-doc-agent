"""Сверка РПД с учебным планом (`applies_to: documents`)."""

from collections import defaultdict

from edu_doc_agent.extraction.codes import canonical_code
from edu_doc_agent.extraction.curriculum import PlanEntry
from edu_doc_agent.validation.checks import (
    CheckOutcome,
    not_checked,
    passed,
    register,
    violation,
)
from edu_doc_agent.validation.units import CourseDocuments

_EPS = 0.01


def _fmt(value: float) -> str:
    return f"{value:g}"


def _entries(unit: CourseDocuments) -> tuple[PlanEntry, ...] | CheckOutcome:
    if unit.plan is None:
        return not_checked("нет учебного плана")
    if not unit.plan_entries:
        return not_checked("дисциплины нет в учебном плане (по коду и названию)")
    return unit.plan_entries


def _by_profile(entries: tuple[PlanEntry, ...]) -> dict[str, list[PlanEntry]]:
    """Строки плана по профилю: общие строки (без профиля) входят в каждый профиль."""
    common = [e for e in entries if e.profile is None]
    profiles: dict[str, list[PlanEntry]] = defaultdict(list)
    for e in entries:
        if e.profile is not None:
            profiles[e.profile].append(e)
    if not profiles:
        return {"": common}
    return {p: common + rows for p, rows in profiles.items()}


@register("credits_match_plan", fields=("credits",))
def credits_match_plan(unit: CourseDocuments) -> CheckOutcome:
    """З.е. дисциплины в РПД равны сумме з.е. по семестрам в учебном плане (хотя бы для одного
    профиля, если дисциплина есть в нескольких)."""
    entries = _entries(unit)
    if isinstance(entries, CheckOutcome):
        return entries
    if unit.course.credits is None:
        return not_checked("не удалось прочитать трудоёмкость дисциплины в з.е.")
    sums = {
        profile: sum(e.credits or 0 for e in rows) for profile, rows in _by_profile(entries).items()
    }
    if any(abs(total - unit.course.credits) <= _EPS for total in sums.values()):
        return passed()
    in_plan = "; ".join(
        f"{profile}: {_fmt(total)} з.е." if profile else f"{_fmt(total)} з.е."
        for profile, total in sums.items()
    )
    return violation(
        f"в РПД {_fmt(unit.course.credits)} з.е., в учебном плане {in_plan}",
        [e.source for e in entries],
    )


@register("competences_match_plan", fields=("competences",))
def competences_match_plan(unit: CourseDocuments) -> CheckOutcome:
    """Компетенции дисциплины в РПД совпадают с компетенциями её строк в учебном плане.

    Если дисциплина есть в нескольких профилях, РПД сравнивается с ближайшим профилем."""
    entries = _entries(unit)
    if isinstance(entries, CheckOutcome):
        return entries
    if not unit.course.competences:
        return not_checked("не удалось прочитать компетенции дисциплины")
    in_rpd = {canonical_code(c.code): c for c in unit.course.competences}
    candidates = []
    for profile, rows in _by_profile(entries).items():
        in_plan = {canonical_code(code) for e in rows for code in e.competences}
        if in_plan:
            only_rpd = sorted(set(in_rpd) - in_plan)
            only_plan = sorted(in_plan - set(in_rpd))
            candidates.append((len(only_rpd) + len(only_plan), profile, only_rpd, only_plan, rows))
    if not candidates:
        return not_checked("в учебном плане у дисциплины не указаны компетенции")
    _, profile, only_rpd, only_plan, rows = min(candidates, key=lambda c: c[0])
    if not only_rpd and not only_plan:
        return passed()
    parts = []
    if only_rpd:
        parts.append(f"только в РПД: {', '.join(only_rpd)}")
    if only_plan:
        where = f"в учебном плане ({profile})" if profile else "в учебном плане"
        parts.append(f"только {where}: {', '.join(only_plan)}")
    sources = [in_rpd[c].source for c in only_rpd] + [e.source for e in rows]
    return violation("; ".join(parts), sources)
