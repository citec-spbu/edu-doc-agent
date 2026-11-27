"""Проверки трудоёмкости дисциплины по таблице «Трудоёмкость, объёмы учебной работы»."""

from collections import defaultdict

from edu_doc_agent.extraction.models import RPDExtraction, WorkloadRow
from edu_doc_agent.validation.checks import (
    CheckOutcome,
    not_checked,
    passed,
    register,
    violation,
)

_EPS = 0.01


def _fmt(value: float) -> str:
    return f"{value:g}"


def _label(row: WorkloadRow) -> str:
    where = "итого" if row.is_total else f"семестр {row.semester}"
    extra = ", ".join(x for x in (row.study_form, row.trajectory) if x)
    return f"{where} ({extra})" if extra else where


def _groups(doc: RPDExtraction) -> dict[tuple, list[WorkloadRow]]:
    """Строки таблицы по траектории и форме обучения: у каждой своя строка ИТОГО."""
    groups: dict[tuple, list[WorkloadRow]] = defaultdict(list)
    for row in doc.workload_table:
        groups[(row.trajectory, row.study_form)].append(row)
    return groups


@register("hours_match_credits", fields=("workload_table",))
def hours_match_credits(doc: RPDExtraction, *, hours_per_credit: float) -> CheckOutcome:
    """Часы каждой строки таблицы трудоёмкости равны з.е. × `hours_per_credit`."""
    rows = [r for r in doc.workload_table if r.credits is not None and r.total_hours > 0]
    if not rows:
        return not_checked("не удалось прочитать часы и з.е. в таблице трудоёмкости")
    wrong = [r for r in rows if abs(r.total_hours - r.credits * hours_per_credit) > _EPS]
    if not wrong:
        return passed()
    lines = [
        f"{_label(r)}: {_fmt(r.total_hours)} ч при {_fmt(r.credits)} з.е., "
        f"нужно {_fmt(r.credits * hours_per_credit)} ч"
        for r in wrong
    ]
    return violation("; ".join(lines), [r.source for r in wrong])


@register("semesters_sum_to_total", fields=("workload_table",))
def semesters_sum_to_total(doc: RPDExtraction) -> CheckOutcome:
    """З.е. и часы семестров дают в сумме строку ИТОГО."""
    lines, sources, compared = [], [], False
    for rows in _groups(doc).values():
        totals = [r for r in rows if r.is_total]
        semesters = [r for r in rows if not r.is_total]
        if not totals and len(semesters) == 1:
            compared = True  # один семестр без строки ИТОГО: складывать нечего
            continue
        if len(totals) != 1 or not semesters:
            continue
        (total,) = totals
        compared = True
        hours = sum(r.total_hours for r in semesters)
        if abs(hours - total.total_hours) > _EPS:
            lines.append(
                f"{_label(total)}: {_fmt(total.total_hours)} ч, по семестрам {_fmt(hours)} ч"
            )
            sources.append(total.source)
        if total.credits is not None and all(r.credits is not None for r in semesters):
            credits = sum(r.credits for r in semesters)
            if abs(credits - total.credits) > _EPS:
                lines.append(
                    f"{_label(total)}: {_fmt(total.credits)} з.е., "
                    f"по семестрам {_fmt(credits)} з.е."
                )
                sources.append(total.source)
    if not compared:
        return not_checked("в таблице трудоёмкости нет строк семестров со строкой ИТОГО")
    return violation("; ".join(lines), sources) if lines else passed()


@register("credits_match_workload_table", fields=("credits", "workload_table"))
def credits_match_workload_table(doc: RPDExtraction) -> CheckOutcome:
    """З.е. дисциплины совпадают со строкой ИТОГО таблицы трудоёмкости."""
    if doc.credits is None:
        return not_checked("не удалось прочитать трудоёмкость дисциплины в з.е.")
    totals = [r for r in doc.workload_table if r.is_total and r.credits is not None]
    if not totals:
        # без строки ИТОГО итог группы — её единственный семестр
        totals = [
            rows[0]
            for rows in _groups(doc).values()
            if len(rows) == 1 and rows[0].credits is not None
        ]
    if not totals:
        return not_checked("не удалось прочитать строку ИТОГО таблицы трудоёмкости")
    wrong = [r for r in totals if abs(r.credits - doc.credits) > _EPS]
    if not wrong:
        return passed()
    lines = [
        f"указано {_fmt(doc.credits)} з.е., в таблице ({_label(r)}) {_fmt(r.credits)} з.е."
        for r in wrong
    ]
    return violation("; ".join(lines), [r.source for r in wrong])
