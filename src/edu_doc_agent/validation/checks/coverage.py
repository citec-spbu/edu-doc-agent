"""Компетенции программы между документами: покрытие дисциплинами учебного плана, матрица
компетенций против плана, повторы кодов, несколько документов одного вида.

Коды сравниваются в канонической форме (`extraction/codes.py`): `ПКП-19-МО-1` в матрице и
`ПКП-МО-1` в плане — один код.
"""

from collections import Counter

from edu_doc_agent.extraction.codes import canonical_code
from edu_doc_agent.extraction.curriculum import PlanEntry
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.validation.checks import (
    CheckOutcome,
    not_checked,
    passed,
    register,
    violation,
)
from edu_doc_agent.validation.units import ProgramDocuments

_KIND_TITLES = {
    "plan": "учебных плана",
    "characteristic": "общих характеристики",
    "ckmv": "ЦКМВ",
    "matrix": "матрицы компетенций",
}


def _codes(codes: list[str]) -> set[str]:
    return {canonical_code(c) for c in codes if c.strip()}


@register("competences_covered", fields=("entries", "competences", "final_attestation"))
def competences_covered(program: ProgramDocuments) -> CheckOutcome:
    """Каждую компетенцию программы (таблица 1.1 учебного плана) формирует хотя бы одна
    дисциплина, практика или ГИА учебного плана."""
    plan = program.plan
    if plan is None:
        return not_checked("нет учебного плана")
    if not plan.competences:
        return not_checked("не удалось прочитать перечень компетенций учебного плана (1.1)")
    formed = _codes([c for e in plan.entries for c in e.competences])
    formed |= _codes([c for a in plan.final_attestation for c in a.competences])
    missing = [c.code for c in plan.competences if canonical_code(c.code) not in formed]
    if not missing:
        return passed()
    return violation(
        f"не формируются ни одной строкой учебного плана: {', '.join(missing)}",
        [c.source for c in plan.competences if c.code in missing],
    )


def _plan_rows(
    entries: list[PlanEntry], discipline: str, semester: int | None, profile: str | None
):
    name = normalize(discipline)
    return [
        e
        for e in entries
        if normalize(e.name).startswith(name)
        and (semester is None or e.semester == semester)
        # строка плана без профиля — общая для всех профилей
        and (
            profile in (None, "база")
            or e.profile is None
            or normalize(profile) in normalize(e.profile)
        )
    ]


_GIA = "государственная итоговая аттестация"


@register("matrix_matches_plan", fields=("entries", "competences", "final_attestation"))
def matrix_matches_plan(program: ProgramDocuments) -> CheckOutcome:
    """Компетенции дисциплины в матрице компетенций совпадают с её строками учебного плана
    (по названию, семестру и профилю), ГИА — с разделом 3 плана. Строки матрицы, которых нет
    в плане под тем же названием, сравнить нельзя: они перечисляются, а без расхождений
    правило не проверено, а не выполнено."""
    plan = program.plan
    if plan is None:
        return not_checked("нет учебного плана")
    if program.matrix is None:
        return not_checked("нет матрицы компетенций")
    lines: dict[str, None] = {}
    sources = []
    unmatched: dict[str, None] = {}
    for row in program.matrix.rows:
        if normalize(row.discipline).startswith(_GIA):
            in_plan = _codes([c for a in plan.final_attestation for c in a.competences])
        else:
            rows = _plan_rows(plan.entries, row.discipline, row.semester, row.profile)
            if not rows:
                unmatched[f"«{row.discipline}» (семестр {row.semester})"] = None
                continue
            in_plan = _codes([c for e in rows for c in e.competences])
        in_matrix = _codes(row.competences)
        if in_matrix == in_plan:
            continue
        parts = []
        if only_plan := sorted(in_plan - in_matrix):
            parts.append(f"только в плане: {', '.join(only_plan)}")
        if only_matrix := sorted(in_matrix - in_plan):
            parts.append(f"только в матрице: {', '.join(only_matrix)}")
        lines[f"«{row.discipline}» (семестр {row.semester}): {'; '.join(parts)}"] = None
        sources.append(row.source)
    skipped = ""
    if unmatched:
        total = len(program.matrix.rows)
        skipped = (
            f"не сверены строки матрицы, которых нет в учебном плане под тем же названием "
            f"({len(unmatched)} из {total}): {', '.join(unmatched)}"
        )
    if lines:
        found = "\n".join(f"- {line}" for line in lines)
        return violation(f"{found}\n{skipped}" if skipped else found, sources)
    return not_checked(skipped) if skipped else passed()


@register("no_duplicate_codes", fields=("competences",))
def no_duplicate_codes(program: ProgramDocuments) -> CheckOutcome:
    """В перечне компетенций учебного плана, характеристики, ЦКМВ и каждой РПД код не
    повторяется."""
    lists = {}
    if program.plan is not None:
        lists[program.plan.file] = [c.code for c in program.plan.competences]
    if program.characteristic is not None:
        lists[program.characteristic.file] = [c.code for c in program.characteristic.competences]
    if program.ckmv is not None:
        lists[program.ckmv.file] = [c.code for c in program.ckmv.competences]
    for course in program.courses:
        lists[course.filename] = [c.code for c in course.data.competences]
    if not any(lists.values()):
        return not_checked("не удалось прочитать перечни компетенций")
    lines = []
    for file, codes in lists.items():
        counts = Counter(canonical_code(c) for c in codes)
        if repeated := [code for code, n in counts.items() if n > 1]:
            lines.append(f"{file}: повторяются {', '.join(repeated)}")
    if not lines:
        return passed()
    return violation("\n".join(f"- {line}" for line in lines), [])


@register("single_document_per_kind")
def single_document_per_kind(program: ProgramDocuments) -> CheckOutcome:
    """Учебный план, общая характеристика, ЦКМВ и матрица загружены по одному файлу: при
    нескольких проверяется только последний загруженный."""
    lines = [
        f"загружено {len(files)} {_KIND_TITLES[kind]}: {', '.join(files)}; "
        f"проверяется последний — {files[-1]}"
        for kind, files in program.files_by_kind.items()
        if len(files) > 1
    ]
    if not lines:
        return passed()
    return violation("\n".join(lines), [])
