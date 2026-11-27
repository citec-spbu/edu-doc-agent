"""Проверки компетенций: коды дисциплины, перечень и формулировки компетенций программы."""

import difflib
import re

from edu_doc_agent.extraction.codes import canonical_code
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.models import RPDExtraction
from edu_doc_agent.validation.checks import (
    CheckOutcome,
    not_checked,
    passed,
    register,
    violation,
)
from edu_doc_agent.validation.units import ProgramDocuments


@register("codes_in_list", fields=("competences",))
def codes_in_list(doc: RPDExtraction, *, prefixes: list[str], codes: list[str]) -> CheckOutcome:
    """Все коды компетенций документа с префиксами `prefixes` есть в списке `codes`.

    Коды сравниваются в канонической форме: `УК - 01` и `УК-1` совпадают.
    """
    if not doc.competences:
        return not_checked("не удалось прочитать компетенции дисциплины")
    allowed = {canonical_code(c) for c in codes}
    wrong = [
        c
        for c in doc.competences
        if canonical_code(c.code).split("-")[0] in prefixes
        and canonical_code(c.code) not in allowed
    ]
    if not wrong:
        return passed()
    listed = ", ".join(dict.fromkeys(c.code for c in wrong))
    return violation(f"коды не из перечня: {listed}", [c.source for c in wrong])


def _program_lists(program: ProgramDocuments) -> dict[str, list[tuple[str, str]]]:
    """Перечни компетенций программы: учебный план (1.1) и общая характеристика."""
    lists = {}
    if program.plan is not None and program.plan.competences:
        lists["учебный план"] = [(c.code, c.name or "") for c in program.plan.competences]
    if program.characteristic is not None and program.characteristic.competences:
        lists["характеристика"] = [(c.code, c.name) for c in program.characteristic.competences]
    return lists


@register("program_codes_present", fields=("competences",))
def program_codes_present(
    program: ProgramDocuments, *, codes: list[str] = (), prefixes: list[str] = ()
) -> CheckOutcome:
    """В перечнях компетенций учебного плана и характеристики есть все коды `codes` и хотя бы
    один код с каждым префиксом `prefixes` (группа компетенций)."""
    lists = _program_lists(program)
    if not lists:
        return not_checked("нет перечня компетенций ни в учебном плане, ни в характеристике")
    lines = []
    for name, items in lists.items():
        present = {canonical_code(code) for code, _ in items}
        missing = [c for c in codes if canonical_code(c) not in present]
        groups = [p for p in prefixes if not any(c.split("-")[0] == p for c in present)]
        if missing:
            lines.append(f"{name}: нет {', '.join(missing)}")
        if groups:
            lines.append(f"{name}: нет компетенций групп {', '.join(groups)}")
    if not lines:
        return passed()
    return violation("; ".join(lines), [])


# сокращения, которые не меняют формулировку: «в т. ч.» и «в том числе» — одно и то же
_ABBREVIATIONS = (
    (re.compile(r"\bт\.\s*ч\.?"), "том числе"),
    (re.compile(r"\bт\.\s*е\.?"), "то есть"),
    (re.compile(r"\bи\s+т\.\s*д\.?"), "и так далее"),
)


def _words(text: str) -> list[str]:
    text = normalize(text)
    for pattern, full in _ABBREVIATIONS:
        text = pattern.sub(full, text)
    return re.findall(r"\w+", text)


def same_wording(text: str, reference: str) -> bool:
    """Формулировки совпадают без учёта регистра, знаков препинания и сокращений."""
    return _words(text) == _words(reference)


def _difference(name: str, reference: str) -> str:
    """Отличающиеся слова формулировки: «происхождения» → «производства»."""
    a, b = _words(reference), _words(name)
    parts = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op == "equal":
            continue
        was, now = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        parts.append(
            f"«{was}» → «{now}»" if was and now else f"нет «{was}»" if was else f"лишнее «{now}»"
        )
    return ", ".join(parts)


@register("competence_wording_matches", fields=("competences",))
def competence_wording_matches(
    program: ProgramDocuments, *, wordings: dict[str, str]
) -> CheckOutcome:
    """Формулировки компетенций `wordings` (код → текст норматива) в учебном плане и
    характеристике совпадают с нормативом. Регистр, знаки препинания и сокращения вроде
    «в т. ч.» не учитываются."""
    lists = _program_lists(program)
    if not lists:
        return not_checked("нет перечня компетенций ни в учебном плане, ни в характеристике")
    reference = {canonical_code(code): text for code, text in wordings.items()}
    lines = []
    for name, items in lists.items():
        for code, text in items:
            expected = reference.get(canonical_code(code))
            if expected is None or not text or _words(text) == _words(expected):
                continue
            lines.append(f"{name}, {code}: {_difference(text, expected)}")
    if not lines:
        return passed()
    return violation("\n".join(f"- {line}" for line in lines), [])
