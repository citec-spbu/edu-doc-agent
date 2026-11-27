"""Состав документов программы: разделы РПД и РПП, сведения общей характеристики, форма
обучения, блоки учебного плана."""

import re

from edu_doc_agent.validation.checks import (
    CheckOutcome,
    not_checked,
    passed,
    register,
    violation,
)
from edu_doc_agent.validation.units import CourseDocuments, ProgramDocuments

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


@register("sections_present")
def sections_present(unit: CourseDocuments, *, sections: dict[str, str]) -> CheckOutcome:
    """В документе дисциплины есть разделы: `sections` — что требуется → регулярное выражение
    для заголовка (без учёта регистра)."""
    headings = _HEADING_RE.findall(unit.text)
    if not headings:
        return not_checked("не удалось прочитать заголовки разделов документа")
    missing = [
        name
        for name, pattern in sections.items()
        if not any(re.search(pattern, h, re.IGNORECASE) for h in headings)
    ]
    if not missing:
        return passed()
    return violation(f"нет разделов: {', '.join(missing)}", [])


@register("characteristic_fields_present")
def characteristic_fields_present(
    program: ProgramDocuments, *, fields: dict[str, str]
) -> CheckOutcome:
    """В общей характеристике заполнены сведения: `fields` — поле данных → что это."""
    characteristic = program.characteristic
    if characteristic is None:
        return not_checked("нет общей характеристики")
    missing = [
        title
        for name, title in fields.items()
        if getattr(characteristic, name, None) in (None, "", [])
    ]
    if not missing:
        return passed()
    return violation(f"в общей характеристике не указано: {', '.join(missing)}", [])


@register("study_form_allowed", fields=("study_form",))
def study_form_allowed(program: ProgramDocuments, *, forms: list[str]) -> CheckOutcome:
    """Форма обучения в учебном плане и общей характеристике — из списка `forms`."""
    values = {
        name: doc.study_form
        for name, doc in (
            ("учебный план", program.plan),
            ("характеристика", program.characteristic),
        )
        if doc is not None and doc.study_form
    }
    if not values:
        return not_checked("форма обучения не указана ни в учебном плане, ни в характеристике")
    allowed = {f.casefold() for f in forms}
    wrong = [
        f"{name}: «{part.strip()}»"
        for name, value in values.items()
        for part in re.split(r",|;|\bи\b", value)
        if part.strip() and part.strip().casefold() not in allowed
    ]
    if not wrong:
        return passed()
    return violation(
        f"форма обучения не из допустимых ({', '.join(forms)}): {'; '.join(wrong)}", []
    )


@register("plan_blocks_present", fields=("entries", "final_attestation"))
def plan_blocks_present(program: ProgramDocuments, *, blocks: dict[str, str]) -> CheckOutcome:
    """В учебном плане есть строки каждого блока: `blocks` — название → начало шифра блока
    («Блок.1»). Строки ГИА учебного плана входят в свой блок."""
    if program.plan is None:
        return not_checked("нет учебного плана")
    codes = [e.block for e in program.plan.entries if e.block]
    codes += [a.block for a in program.plan.final_attestation if a.block]
    missing = [
        name for name, prefix in blocks.items() if not any(c.startswith(prefix) for c in codes)
    ]
    if not missing:
        return passed()
    return violation(f"в учебном плане нет блоков: {', '.join(missing)}", [])
