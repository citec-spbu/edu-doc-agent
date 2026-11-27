"""Проверки РПП: тип практики."""

import re

from edu_doc_agent.extraction.models import RPDExtraction
from edu_doc_agent.validation.checks import (
    CheckOutcome,
    not_checked,
    passed,
    register,
    violation,
)

_SEPARATORS_RE = re.compile(r",|;|\bи\b")


@register("practice_type_allowed", fields=("practice",))
def practice_type_allowed(doc: RPDExtraction, *, types: dict[str, str]) -> CheckOutcome:
    """Каждый тип практики из раздела «Тип практики» — один из типов ФГОС: `types` — тип →
    регулярное выражение для его формулировок (без учёта регистра). Несколько типов в
    документе перечислены через запятую. Название практики не учитывается."""
    if doc.practice is None or not doc.practice.type:
        return not_checked("не удалось прочитать тип практики")
    parts = [p.strip(" .") for p in _SEPARATORS_RE.split(doc.practice.type)]
    wrong = [
        p
        for p in parts
        if p and not any(re.search(pattern, p, re.IGNORECASE) for pattern in types.values())
    ]
    if not wrong:
        return passed()
    allowed = ", ".join(types)
    return violation(
        f"тип практики «{', '.join(wrong)}» не из ФГОС; допустимы: {allowed}",
        [doc.practice.source],
    )
