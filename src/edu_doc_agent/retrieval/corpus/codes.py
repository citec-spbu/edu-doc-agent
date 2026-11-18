"""Коды компетенций и профстандартов в тексте нормативов.

Коды приводятся к той же канонической форме, что и в данных программы
(`edu_doc_agent.codes.canonical_code`, ADR-0005 п. 4): иначе `УК‑1` из ФГОС и `УК-1` из РПД
не совпадут при точном поиске. Здесь только поиск кодов: префиксы шире, чем в
`extraction.patterns`, — в ОС СПбГУ есть УКС, УКМ и УКО.
"""

import re

from edu_doc_agent.codes import canonical_code

COMPETENCY_PREFIX = r"(?:УКБ|УКС|УКМ|УКО|УНК|УК|ОПК|ПКА|ПКП|ПК)"
DASH = r"\s*[-‐‑‒–—―−]\s*"
# УК-1, ОПК-2, УКБ-3, ПКП-1-ППК-Р1, ПКП-УНК-03; индикатор (УК-1.2) обрезается до компетенции
COMPETENCY_CODE = re.compile(
    rf"(?<![0-9A-Za-zА-Яа-яЁё]){COMPETENCY_PREFIX}(?:{DASH}[0-9А-ЯЁ]+)+(?![0-9A-Za-zА-Яа-яЁё-])"
)
# Код профессионального стандарта: 06.001, 40.011
PROF_STANDARD_CODE = re.compile(r"(?<![\d.])\d{2}\.\d{3}(?![\d.])")


def normalize_code(code: str) -> str:
    """Каноническая форма кода (как в данных программы): `'УК – 01'` -> `'УК-1'`."""
    return canonical_code(re.sub(DASH, "-", code))


def find_competency_codes(text: str) -> list[str]:
    """Коды компетенций в тексте, канонические, без повторов, в порядке появления."""
    seen: dict[str, None] = {}
    for match in COMPETENCY_CODE.finditer(text):
        seen.setdefault(normalize_code(match.group(0)), None)
    return list(seen)
