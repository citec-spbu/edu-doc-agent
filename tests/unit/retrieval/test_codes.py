"""Поиск кодов в тексте нормативов (retrieval.corpus.codes)."""

import pytest

from edu_doc_agent.retrieval.corpus.codes import (
    PROF_STANDARD_CODE,
    find_competency_codes,
    normalize_code,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("УК-1", "УК-1"),
        ("УК – 01", "УК-1"),
        ("ОПК‑2", "ОПК-2"),  # неразрывный дефис
        ("УКМ-3", "УКМ-3"),
        ("УНК-03", "ПКП-УНК-3"),  # как в данных программы
    ],
)
def test_normalize_code_matches_program_data(raw: str, expected: str) -> None:
    assert normalize_code(raw) == expected


def test_find_competency_codes_unique_in_order() -> None:
    text = (
        "Формирует ОПК‑2 и УК-1; см. также ОПК-2, УКБ-3 и индикатор УК-1.2. Слово РУК-1 — не код."
    )
    assert find_competency_codes(text) == ["ОПК-2", "УК-1", "УКБ-3"]


def test_prof_standard_code() -> None:
    text = "Профстандарты 06.001 и 40.011; не коды: 01.03.02, 2.06.0011"
    assert PROF_STANDARD_CODE.findall(text) == ["06.001", "40.011"]
