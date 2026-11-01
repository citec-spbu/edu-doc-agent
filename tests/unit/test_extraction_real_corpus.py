"""Сущности ADR-0002 на всём реальном корпусе РПД/РПП из data/ (docx → md текущим парсером).

Проверяем то, что должно выполняться для любого документа корпуса, а не точные значения.
Если сущность не нашлась, тест проверяет, что её действительно нет в документе (нет столбца
часов в 2.2, нет часов в 1.4 и т. д.), а не просто закрывает глаза.
"""

import re
from pathlib import Path

import pytest

from edu_doc_agent.extraction import RPDExtraction, extract_rpd_file
from tests.real_data import REAL_MD

pytestmark = pytest.mark.skipif(not REAL_MD, reason="нет реальных docx РПД в data/")

_CACHE: dict[Path, RPDExtraction] = {}


def _extract(path: Path) -> RPDExtraction:
    if path not in _CACHE:
        _CACHE[path] = extract_rpd_file(path)
    return _CACHE[path]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


DISCIPLINES = [
    p for p in REAL_MD if "".join(_text(p)[:1500].split()).upper().find("УЧЕБНОЙДИСЦИПЛИНЫ") >= 0
]
PRACTICES = [
    p
    for p in REAL_MD
    if p not in DISCIPLINES and "ПРАКТИКИ" in "".join(_text(p)[:600].split()).upper()
]

# В самих документах ИТОГО не сходится с семестрами (опечатки в часах): извлечение
# предупреждает об этом, и это верное поведение.
TOTAL_TYPOS = {
    "С5_076083_Глубокое обучение": "в ИТОГО 38 ч практических при 28 в обоих семестрах",
    "С7-8_Стандарт_058776_Интернет вещей": "в ИТОГО нет 26 ч семинаров семестра 7",
    "С7_Стандарт_001882_Интеллектуальные системы": "в ИТОГО 28 ч самостоятельной вместо 64",
}
# Известное ограничение: в docx строки шапки и тела разрезаны по сетке по-разному
# (ячейка тела перекрывает два столбца шапки), и привязка по номеру столбца неверна.
# Нужны ширины столбцов из tblGrid в парсере и привязка по перекрытию.
MISALIGNED_GRID = {"С3_001868_Математическая логика и теория алгоритмов"}


def _is_known(path: Path, known: dict | set) -> bool:
    return path.stem in known


def test_corpus_is_loaded() -> None:
    assert len(DISCIPLINES) >= 60 and len(PRACTICES) >= 5


@pytest.mark.parametrize("path", DISCIPLINES, ids=lambda p: p.stem)
def test_identification_is_complete(path: Path) -> None:
    r = _extract(path)
    assert r.title
    assert r.language, "язык обучения"
    assert r.discipline_code and re.fullmatch(r"\d{5,6}", r.discipline_code)


@pytest.mark.parametrize("path", DISCIPLINES, ids=lambda p: p.stem)
def test_workload_table_semester_rows(path: Path) -> None:
    """Таблица 2.1: у каждой дисциплины есть строки семестров, часы равны 36 × з.е."""
    if _is_known(path, MISALIGNED_GRID):
        pytest.xfail("шапка и тело таблицы 2.1 разрезаны по сетке по-разному")
    r = _extract(path)
    rows = [w for w in r.workload_table if not w.is_total]
    assert rows, "строки семестров таблицы 2.1"
    for row in rows:
        assert row.credits is not None, f"семестр {row.semester}: нет з.е."
        assert row.total_hours == row.credits * 36, f"семестр {row.semester}"
    assert sum(1 for w in r.warnings if w.startswith("трудоёмкость")) == (
        2 if path.stem == "С5_076083_Глубокое обучение" else int(_is_known(path, TOTAL_TYPOS))
    )


@pytest.mark.parametrize("path", DISCIPLINES, ids=lambda p: p.stem)
def test_semesters_of_control_forms_are_in_workload_table(path: Path) -> None:
    if _is_known(path, MISALIGNED_GRID):
        pytest.xfail("см. MISALIGNED_GRID")
    r = _extract(path)
    assert {f.semester for f in r.control_forms} <= {w.semester for w in r.workload_table}


@pytest.mark.parametrize("path", DISCIPLINES, ids=lambda p: p.stem)
def test_topic_hours_found_when_2_2_has_hours_column(path: Path) -> None:
    r = _extract(path)
    has_hours_column = bool(re.search(r"\|\s*Наименование темы[^\n]*Количество часов", _text(path)))
    assert bool(r.topic_hours) == has_hours_column


@pytest.mark.parametrize("path", DISCIPLINES, ids=lambda p: p.stem)
def test_active_forms_found_when_1_4_names_hours(path: Path) -> None:
    section = re.search(r"## 1\.4\..*?(?=\n#+ )", _text(path), re.S)
    named = bool(section and re.search(r"\d\s*(?:час|ч\b)", section.group(0)))
    assert bool(_extract(path).active_forms) == named


@pytest.mark.parametrize("path", DISCIPLINES, ids=lambda p: p.stem)
def test_developers_found_when_table_exists(path: Path) -> None:
    has_table = bool(re.search(r"\|\s*Фамилия, имя, отчество", _text(path)))
    assert bool(_extract(path).developers) == has_table


@pytest.mark.parametrize("path", DISCIPLINES, ids=lambda p: p.stem)
def test_all_competences_have_group(path: Path) -> None:
    assert all(c.group for c in _extract(path).competences)


@pytest.mark.parametrize("path", PRACTICES, ids=lambda p: p.stem)
def test_practice_entities(path: Path) -> None:
    """РПП: вид/тип/способ, профстандарты с функциями, компетенции по группам."""
    r = _extract(path)
    assert r.doc_kind == "practice"
    assert r.practice and r.practice.type and r.practice.method
    assert r.prof_standards and all(re.fullmatch(r"\d\d\.\d{3}", s.code) for s in r.prof_standards)
    assert r.standard_links
    assert r.competences and all(c.group for c in r.competences)
