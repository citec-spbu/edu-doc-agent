"""Матрица компетенций: синтетический xlsx, эталоны, сверка с планом, реальный файл."""

# ruff: noqa: E501  (строки таблиц Markdown)

import copy
import io
from pathlib import Path

import pytest
from openpyxl import Workbook

from edu_doc_agent.eval.matrix_gold import (
    MATRIX_GOLD,
    MATRIX_GOLD_FULL,
    THRESHOLD,
    load_matrix_gold,
    score_matrix,
)
from edu_doc_agent.extraction.crossdoc import check_documents, compare_matrix_with_plan
from edu_doc_agent.extraction.curriculum import extract_plan
from edu_doc_agent.extraction.matrix import extract_matrix
from tests.real_data import real_matrix_xlsx, real_plan_markdown

CODES = ["УК-1", "УНК-03", "ПКП-2-ППК-Р2"]


def _xlsx(rows: list[list], types: bool = True) -> io.BytesIO:
    wb = Workbook()
    ws = wb.active
    ws.title = "Матрица"
    ws.append(["Тип компетенции →", None, None, None, None, None, "УК", "УНК", "ПКП"])
    ws.append(["№ строки", "Семестр", "Профиль", "Часть", "ЗЕТ", "Дисциплина", *CODES])
    for r in rows:
        ws.append(r)
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return out


GOOD = [
    [1, 1, "база", "базовая", 5, "Алгебра", 1, 1, 0],
    [2, 1, "стандарт", "вариативная", 3, "Практикум", 0, 0, 1],
    [3, 2, "топ", "вариативная", 3, "Машинное обучение", 1, 0, 1],
    ["Итого (СУММ)", None, None, None, None, None, 2, 1, 2],
    ["Итого (Стандарт)", None, None, None, None, None, 1, 1, 1],
    ["Итого (ТОП)", None, None, None, None, None, 2, 1, 1],
]


def test_columns_rows_and_totals() -> None:
    m = extract_matrix(_xlsx(GOOD), "m.xlsx")
    assert [(c.code, c.type) for c in m.columns] == [
        ("УК-1", "УК"),
        ("УНК-03", "УНК"),
        ("ПКП-2-ППК-Р2", "ПКП"),
    ]
    assert [(r.number, r.semester, r.profile, r.part, r.credits) for r in m.rows] == [
        (1, 1, "база", "базовая", 5.0),
        (2, 1, "стандарт", "вариативная", 3.0),
        (3, 2, "топ", "вариативная", 3.0),
    ]
    assert [r.competences for r in m.rows] == [
        ["УК-1", "УНК-03"],
        ["ПКП-2-ППК-Р2"],
        ["УК-1", "ПКП-2-ППК-Р2"],
    ]
    assert m.totals["СУММ"] == {"УК-1": 2, "УНК-03": 1, "ПКП-2-ППК-Р2": 2}
    assert m.rows[0].source.line == 3 and m.sheet == "Матрица"
    assert m.warnings == []


def test_columns_are_found_by_header_not_position() -> None:
    wb = Workbook()
    ws = wb.active
    ws.append(["служебная строка"])
    ws.append([None, "ОПК", "ПКА"])
    ws.append(["Дисциплина", "ОПК-1", "ПКА-2"])
    ws.append(["Алгебра", 1, 0])
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    m = extract_matrix(out, "m.xlsx")
    assert [c.code for c in m.columns] == ["ОПК-1", "ПКА-2"]
    assert [(r.discipline, r.competences) for r in m.rows] == [("Алгебра", ["ОПК-1"])]
    assert m.rows[0].number == 1 and m.rows[0].semester is None


def test_inconsistent_totals_and_bad_values_are_reported() -> None:
    rows = copy.deepcopy(GOOD)
    rows[3][6] = 5  # СУММ по УК-1 неверна
    rows[1][6] = 2  # в данных не 0/1
    text = "\n".join(extract_matrix(_xlsx(rows), "m.xlsx").warnings)
    assert "итого (СУММ), УК-1: в таблице 5" in text
    assert "значение 2 вместо 0/1" in text


def test_type_mismatch_and_duplicate_columns() -> None:
    wb = Workbook()
    ws = wb.active
    ws.append([None, "УК", "ОПК"])
    ws.append(["Дисциплина", "УК-1", "УК-1"])
    ws.append(["Алгебра", 1, 1])
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    text = "\n".join(extract_matrix(out, "m.xlsx").warnings)
    assert "повтор кода" in text and "тип над кодом «ОПК»" in text


def test_missing_header_is_reported() -> None:
    wb = Workbook()
    wb.active.append(["просто", "текст"])
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    assert "не найдена шапка" in "\n".join(extract_matrix(out, "m.xlsx").warnings)


# --- сверка с планом -------------------------------------------------------------------------

PLAN_MD = """| Код компетенции | Наименование и (или) описание компетенции |
| --- | --- |
| УК-1 | Способен искать |
| ПКП-УНК-03 | Способен планировать |
| ПКП-ППК-Р2 | Способен проверять |

# Раздел 2

| Код Блока | Трудоёмкость | Код компетенции | Наименование | Виды контроля | Лекции | Промежуточная аттестация | Текущий контроль | Промежуточная аттестация |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Код Блока | Трудоёмкость | Код компетенции | Наименование | Виды контроля | Лекции | Промежуточная аттестация | Текущий контроль | Промежуточная аттестация |
| С01. Семестр 1 |  |  |  |  |  |  |  |  |
| Базовая часть периода обучения |  |  |  |  |  |  |  |  |
| Блок.1.дисц | 1 | УК-1, ПКП-УНК-03 | [000001] Алгебра<br>Algebra | зачёт | 34 | 2 | 0 | 0 |
| Вариативная часть периода обучения |  |  |  |  |  |  |  |  |
| Блок дисциплин по выбору |  |  |  |  |  |  |  |  |
| Профиль СТАНДАРТ |  |  |  |  |  |  |  |  |
| Блок.1.дисц | 1 | ПКП-ППК-Р2 | [000002] Практикум на ЭВМ<br>Workshop | зачёт | 34 | 2 | 0 | 0 |
| С02. Семестр 2 |  |  |  |  |  |  |  |  |
| Вариативная часть периода обучения |  |  |  |  |  |  |  |  |
| Блок дисциплин по выбору |  |  |  |  |  |  |  |  |
| Профиль ТОП ИТ ПРО |  |  |  |  |  |  |  |  |
| Блок.1.дисц | 1 | УК-1 | [000003] Машинное обучение (осн курс)<br>ML | экзамен | 34 | 2 | 0 | 0 |
"""


def test_compare_with_plan_equal_different_and_unmatched() -> None:
    plan = extract_plan(PLAN_MD, "plan.md")
    rows = copy.deepcopy(GOOD)
    rows[2][8] = 0  # у «Машинное обучение» в матрице нет ПКП-2-ППК-Р2, в плане нет тоже, но…
    rows[2][6] = 0  # …и нет УК-1, которая в плане есть
    rows.insert(3, [4, 2, "база", "базовая", 1, "Ин. Язык", 0, 0, 0])
    m = extract_matrix(_xlsx(rows), "m.xlsx")
    cmp = compare_matrix_with_plan(m, plan)
    assert (cmp.matched, cmp.equal) == (3, 2)
    assert [d.row.discipline for d in cmp.different] == ["Машинное обучение"]
    assert cmp.different[0].only_plan == ["УК-1"] and cmp.different[0].only_matrix == []
    assert [r.discipline for r in cmp.unmatched] == ["Ин. Язык"]
    assert cmp.columns_not_in_plan == [] and cmp.plan_competences_not_in_columns == []
    text = "\n".join(cmp.warnings)
    assert "только в плане: УК-1" in text and "«Ин. Язык»" in text


def test_compare_reports_columns_missing_on_either_side() -> None:
    plan = extract_plan(PLAN_MD, "plan.md")
    wb = Workbook()
    ws = wb.active
    ws.append([None, "УК", "ОПК"])
    ws.append(["Дисциплина", "УК-1", "ОПК-9"])
    ws.append(["Алгебра", 1, 0])
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    cmp = compare_matrix_with_plan(extract_matrix(out, "m.xlsx"), plan)
    assert cmp.columns_not_in_plan == ["ОПК-9"]
    assert cmp.plan_competences_not_in_columns == ["ПКП-ППК-Р2", "ПКП-УНК-3"]


def test_check_documents_collects_matrix_warnings() -> None:
    plan = extract_plan(PLAN_MD, "plan.md")
    m = extract_matrix(_xlsx(GOOD), "m.xlsx")
    warnings = check_documents(plan, matrix=m)
    # две строки матрицы совпадают с планом, у «Машинное обучение» в матрице лишняя компетенция
    assert len(warnings) == 1 and "Машинное обучение" in warnings[0]
    assert "только в матрице: ПКП-ППК-Р2" in warnings[0]
    assert check_documents(plan) == []


# --- эталоны и реальный файл -----------------------------------------------------------------

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MANUAL = load_matrix_gold(_FIXTURES / "matrix_gold.json")
FULL = load_matrix_gold(_FIXTURES / "matrix_gold_full.json")
needs_data = pytest.mark.skipif(
    real_matrix_xlsx() is None,
    reason="нет data/Матрица компетенций.zip",
)

needs_gold = pytest.mark.skipif(
    not MATRIX_GOLD_FULL.exists(), reason="нет локального эталона matrix_gold_full.json"
)


def _real():
    return extract_matrix(io.BytesIO(real_matrix_xlsx() or b""), "matrix.xlsx")


def test_gold_shapes_and_manual_agrees_with_full() -> None:
    assert FULL["totals_counts"] == {"columns": 2, "rows": 2, "ones": 3}
    assert len(FULL["rows"]) == 2 and len(FULL["columns"]) == 2
    full = {r["number"]: r for r in FULL["rows"]}
    for r in MANUAL["rows"]:
        assert r == full[r["number"]]
    for code, value in MANUAL["totals"]["СУММ"].items():
        assert FULL["totals"]["СУММ"][code] == value


@needs_data
def test_real_matrix_has_no_warnings() -> None:
    m = _real()
    assert (len(m.columns), len(m.rows)) == (49, 118) and m.warnings == []


@needs_data
@pytest.mark.parametrize("full", [False, True], ids=["manual", "full"])
def test_real_matrix_accuracy(full: bool) -> None:
    path = MATRIX_GOLD_FULL if full else MATRIX_GOLD
    if not path.exists():
        pytest.skip(f"нет локального эталона {path.name}")
    report = score_matrix(_real(), load_matrix_gold(path))
    assert report.passed(), report.summary()


@needs_data
@needs_gold
def test_real_matrix_cells_are_all_checked() -> None:
    report = score_matrix(_real(), load_matrix_gold(MATRIX_GOLD_FULL))
    assert report.total["cells"] == 118 * 49 and report.total["totals"] == 3 * 49


def test_metric_detects_corruptions() -> None:
    m = extract_matrix(_xlsx(GOOD), "m.xlsx")
    gold = {
        "rows": [
            {
                "number": 1,
                "semester": 1,
                "profile": "база",
                "part": "базовая",
                "credits": 5.0,
                "discipline": "Алгебра",
                "competences": ["УК-1", "УНК-03"],
            },
            {
                "number": 2,
                "semester": 1,
                "profile": "стандарт",
                "part": "вариативная",
                "credits": 3.0,
                "discipline": "Практикум",
                "competences": ["ПКП-2-ППК-Р2"],
            },
        ],
        "columns": [c.code for c in m.columns],
        "totals": {"СУММ": {"УК-1": 2}},
    }
    assert score_matrix(m, gold).passed()
    for field, value in (
        ("semester", 2),
        ("profile", "топ"),
        ("credits", 4.0),
        ("part", "x"),
        ("discipline", "Иное"),
        ("competences", ["УК-1"]),
    ):
        bad = copy.deepcopy(gold)
        bad["rows"][0][field] = value
        report = score_matrix(m, bad)
        assert report.accuracy(f"row.{field}") < 1.0, field
        assert not report.passed(1.0)
    bad = copy.deepcopy(gold)
    bad["totals"]["СУММ"]["УК-1"] = 9
    assert score_matrix(m, bad).accuracy("totals") < 1.0
    bad = copy.deepcopy(gold)
    bad["rows"].append({**gold["rows"][0], "number": 77})
    assert score_matrix(m, bad).missing
    assert THRESHOLD == 0.95


@needs_data
def test_real_matrix_against_real_plan_known_counts() -> None:
    if real_plan_markdown() is None:
        pytest.skip("нет data/Учебный план.zip")
    plan = extract_plan(real_plan_markdown() or "", "plan.docx")
    cmp = compare_matrix_with_plan(_real(), plan)
    assert cmp.columns_not_in_plan == [] and cmp.plan_competences_not_in_columns == []
    assert (cmp.matched, cmp.equal, len(cmp.different), len(cmp.unmatched)) == (67, 45, 22, 51)
    # Несовместимые профиль/часть и неоднозначные траектории не подбираются по компетенциям.
    assert sum("неоднозначное" in reason for reason in cmp.unmatched_reasons.values()) == 5
