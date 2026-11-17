"""Сетка Word: смещённые границы, пустые ячейки и неоднозначное перекрытие."""

from pathlib import Path

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Twips

from edu_doc_agent.extraction import extract_rpd, extract_rpd_file
from edu_doc_agent.extraction.md_utils import MdTable
from edu_doc_agent.extraction.workload_table import CONTACT, CREDITS, _aligned_columns
from edu_doc_agent.parsing import CellSpan, parse_document
from tests.real_data import REAL_DOCX_BY_MD, find_md


def _shifted_docx(tmp_path: Path) -> Path:
    doc = Document()
    doc.add_heading("2.1. Трудоёмкость", level=2)
    table = doc.add_table(rows=4, cols=7)
    # Практические занятия тела перекрывают лекции сильнее своего заголовка.
    # Сопоставление каждого числа по локальному максимуму здесь неверно.
    for col, width in zip(
        table._tbl.tblGrid.gridCol_lst, [100, 10, 80, 10, 40, 60, 100], strict=True
    ):
        col.w = Twips(width)
    table.cell(0, 0).merge(table.cell(1, 0)).text = "Период"
    table.cell(0, 1).merge(table.cell(0, 4)).text = "Контактная работа"
    table.cell(0, 5).text = "Самостоятельная работа"
    table.cell(1, 5).text = "с использованием методических материалов"
    table.cell(0, 6).merge(table.cell(1, 6)).text = "Трудоёмкость"
    table.cell(1, 1).merge(table.cell(1, 2)).text = "Лекции"
    table.cell(1, 3).merge(table.cell(1, 4)).text = "Практические занятия"
    for row, label in [(2, "Семестр 2"), (3, "ИТОГО")]:
        table.cell(row, 0).text = label
        table.cell(row, 1).text = "20"
        table.cell(row, 2).merge(table.cell(row, 4)).text = "30"
        table.cell(row, 5).text = "22"
        table.cell(row, 6).text = "2"
    path = tmp_path / "shifted.docx"
    doc.save(path)
    return path


def test_docx_geometry_aligns_hours_without_reusing_cells(tmp_path):
    path = _shifted_docx(tmp_path)
    parsed = parse_document(path)
    assert parsed.tables[0].grid_widths == [100, 10, 80, 10, 40, 60, 100]
    assert parsed.to_dict()["blocks"][1]["grid_widths"] == [100, 10, 80, 10, 40, 60, 100]
    for result in (extract_rpd(parsed, path.name), extract_rpd_file(path)):
        assert len(result.workload_table) == 2
        for row in result.workload_table:
            assert row.contact_hours == {"лекции": 20, "практические занятия": 30}
            assert row.self_study_hours == {"с использованием методических материалов": 22}
            assert row.credits == 2 and row.total_hours == 72
            assert row.source.quote and row.source.line
        assert not any(w.startswith("трудоёмкость") for w in result.warnings)


def test_empty_cell_still_reserves_its_header():
    table = MdTable(
        rows=[
            ["Период", "Лекции", "", "Практические занятия", "Трудоёмкость"],
            ["Семестр 1", "", "36", "", "1"],
        ],
        row_lines=[1, 2],
        section="2.1",
        grid_widths=[100, 10, 80, 40, 100],
        spans=[CellSpan(0, 1, col_span=2), CellSpan(1, 2, col_span=2)],
    )
    warnings = []
    mapping = _aligned_columns(
        table,
        0,
        1,
        {1: (CONTACT, "лекции"), 3: (CONTACT, "практические занятия"), 4: (None, CREDITS)},
        warnings,
    )
    assert mapping == {
        1: (CONTACT, "лекции"),
        2: (CONTACT, "практические занятия"),
        4: (None, CREDITS),
    }
    assert warnings == []


def test_ambiguous_numeric_cell_is_not_assigned_to_two_kinds():
    table = MdTable(
        rows=[["Период", "Лекции", "Практические занятия"], ["Семестр 1", "36", ""]],
        row_lines=[1, 2],
        section="2.1",
        grid_widths=[100, 100, 100],
        spans=[CellSpan(1, 1, col_span=2)],
    )
    warnings = []
    assert (
        _aligned_columns(
            table, 0, 1, {1: (CONTACT, "лекции"), 2: (CONTACT, "практические занятия")}, warnings
        )
        == {}
    )
    assert len(warnings) == 1 and "неоднозначное перекрытие" in warnings[0]


def test_parser_preserves_grid_before_and_after(tmp_path):
    doc = Document()
    table = doc.add_table(rows=2, cols=3)
    for i, label in enumerate(["Слева", "По центру", "Справа"]):
        table.cell(0, i).text = label
    table.cell(1, 1).text = "Значение"
    tr = table.rows[1]._tr
    tr.remove(tr.tc_lst[-1])
    tr.remove(tr.tc_lst[0])
    for name in ("gridBefore", "gridAfter"):
        node = OxmlElement(f"w:{name}")
        node.set(qn("w:val"), "1")
        tr.get_or_add_trPr().append(node)
    path = tmp_path / "omitted.docx"
    doc.save(path)
    parsed = parse_document(path)
    assert parsed.tables[0].rows[1] == ["", "Значение", ""]
    assert parsed.tables[0].spans == []


@pytest.mark.skipif(
    find_md("С3_001868") is None, reason="нет локального DOCX математической логики"
)
def test_real_shifted_grid_has_exact_hours_and_matching_semesters():
    path = find_md("С3_001868")
    result = extract_rpd_file(REAL_DOCX_BY_MD[path])
    semester, total = result.workload_table
    for row in (semester, total):
        assert row.contact_hours == {
            "лекции": 26,
            "консультации": 2,
            "практические занятия": 26,
            "контрольные работы": 4,
            "коллоквиумы": 2,
            "промежуточная аттестация": 2,
        }
        assert row.self_study_hours == {
            "с использованием методических материалов": 22,
            "промежуточная аттестация": 24,
        }
        assert (row.credits, row.total_hours, row.active_hours) == (3, 108, 26)
    assert semester.semester == 3 and total.is_total
    assert {f.semester for f in result.control_forms} == {3}
    assert not any(w.startswith("трудоёмкость") for w in result.warnings)
    # Обычный Markdown не переносит геометрию Word: неполный разбор видим пользователю.
    md = parse_document(REAL_DOCX_BY_MD[path]).to_markdown()
    assert any("не извлечены зачётные единицы" in w for w in extract_rpd(md, "plain.md").warnings)
