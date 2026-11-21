"""XLSX → сущности: эквивалентность Markdown, объединения, источники и формулы."""

import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest
from openpyxl import Workbook, load_workbook

from edu_doc_agent.extraction.__main__ import main
from edu_doc_agent.extraction.curriculum import extract_plan, extract_plan_file
from edu_doc_agent.extraction.md_utils import parse_tables
from edu_doc_agent.parsing import parse_document
from tests.unit.test_extraction_curriculum import PLAN


@pytest.fixture
def plan_xlsx(tmp_path: Path) -> Path:
    """Та же программа, что в Markdown-фикстуре, на одном листе, с разными сетками."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Учебный план"
    ws.append(
        ["Программирование и информационные технологии Programming and Information Technology"]
    )
    for table in parse_tables(PLAN):
        ws.append([])
        if table.section:
            ws.append([table.section])
        for idx, row in enumerate(table.rows):
            # Широкая ГИА: компетенции в K, а не в D, как в Markdown.
            if table.section and "Раздел 3" in table.section:
                row = row[:3] + [""] * 7 + row[3:]
            ws.append(row)
            if idx == 0 and "Аудиторная работа" in row:
                top = ws.max_row
        if table.section and "Раздел 2" in table.section:
            ws.merge_cells(start_row=top, start_column=6, end_row=top, end_column=8)
            ws.merge_cells(start_row=top, start_column=9, end_row=top, end_column=10)
            for col in [1, 2, 3, 4, 5, 11]:
                ws.merge_cells(start_row=top, start_column=col, end_row=top + 1, end_column=col)
    ws["Z500"].number_format = "0.00"
    path = tmp_path / "plan.xlsx"
    wb.save(path)
    return path


def _without_sources(value: object) -> object:
    if isinstance(value, dict):
        return {
            k: _without_sources(v)
            for k, v in value.items()
            if k not in {"source", "sources", "file", "warnings"}
        }
    if isinstance(value, list):
        return [_without_sources(v) for v in value]
    return value


def test_xlsx_and_markdown_have_same_entities(plan_xlsx: Path) -> None:
    xlsx = extract_plan_file(plan_xlsx)
    markdown = extract_plan(PLAN)
    assert _without_sources(xlsx.model_dump()) == _without_sources(markdown.model_dump())
    assert [re.sub(r"строка \d+", "строка N", w) for w in xlsx.warnings] == [
        re.sub(r"строка \d+", "строка N", w) for w in markdown.warnings
    ]
    assert len(xlsx.entries) == 4
    assert len(xlsx.competences) == 2
    assert len(xlsx.final_attestation) == 1


def test_sources_are_original_sheet_rows(plan_xlsx: Path) -> None:
    ws = load_workbook(plan_xlsx, data_only=True).active
    plan = extract_plan(parse_document(plan_xlsx))
    assert plan.file == plan_xlsx.name
    for entry in plan.entries:
        ref = entry.source
        assert ref.section == ws.title
        assert entry.code in str(ws.cell(ref.line, 4).value)
        assert entry.code in ref.quote
    assert plan.sources["level"].quote
    assert plan.sources["registration_number"].section == ws.title


def _save_bilingual(plan_xlsx: Path) -> None:
    wb = load_workbook(plan_xlsx)
    ws = wb.active
    row = next(c.row for c in ws["D"] if c.value and "[000160]" in c.value)
    ws.insert_rows(row + 1)
    ws.cell(row, 4, "[000160] Алгебра")
    ws.cell(row + 1, 4, "Algebra")
    ws.merge_cells(start_row=row, start_column=1, end_row=row + 1, end_column=1)
    ws.merge_cells(start_row=row, start_column=2, end_row=row + 1, end_column=2)
    ws.merge_cells(start_row=row, start_column=5, end_row=row + 1, end_column=5)
    wb.save(plan_xlsx)


def test_bilingual_rows_are_one_entity(plan_xlsx: Path) -> None:
    _save_bilingual(plan_xlsx)
    plan = extract_plan_file(plan_xlsx)
    assert len(plan.entries) == 4
    assert (plan.entries[0].name, plan.entries[0].name_en) == ("Алгебра", "Algebra")
    assert plan.entries[1].code == "700000"


def _formula(path: Path, cached: str | None) -> int:
    wb = load_workbook(path)
    ws = wb.active
    row = next(c.row for c in ws["D"] if c.value and "[000160]" in c.value)
    ws.cell(row, 2, f"=SUM(F{row}:J{row})/36")
    wb.save(path)
    if cached is not None:
        with ZipFile(path) as z:
            parts = {name: z.read(name) for name in z.namelist()}
        ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheet = ET.fromstring(parts["xl/worksheets/sheet1.xml"])
        cell = sheet.find(f'.//s:c[@r="B{row}"]', ns)
        if cached.startswith("#"):
            cell.set("t", "e")
        cell.find("s:v", ns).text = cached
        parts["xl/worksheets/sheet1.xml"] = ET.tostring(sheet)
        with ZipFile(path, "w") as z:
            for name, data in parts.items():
                z.writestr(name, data)
    return row


@pytest.mark.parametrize("value", ["2", "0"])
def test_cached_formula_is_used(plan_xlsx: Path, value: str) -> None:
    _formula(plan_xlsx, value)
    plan = extract_plan_file(plan_xlsx)
    assert plan.entries[0].credits == float(value)
    assert not any("формул" in w for w in plan.warnings)


def test_uncached_formula_stays_unknown_and_is_reported(plan_xlsx: Path) -> None:
    row = _formula(plan_xlsx, None)
    plan = extract_plan_file(plan_xlsx)
    assert plan.entries[0].credits is None
    assert any(f"Учебный план!B{row}" in w and "без кэшированного" in w for w in plan.warnings)


def test_formula_error_is_reported_and_stays_unknown(plan_xlsx: Path) -> None:
    row = _formula(plan_xlsx, "#DIV/0!")
    plan = extract_plan_file(plan_xlsx)
    assert plan.entries[0].credits is None
    assert any(f"Учебный план!B{row}: ошибка формулы #DIV/0!" in w for w in plan.warnings)


def test_hidden_data_is_excluded(plan_xlsx: Path) -> None:
    wb = load_workbook(plan_xlsx)
    ws = wb.active
    row = next(c.row for c in ws["D"] if c.value and "[000160]" in c.value)
    ws.row_dimensions[row].hidden = True
    ws.column_dimensions["K"].hidden = True
    hidden = wb.create_sheet("Черновик")
    hidden.append(["Код компетенции", "Наименование"])
    hidden.append(["УК-9", "Не включать"])
    hidden.sheet_state = "hidden"
    wb.save(plan_xlsx)
    plan = extract_plan_file(plan_xlsx)
    assert len(plan.entries) == 3
    assert all(e.code != "000160" for e in plan.entries)
    assert all(not e.active_hours for e in plan.entries)
    assert [c.code for c in plan.competences] == ["УК-1", "ПКП-УНК-03"]


def test_cli_plan(plan_xlsx: Path, capsys: pytest.CaptureFixture) -> None:
    assert main([str(plan_xlsx), "--type", "plan"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["entries"][0]["code"] == "000160"


def _small_plan(tmp_path: Path, rows: list[list]) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Нагрузка"
    ws.append(
        [
            "Код Блока",
            "Трудоёмкость",
            "Код компетенции",
            "Наименование",
            "Виды текущего контроля",
            "Аудиторная работа",
            "",
            "Самостоятельная работа",
        ]
    )
    ws.append(
        [
            "Код Блока",
            "Трудоёмкость",
            "Код компетенции",
            "Наименование",
            "Виды текущего контроля",
            "Лекции",
            "Практические занятия",
            "Текущий контроль",
        ]
    )
    for row in rows:
        ws.append(row)
    path = tmp_path / "small.xlsx"
    wb.save(path)
    return path


def test_merged_block_across_several_disciplines_keeps_both(tmp_path: Path) -> None:
    path = _small_plan(
        tmp_path,
        [
            ["Блок.1.дисц", 1, "УК-1", "[100] Первый курс", "зачёт", 12, 12, 12],
            ["", "", "", "First Course"],
            ["", "", "УК-2", "[200] Второй курс", "", 0, 36, 0],
            ["", "", "", "Second Course"],
        ],
    )
    wb = load_workbook(path)
    for col in ("A", "B", "E"):
        wb.active.merge_cells(f"{col}3:{col}6")
    wb.save(path)
    plan = extract_plan_file(path)
    assert [(e.code, e.name_en, e.credits) for e in plan.entries] == [
        ("100", "First Course", 1),
        ("200", "Second Course", 1),
    ]
    assert [e.source.line for e in plan.entries] == [3, 5]


def test_choice_headers_are_context_and_uncoded_discipline_is_kept(tmp_path: Path) -> None:
    path = _small_plan(
        tmp_path,
        [
            ["Блок.1.дисц", "от 1", "УК-1", "Дисциплины по выбору:", "зачёты:"],
            ["", "до 2", "", "Иностранный язык", "от 1 до 2"],
            ["", "", "", "(выбрать от 1 до 2 дисц.)"],
            ["Блок.1.дисц", 1, "УК-1", "Китайский язык", "зачёт", 0, 36, 0],
        ],
    )
    plan = extract_plan_file(path)
    assert len(plan.entries) == 1
    assert plan.entries[0].code is None
    assert plan.entries[0].name == "Китайский язык"
    assert plan.entries[0].group == "Блок дисциплин по выбору: Иностранный язык"
    assert plan.entries[0].total_hours == 36


def test_semester_in_choice_group_does_not_reset_plan_context(tmp_path: Path) -> None:
    path = _small_plan(
        tmp_path,
        [
            ["С02. Семестр 2"],
            ["Факультативные занятия"],
            ["Блок.1.дисц", "от 1", "УК-1", "Дисциплины по выбору:", "зачёты:"],
            ["", "до 2", "", "Китайский язык. Семестр 1", "от 1 до 2"],
            ["Блок.1.дисц", 1, "УК-1", "[990001] Китайский язык", "зачёт", 0, 36, 0],
        ],
    )
    entry = extract_plan_file(path).entries[0]
    assert entry.semester == 2
    assert entry.part == "Факультативные занятия"
    assert entry.group == "Блок дисциплин по выбору: Китайский язык. Семестр 1"


def test_multiple_sheets_are_processed(tmp_path: Path, plan_xlsx: Path) -> None:
    wb = load_workbook(plan_xlsx)
    ws = wb.create_sheet("Компетенции дополнительно")
    ws.append(["Код компетенции", "Наименование"])
    ws.append(["УК-9", "Дополнительная компетенция"])
    wb.save(plan_xlsx)
    plan = extract_plan_file(plan_xlsx)
    assert len(plan.competences) == 3
    assert plan.competences[-1].source.section == ws.title
    assert not any("УК-9" in w for w in plan.warnings)


def test_vertical_name_merge_does_not_duplicate_a_record(tmp_path: Path) -> None:
    path = _small_plan(
        tmp_path,
        [
            ["Блок.1.дисц", 1, "УК-1", "[100] Алгебра", "зачёт", 12, 12, 12],
            [],
        ],
    )
    wb = load_workbook(path)
    for col in ("A", "B", "C", "D", "E"):
        wb.active.merge_cells(f"{col}3:{col}4")
    wb.save(path)
    plan = extract_plan_file(path)
    assert len(plan.entries) == 1
    assert plan.entries[0].code == "100"


def test_incomplete_headers_are_reported(plan_xlsx: Path) -> None:
    wb = load_workbook(plan_xlsx)
    wb.active.column_dimensions["D"].hidden = True
    wb.active.column_dimensions["B"].hidden = True
    wb.save(plan_xlsx)
    plan = extract_plan_file(plan_xlsx)
    assert any("нет наименования" in w for w in plan.warnings)
    assert any("неполная шапка итоговой" in w for w in plan.warnings)


def test_markdown_file_remains_supported(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    path.write_text(PLAN, encoding="utf-8")
    assert extract_plan_file(path) == extract_plan(PLAN, path.name)


def test_docx_and_xlsx_have_same_entities(plan_xlsx: Path, tmp_path: Path) -> None:
    from docx import Document

    doc = Document()
    doc.add_paragraph(
        "Программирование и информационные технологии Programming and Information Technology"
    )
    for table in parse_tables(PLAN):
        if table.section:
            doc.add_heading(table.section, level=1)
        out = doc.add_table(rows=0, cols=len(table.rows[0]))
        for row in table.rows:
            cells = out.add_row().cells
            for cell, value in zip(cells, row, strict=True):
                cell.text = value.replace("<br>", "\n")
    path = tmp_path / "plan.docx"
    doc.save(path)
    assert _without_sources(extract_plan_file(path).model_dump()) == _without_sources(
        extract_plan_file(plan_xlsx).model_dump()
    )


def test_local_xlsx_has_no_missing_or_duplicate_records(tmp_path: Path) -> None:
    """Реальный файл локальный; проверка полноты по исходным записям, без DOCX другой версии."""
    from tests.real_data import DATA_DIR

    archives = list(DATA_DIR.glob("Учеб*план.zip"))
    if not archives:
        pytest.skip("нет локального архива учебного плана")
    with ZipFile(archives[0]) as z:
        names = [n for n in z.namelist() if n.endswith(".xlsx")]
        if not names:
            pytest.skip("в архиве нет XLSX")
        path = tmp_path / "real-plan.xlsx"
        path.write_bytes(z.read(names[0]))
    plan = extract_plan_file(path)
    ws = load_workbook(path, data_only=True).active
    coded_rows = {
        c.row
        for c in ws._cells.values()
        if isinstance(c.value, str) and re.match(r"^\[\d+\]", c.value.strip())
    }
    assert {e.source.line for e in plan.entries if e.code} == coded_rows
    assert len([e for e in plan.entries if e.code]) == len(coded_rows)
    assert len(plan.entries) == 222
    assert len(plan.competences) == 49
    assert len(plan.final_attestation) == 1
    assert plan.warnings == []
