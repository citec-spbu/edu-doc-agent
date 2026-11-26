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
from tests.fixtures.curriculum_plan import PLAN


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


@pytest.mark.parametrize("load", [None, "#N/A"])
def test_choice_keeps_uncoded_entry_with_unknown_hours(tmp_path: Path, load: str | None) -> None:
    path = _small_plan(
        tmp_path,
        [
            ["С02. Семестр 2"],
            ["Базовая часть периода обучения"],
            ["Блок.1.дисц", "от 1", "УК-1", "Дисциплины по выбору:", "зачёты:"],
            ["", "до 2", "", "Проектный практикум. Часть 2", "от 1 до 2"],
            ["Блок.1.дисц", None, "УК-1", "Китайский язык", "зачёт", load],
            ["Блок.1.дисц", 1, "УК-1", "[200] Второй курс", "зачёт", 12, 12, 12],
        ],
    )
    if load is not None:
        _cache_formula(path, "F7", "#N/A")
    plan = extract_plan_file(path)
    assert [(e.code, e.name) for e in plan.entries] == [
        (None, "Китайский язык"),
        ("200", "Второй курс"),
    ]
    first, second = plan.entries
    assert first.credits is None and first.contact_hours == {} and first.self_study_hours == {}
    assert first.competences == ["УК-1"] and first.control == ["зачёт"]
    assert first.source.line == 7
    for entry in (first, second):
        assert entry.group == "Блок дисциплин по выбору: Проектный практикум. Часть 2"
        assert entry.semester == 2 and entry.part == "Базовая часть периода обучения"
    if load is not None:
        assert any("Нагрузка!F7: ошибка формулы #N/A" in w for w in plan.warnings)


def _cache_formula(path: Path, address: str, value: str) -> None:
    """Сохранённая Excel формула с ошибкой в кэше, без запуска Excel."""
    wb = load_workbook(path)
    wb.active[address] = "=NA()"
    wb.save(path)
    with ZipFile(path) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    sheet = ET.fromstring(parts["xl/worksheets/sheet1.xml"])
    cell = sheet.find(f'.//s:c[@r="{address}"]', ns)
    cell.set("t", "e")
    cell.find("s:v", ns).text = value
    parts["xl/worksheets/sheet1.xml"] = ET.tostring(sheet)
    with ZipFile(path, "w") as z:
        for name, data in parts.items():
            z.writestr(name, data)


@pytest.mark.parametrize("merged_columns", [("A",), ("A", "B", "E")])
def test_english_course_with_own_fields_is_not_translation(tmp_path: Path, merged_columns) -> None:
    path = _small_plan(
        tmp_path,
        [
            ["Блок.1.дисц", 1, "УК-1", "[100] Первый курс", "зачёт", 12, 12, 12],
            ["Блок.1.дисц", 1, "УК-1", "Academic Writing", "зачёт", 12, 12, 12],
        ],
    )
    wb = load_workbook(path)
    for col in merged_columns:
        wb.active.merge_cells(f"{col}3:{col}4")
    wb.save(path)
    first, second = extract_plan_file(path).entries
    assert (first.code, first.name_en, first.total_hours) == ("100", None, 36)
    assert (second.code, second.name, second.name_en, second.total_hours) == (
        None,
        "Academic Writing",
        None,
        36,
    )
    assert second.source.line == 4


@pytest.mark.parametrize("cached", [False, True])
def test_control_error_is_unknown_and_located(tmp_path: Path, cached: bool) -> None:
    path = _small_plan(tmp_path, [["Блок.1.дисц", 1, "УК-1", "[100] Курс", "#N/A", 12, 12, 12]])
    if cached:
        _cache_formula(path, "E3", "#N/A")
    plan = extract_plan_file(path)
    assert plan.entries[0].control == []
    assert plan.entries[0].total_hours == 36
    assert any("Нагрузка!E3" in w and "#N/A" in w for w in plan.warnings)
    assert "#N/A" not in plan.entries[0].source.quote


def test_repeated_header_refreshes_fields_and_hours_with_context(tmp_path: Path) -> None:
    path = _small_plan(
        tmp_path,
        [
            ["С02. Семестр 2"],
            ["Базовая часть периода обучения"],
            ["Блок.1.дисц", 1, "УК-1", "[100] Первый курс", "зачёт", 12, 12, 12],
        ],
    )
    wb = load_workbook(path)
    ws = wb.active
    # Во второй шапке переставлены название/контроль и две группы часов.
    for original in (1, 2):
        row = [ws.cell(original, col).value for col in range(1, 9)]
        row[3], row[4] = row[4], row[3]
        if original == 1:
            row[5], row[7] = "Самостоятельная работа", "Аудиторная работа"
        ws.append(row)
    ws.append(["Блок.1.дисц", 1, "УК-1", "экзамен", "[200] Второй курс", 10, 14, 12])
    ws.append(["", "", "", "экзамен", "Second Course"])
    for col in ("A", "B", "C", "D"):
        ws.merge_cells(f"{col}8:{col}9")
    wb.save(path)
    first, second = extract_plan_file(path).entries
    assert (
        second.code == "200" and second.name == "Второй курс" and second.name_en == "Second Course"
    )
    assert second.control == ["экзамен"]
    assert second.contact_hours == {"текущий контроль": 12}
    assert second.self_study_hours == {"лекции": 10, "практические занятия": 14}
    assert (first.semester, second.semester) == (2, 2)
    assert first.part == second.part == "Базовая часть периода обучения"
    assert (second.source.line, second.source.end_line) == (8, 9)
    assert "Second Course" in second.source.quote


@pytest.mark.parametrize(
    "note",
    [
        "[990001] Сверить список компетенций для итоговой аттестации",
        "Код компетенции: уточнить наименование",
    ],
)
def test_note_does_not_start_semantic_section(plan_xlsx: Path, note: str) -> None:
    before = _without_sources(extract_plan_file(plan_xlsx).model_dump())
    wb = load_workbook(plan_xlsx)
    ws = wb.active
    row = next(c.row for c in ws["D"] if c.value and "[700000]" in c.value)
    ws.cell(row, 21, note)
    wb.save(plan_xlsx)
    assert _without_sources(extract_plan_file(plan_xlsx).model_dump()) == before


@pytest.mark.parametrize(
    "label",
    [
        "С02. Семестр 2",
        "Вариативная часть периода обучения",
        "Блок дисциплин Практикум. Часть 2",
        "Профиль 2",
        "Траектория 2",
    ],
)
def test_note_outside_table_does_not_cancel_context(plan_xlsx: Path, label: str) -> None:
    wb = load_workbook(plan_xlsx)
    ws = wb.active
    row = (
        next(c.row for c in ws["A"] if c.value == label)
        if label == "С02. Семестр 2"
        else next(c.row for c in ws["A"] if c.value == "Базовая часть периода обучения")
    )
    ws.cell(row, 1, label)
    wb.save(plan_xlsx)
    before = _without_sources(extract_plan_file(plan_xlsx).model_dump())
    ws.cell(row, 21, "Уточнить даты")
    wb.save(plan_xlsx)
    assert _without_sources(extract_plan_file(plan_xlsx).model_dump()) == before
    if label == "С02. Семестр 2":
        assert extract_plan_file(plan_xlsx).entries[2].semester == 2


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\u00a0\n"])
def test_header_line_break_keeps_entries_and_named_hours(plan_xlsx: Path, separator: str) -> None:
    before = _without_sources(extract_plan_file(plan_xlsx).model_dump())
    wb = load_workbook(plan_xlsx)
    for cell in list(wb.active._cells.values()):
        if cell.value in {"Аудиторная работа", "Самостоятельная работа"}:
            cell.value = cell.value.replace(" ", separator)
    wb.save(plan_xlsx)
    assert _without_sources(extract_plan_file(plan_xlsx).model_dump()) == before


def test_competence_text_uses_named_column(plan_xlsx: Path) -> None:
    wb = load_workbook(plan_xlsx)
    ws = wb.create_sheet("Согласование")
    ws.append(["Статус", "Код\nкомпетенции", "Наименование"])
    ws.append(["Согласовано", "УК-999", "Умеет выполнять расчёты"])
    wb.save(plan_xlsx)
    competence = extract_plan_file(plan_xlsx).competences[-1]
    assert (competence.code, competence.name) == ("УК-999", "Умеет выполнять расчёты")
    assert (competence.source.section, competence.source.line) == ("Согласование", 2)


def test_duplicate_visible_sections_warn_with_both_sheet_names(plan_xlsx: Path) -> None:
    before = extract_plan_file(plan_xlsx)
    wb = load_workbook(plan_xlsx)
    wb.copy_worksheet(wb.active).title = "Рабочая копия"
    wb.save(plan_xlsx)
    after = extract_plan_file(plan_xlsx)
    assert len(after.entries) == 2 * len(before.entries)
    assert len(after.competences) == 2 * len(before.competences)
    assert len(after.final_attestation) == 2 * len(before.final_attestation)
    duplicates = [w for w in after.warnings if "полностью совпадает" in w]
    assert len(duplicates) == 3
    assert all("Учебный план" in w and "Рабочая копия" in w for w in duplicates)


def test_translation_source_includes_both_rows_for_entry_and_attestation(plan_xlsx: Path) -> None:
    _save_bilingual(plan_xlsx)
    wb = load_workbook(plan_xlsx)
    ws = wb.active
    row = next(c.row for c in ws["C"] if c.value and "Thesis Defense" in c.value)
    ws.cell(row, 3, "Защита ВКР")
    ws.cell(row + 1, 3, "Thesis Defense")
    for col in ("A", "B", "K"):
        ws.merge_cells(f"{col}{row}:{col}{row + 1}")
    wb.save(plan_xlsx)
    plan = extract_plan_file(plan_xlsx)
    for entity, english in (
        (plan.entries[0], "Algebra"),
        (plan.final_attestation[0], "Thesis Defense"),
    ):
        assert entity.name_en == english
        assert entity.source.end_line == entity.source.line + 1
        assert entity.name in entity.source.quote and english in entity.source.quote


def test_synthetic_test_import_does_not_read_real_corpus() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['tests.real_data'] = None; "
            "import tests.unit.test_extraction_curriculum_xlsx",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def local_plan_xlsx(tmp_path: Path) -> Path:
    """Локальный XLSX читается только при выполнении выбранного теста."""
    import os

    from edu_doc_agent.eval.xlsx_documents import _member

    data_dir = Path(os.environ.get("DATA_DIR") or Path(__file__).resolve().parents[2] / "data")
    try:
        source, _ = _member(data_dir, "Учебный план.zip")
    except FileNotFoundError:
        pytest.skip("нет локального архива учебного плана")
    path = tmp_path / "local-plan.xlsx"
    path.write_bytes(source)
    return path


@pytest.mark.parametrize(
    "address,value",
    [
        ("U91", "[990001] Сверить список компетенций для итоговой аттестации"),
        ("U168", "Уточнить даты"),
        ("D139", "Проектный практикум. Часть 2"),
        ("F84", "Аудиторная\nработа"),
    ],
)
def test_local_excel_edit_preserves_plan(local_plan_xlsx: Path, address: str, value: str) -> None:
    before = extract_plan_file(local_plan_xlsx)
    assert len(before.entries) == 222 and len(before.final_attestation) == 1
    # Меняем одну ячейку в OOXML: openpyxl при сохранении удаляет кэш всех формул.
    with ZipFile(local_plan_xlsx) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    sheet = ET.fromstring(parts["xl/worksheets/sheet1.xml"])
    row_number = re.search(r"\d+$", address)[0]
    row = sheet.find(f'.//{{{ns}}}row[@r="{row_number}"]')
    cell = row.find(f'{{{ns}}}c[@r="{address}"]')
    if cell is None:
        cell = ET.SubElement(row, f"{{{ns}}}c", r=address)
    cell.clear()
    cell.attrib.update(r=address, t="inlineStr")
    ET.SubElement(ET.SubElement(cell, f"{{{ns}}}is"), f"{{{ns}}}t").text = value
    parts["xl/worksheets/sheet1.xml"] = ET.tostring(sheet)
    with ZipFile(local_plan_xlsx, "w") as z:
        for name, data in parts.items():
            z.writestr(name, data)
    after = extract_plan_file(local_plan_xlsx)
    assert len(after.entries) == 222 and len(after.final_attestation) == 1
    assert after.warnings == before.warnings == []
    expected = _without_sources(before.model_dump())
    if address == "D139":
        old_group = before.entries[
            next(i for i, e in enumerate(before.entries) if e.source.line == 143)
        ].group
        changed = 0
        for entry in expected["entries"]:
            if entry["group"] == old_group and entry["semester"] == 1:
                entry["group"] = "Блок дисциплин по выбору: " + value
                changed += 1
        assert changed > 0
    assert _without_sources(after.model_dump()) == expected
    for entity in [*after.entries, *after.final_attestation]:
        if entity.name_en is not None:
            assert entity.name_en in entity.source.quote
            # Перевод может уже находиться в основной ячейке.
            if entity.source.end_line is not None:
                assert entity.source.end_line == entity.source.line + 1


def test_choice_group_translation_does_not_change_group_name(tmp_path: Path) -> None:
    path = _small_plan(
        tmp_path,
        [
            ["Блок.1.дисц", "от 1", "УК-1", "Дисциплины по выбору:", "зачёты:"],
            ["", "до 2", "", "Иностранный язык", "от 1 до 2"],
            ["", "", "", "Foreign Language"],
            ["", "", "", "(выбрать от 1 до 2 дисц.)"],
            ["Блок.1.дисц", 1, "УК-1", "[200] Китайский язык", "зачёт", 0, 36, 0],
        ],
    )
    wb = load_workbook(path)
    wb.active.merge_cells("A3:A6")
    wb.save(path)
    entry = extract_plan_file(path).entries[0]
    assert entry.group == "Блок дисциплин по выбору: Иностранный язык"
