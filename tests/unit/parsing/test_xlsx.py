"""Тесты парсера XLSX на синтетической книге (как матрица компетенций / учебный план)"""

from datetime import date, datetime
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook

from edu_doc_agent.parsing import BlockKind, CellSpan, parse_document, supported_suffixes
from edu_doc_agent.parsing.xlsx_parser import _grid, _value_text


@pytest.fixture
def matrix(tmp_path: Path) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Матрица покрытия"
    ws["A1"] = "Дисциплина"
    ws.merge_cells("A1:A2")  # по вертикали — две строки шапки
    ws["B1"] = "ЗЕТ"
    ws.merge_cells("B1:B2")
    ws["C1"] = "УК"
    ws.merge_cells("C1:D1")  # по горизонтали — группа над двумя столбцами
    ws["C2"], ws["D2"] = "УК-1", "УК-2"
    ws["E2"] = "служебный"  # скрытый столбец
    ws.column_dimensions["E"].hidden = True
    ws.append(["Алгебра и геометрия", 5, 1, 0, "x"])
    ws.append(["Черновик", 0, 0, 0])  # скрытая строка
    ws.row_dimensions[4].hidden = True
    ws.append(["Основы\nпрограммирования", 3.0, 0, 1])  # 3.0 -> '3', перенос строки в ячейке
    ws["H40"].number_format = "0.00"  # оформленная, но пустая область — не попадает
    hidden = wb.create_sheet("Служебный")
    hidden["A1"] = "не показывать"
    hidden.sheet_state = "hidden"
    wb.create_sheet("Пустой")
    path = tmp_path / "matrix.xlsx"
    wb.save(path)
    return path


def test_sheet_to_heading_and_table(matrix: Path) -> None:
    blocks = parse_document(matrix).blocks
    assert [(b.kind, b.text) for b in blocks] == [
        (BlockKind.HEADING, "Матрица покрытия"),  # скрытый и пустой листы пропущены
        (BlockKind.TABLE, ""),
    ]
    table = blocks[1]
    assert table.rows == [
        ["Дисциплина", "ЗЕТ", "УК", ""],  # по горизонтали — значение в первой ячейке
        ["Дисциплина", "ЗЕТ", "УК-1", "УК-2"],  # по вертикали — повтор
        ["Алгебра и геометрия", "5", "1", "0"],
        ["Основы\nпрограммирования", "3", "0", "1"],
    ]
    assert table.spans == [
        CellSpan(row=0, col=0, row_span=2),
        CellSpan(row=0, col=1, row_span=2),
        CellSpan(row=0, col=2, col_span=2),
    ]


def test_markdown(matrix: Path) -> None:
    assert parse_document(matrix).to_markdown() == (
        "# Матрица покрытия\n\n"
        "| Дисциплина | ЗЕТ | УК |  |\n"
        "| --- | --- | --- | --- |\n"
        "| Дисциплина | ЗЕТ | УК-1 | УК-2 |\n"
        "| Алгебра и геометрия | 5 | 1 | 0 |\n"
        "| Основы<br>программирования | 3 | 0 | 1 |\n"
    )


def test_value_text() -> None:
    assert _value_text(None) == ""
    assert _value_text(7) == "7"
    assert _value_text(2.0) == "2"
    assert _value_text(1.5) == "1.5"
    assert _value_text(True) == "да"
    assert _value_text(date(2026, 9, 1)) == "01.09.2026"
    assert _value_text(datetime(2026, 9, 1, 10, 30)) == "01.09.2026 10:30"
    assert _value_text("  поле ______ \n\n строка ") == "поле …\nстрока"


def test_xlsx_is_supported() -> None:
    assert ".xlsx" in supported_suffixes()


def test_formatting_far_from_data_does_not_grow_grid(tmp_path: Path) -> None:
    """Формат в CV1000 сдвигает max_row/max_column, но сетка строится только по данным."""
    wb = Workbook()
    ws = wb.active
    ws["A1"], ws["A2"] = "Дисциплина", "Алгебра"
    ws["CV1000"].number_format = "0.00"
    ws["XFD1048576"].number_format = "0.00"  # последний адрес листа: полный обход не закончится
    ws.merge_cells("C5:Z900")  # пустое объединение — тоже оформление
    path = tmp_path / "sparse.xlsx"
    wb.save(path)

    loaded = load_workbook(path)
    sheet = loaded.active
    stored = len(sheet._cells)
    assert _grid(sheet) == [[((1, 1), "Дисциплина")], [((2, 1), "Алгебра")]]
    assert len(sheet._cells) == stored  # обход не создал ячеек

    table = parse_document(path).blocks[1]
    assert table.rows == [["Дисциплина"], ["Алгебра"]]


def test_sparse_values_keep_only_filled_rows_and_columns(tmp_path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws["A1"], ws["CV1"] = "Код", "Итого"
    ws["A1000"], ws["CV1000"] = "УК-1", 5
    path = tmp_path / "far.xlsx"
    wb.save(path)
    assert parse_document(path).blocks[1].rows == [["Код", "Итого"], ["УК-1", "5"]]
