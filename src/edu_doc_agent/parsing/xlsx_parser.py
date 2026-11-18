"""XLSX -> блоки: каждый видимый лист — заголовок (имя листа) и таблица"""

from collections.abc import Hashable
from datetime import date, datetime, time
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet

from edu_doc_agent.parsing.common import Cell, clean_text, table_block
from edu_doc_agent.parsing.models import BaseParser, Block, BlockKind, ParsedDocument


class XlsxParser(BaseParser):
    def parse(self, path: Path) -> ParsedDocument:
        # data_only: значения формул из кэша Excel; если файл сохранён без кэша — ячейка пустая
        wb = load_workbook(path, data_only=True)
        blocks = []
        try:
            for ws in wb.worksheets:
                if ws.sheet_state != "visible":
                    continue
                table = table_block(_grid(ws))
                if table is None:
                    continue
                blocks.append(Block(BlockKind.HEADING, clean_text(ws.title), level=1))
                blocks.append(table)
        finally:
            wb.close()
        return ParsedDocument(source=str(path), blocks=blocks)


def _grid(ws: Worksheet) -> list[list[Cell]]:
    """Сетка видимых ячеек листа; столбцы без единого значения отброшены.

    Ключ ячейки — её координата, у объединённой — координата всего диапазона, а значение
    объединения записано во все его позиции (`table_block` сам оставит его по контракту).
    """
    merged: dict[tuple[int, int], tuple[Hashable, str]] = {}
    for rng in ws.merged_cells.ranges:
        text = _value_text(ws.cell(rng.min_row, rng.min_col).value)
        for row, col in rng.cells:
            merged[(row, col)] = (rng.coord, text)

    hidden_cols = {
        i
        for dim in ws.column_dimensions.values()
        if dim.hidden and dim.min is not None
        for i in range(dim.min, (dim.max or dim.min) + 1)
    }
    cols = [c for c in range(1, ws.max_column + 1) if c not in hidden_cols]
    grid = []
    for r, row in enumerate(ws.iter_rows(min_row=1, max_row=ws.max_row), start=1):
        if ws.row_dimensions[r].hidden:
            continue
        values = {cell.column: cell.value for cell in row}
        grid.append([merged.get((r, c)) or ((r, c), _value_text(values.get(c))) for c in cols])

    filled = [any(line[i][1] for line in grid) for i in range(len(cols))]
    return [[cell for cell, keep in zip(line, filled, strict=True) if keep] for line in grid]


def _value_text(value: object) -> str:
    """Значение ячейки -> текст: 5.0 -> '5', дата -> 'ДД.ММ.ГГГГ', строка — как в DOCX."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else format(value, ".15g")
    if isinstance(value, datetime):
        return value.strftime("%d.%m.%Y" if value.time() == time() else "%d.%m.%Y %H:%M")
    if isinstance(value, date):
        return value.strftime("%d.%m.%Y")
    if isinstance(value, time):
        return value.strftime("%H:%M")
    return clean_text(str(value), keep_newlines=True)
