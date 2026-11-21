"""XLSX -> блоки: каждый видимый лист — заголовок (имя листа) и таблица"""

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
        warnings = []
        try:
            formulas = load_workbook(path, data_only=False)
            try:
                for ws in formulas.worksheets:
                    if ws.sheet_state != "visible":
                        continue
                    hidden = _hidden_columns(ws)
                    for cell in ws._cells.values():
                        if cell.data_type != "f" or _row_hidden(ws, cell.row):
                            continue
                        if cell.column in hidden:
                            continue
                        cached = wb[ws.title][cell.coordinate]
                        if cached.value is None:
                            warnings.append(
                                f"{ws.title}!{cell.coordinate}: формула без кэшированного значения"
                            )
                        elif cached.data_type == "e":
                            warnings.append(
                                f"{ws.title}!{cell.coordinate}: ошибка формулы {cached.value}"
                            )
            finally:
                formulas.close()
            for ws in wb.worksheets:
                if ws.sheet_state != "visible":
                    continue
                grid, rows, cols = _sheet_grid(ws)
                table = table_block(grid)
                if table is None:
                    continue
                table.source_rows = [
                    r for r, line in zip(rows, grid, strict=True) if table_block([line]) is not None
                ]
                table.source_columns = cols
                blocks.append(Block(BlockKind.HEADING, clean_text(ws.title), level=1))
                blocks.append(table)
        finally:
            wb.close()
        return ParsedDocument(source=str(path), blocks=blocks, warnings=warnings)


def _grid(ws: Worksheet) -> list[list[Cell]]:
    return _sheet_grid(ws)[0]


def _sheet_grid(ws: Worksheet) -> tuple[list[list[Cell]], list[int], list[int]]:
    """Сетка видимых ячеек листа: только строки и столбцы, где есть значения.

    Ключ ячейки — её координата, у объединённой — координата всего диапазона, а значение
    объединения записано во все его позиции (`table_block` сам оставит его по контракту).

    Сетка строится по сохранённым в файле ячейкам, а не по прямоугольнику до
    `max_row`/`max_column`: эти границы учитывают и ячейки с одним оформлением, и маленький
    файл с форматом в CV1000 давал бы сетку на 100 000 позиций.
    """
    # ws._cells — только ячейки из файла; ws.iter_rows() создал бы весь прямоугольник
    values = {pos: text for pos, cell in ws._cells.items() if (text := _value_text(cell.value))}
    merged: dict[tuple[int, int], Cell] = {}
    for rng in ws.merged_cells.ranges:
        text = values.get((rng.min_row, rng.min_col), "")
        if not text:
            continue  # пустое объединение — то же оформление, сетку не расширяет
        for row, col in rng.cells:
            merged[(row, col)] = (rng.coord, text)

    hidden_cols = _hidden_columns(ws)
    positions = values.keys() | merged.keys()
    rows = sorted({r for r, _ in positions if not _row_hidden(ws, r)})
    cols = sorted({c for _, c in positions if c not in hidden_cols})
    grid = [[merged.get((r, c)) or ((r, c), values.get((r, c), "")) for c in cols] for r in rows]
    return grid, rows, cols


def _hidden_columns(ws: Worksheet) -> set[int]:
    return {
        i
        for dim in ws.column_dimensions.values()
        if dim.hidden and dim.min is not None
        for i in range(dim.min, (dim.max or dim.min) + 1)
    }


def _row_hidden(ws: Worksheet, row: int) -> bool:
    dim = ws.row_dimensions.get(row)  # .get — без создания записи для каждой строки
    return dim is not None and dim.hidden


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
