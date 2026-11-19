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

    hidden_cols = {
        i
        for dim in ws.column_dimensions.values()
        if dim.hidden and dim.min is not None
        for i in range(dim.min, (dim.max or dim.min) + 1)
    }
    positions = values.keys() | merged.keys()
    rows = sorted({r for r, _ in positions if not _row_hidden(ws, r)})
    cols = sorted({c for _, c in positions if c not in hidden_cols})
    return [[merged.get((r, c)) or ((r, c), values.get((r, c), "")) for c in cols] for r in rows]


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
