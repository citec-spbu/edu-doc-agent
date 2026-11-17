"""Общие части парсеров: одинаковое поведение для всех форматов"""

from edu_doc_agent.parsing.common import clean_text, is_rule, paragraph_block, table_block
from edu_doc_agent.parsing.models import BlockKind, CellSpan


def test_table_block_merges_by_key() -> None:
    """Объединение 2×2 и служебная строка: ключи вместо элементов Word."""
    a, b = "A1:B2", "A3:C3"  # ключи объединений, как координаты в XLSX
    grid = [
        [(a, "Период"), (a, "Период"), ("C1", "Итого")],
        [(a, "Период"), (a, "Период"), ("C2", "з.е.")],
        [(b, "ТРАЕКТОРИЯ"), (b, "ТРАЕКТОРИЯ"), (b, "ТРАЕКТОРИЯ")],
        [("A4", ""), ("B4", ""), ("C4", "")],  # пустая строка — пропускается
        [("A5", "Семестр 1"), ("B5", "30")],  # короткая строка — дополняется
    ]
    table = table_block(grid)
    assert table is not None and table.kind == BlockKind.TABLE
    assert table.rows == [
        ["Период", "", "Итого"],  # по горизонтали — значение только в первой ячейке
        ["Период", "", "з.е."],  # по вертикали — повтор
        ["ТРАЕКТОРИЯ", "", ""],
        ["Семестр 1", "30", ""],
    ]
    assert table.spans == [
        CellSpan(row=0, col=0, row_span=2, col_span=2),
        CellSpan(row=2, col=0, col_span=3),
    ]


def test_table_block_empty() -> None:
    assert table_block([]) is None
    assert table_block([[("A1", ""), ("B1", "")]]) is None


def test_paragraph_block() -> None:
    assert paragraph_block("1.2. Вид практики", bold=True).level == 2
    assert paragraph_block("Аннотация", bold=True).kind == BlockKind.HEADING
    assert paragraph_block("Аннотация").kind == BlockKind.PARAGRAPH
    item = paragraph_block("– пункт с маркером")
    assert (item.kind, item.text) == (BlockKind.LIST_ITEM, "пункт с маркером")
    assert paragraph_block("знать модель;", list_style=True).kind == BlockKind.LIST_ITEM


def test_clean_text() -> None:
    assert clean_text("от\xa0____ №  ____\n\n") == "от … № …"
    assert clean_text("строка\nвторая", keep_newlines=True) == "строка\nвторая"
    assert is_rule("________") and not is_rule("См. 1.1 ....")
