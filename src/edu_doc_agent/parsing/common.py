"""Общие части парсеров: очистка текста, абзацы и таблицы -> блоки.

Формат-специфичный код (DOCX, XLSX, PDF) только достаёт из файла текст абзаца с признаками
оформления или сетку ячеек таблицы, а превращает их в блоки по контракту из `models.py`
этот модуль — так все форматы выдают одинаковый результат.
"""

import re
from collections.abc import Hashable, Sequence

from edu_doc_agent.parsing.models import Block, BlockKind, CellSpan

SECTION = re.compile(r"^раздел\s+\d", re.IGNORECASE)  # раздел
NUMBERED = re.compile(r"^(\d+(?:\.\d+)+)\.?\s")  # нумерация
BULLET = re.compile(r"^[•·▪◦‣−–—-]\s+")  # маркер
_BLANK = re.compile(r"_{4,}")  # поле для заполнения '______' -> '…'
_RULE = re.compile(r"[\s_.…\-–—=*]{4,}")  # абзац целиком из таких символов - линия-разделитель
_SPACES = re.compile(r"[ \t ]+")  # пробелы, табы, неразрывные пробелы
MAX_HEADING_LEN = 200

# Ячейка сетки: (ключ, текст). Ключ одинаков у всех позиций одной объединённой ячейки,
# текст — значение этой ячейки. Ключом может быть любой hashable-объект: элемент XML ячейки
# (DOCX), координата объединения (XLSX), рамка ячейки (PDF).
# None обозначает пропущенную позицию сетки и не участвует в объединениях.
Cell = tuple[Hashable, str]


def clean_text(text: str, keep_newlines: bool = False) -> str:
    """Мягкие переносы убраны, поля '____' -> '…', пробелы схлопнуты, пустые строки убраны."""
    text = _BLANK.sub("…", text.replace("\xad", ""))
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return ("\n" if keep_newlines else " ").join(line for line in lines if line)


def is_rule(text: str) -> bool:
    """Абзац-линия: '--------', '________', '........' без текста."""
    return bool(_RULE.fullmatch(text.strip()))


def heading_level(text: str) -> int:
    """Уровень заголовка по тексту: «Раздел N» -> 1, «1.2.3.» -> 3, иначе 3."""
    if SECTION.match(text):
        return 1
    if match := NUMBERED.match(text):
        return match.group(1).count(".") + 1
    return 3


def paragraph_block(text: str, *, bold: bool = False, list_style: bool = False) -> Block:
    """Абзац без автонумерации -> заголовок, пункт списка или обычный абзац.

    `text` — уже очищенный (`clean_text`) и непустой. `bold` — весь абзац жирный,
    `list_style` — абзац оформлен стилем списка.
    """
    is_title = not text.endswith((".", ";", ","))
    is_short = len(text) <= MAX_HEADING_LEN and not text.endswith((";", ","))
    typed_section = SECTION.match(text) or NUMBERED.match(text)
    if bold and (is_short or (typed_section and is_title)):
        return Block(BlockKind.HEADING, text, level=heading_level(text))
    if match := BULLET.match(text):
        return Block(BlockKind.LIST_ITEM, text[match.end() :])
    if list_style:
        return Block(BlockKind.LIST_ITEM, text)
    return Block(BlockKind.PARAGRAPH, text)


def table_block(grid: Sequence[Sequence[Cell]]) -> Block | None:
    """Сетка ячеек -> блок-таблица по контракту.

    Объединение по горизонтали — значение остаётся только в первой ячейке, по вертикали —
    повторяется в каждой строке. Пустые строки пропускаются, строки дополняются до одной
    длины, объединённые ячейки описываются в `Block.spans`. Если таблица пустая — None.
    """
    lines = []
    for row in grid:
        line = [
            (key, "" if key is not None and c > 0 and key == row[c - 1][0] else text)
            for c, (key, text) in enumerate(row)
        ]
        if any(text for _, text in line):
            lines.append(line)
    if not lines:
        return None
    width = max(len(line) for line in lines)
    rows = [[text for _, text in line] + [""] * (width - len(line)) for line in lines]
    return Block(BlockKind.TABLE, rows=rows, spans=_spans(lines))


def _spans(lines: list[list[Cell]]) -> list[CellSpan]:
    """Объединенные ячейки: позиции сетки, которые занимает одна и та же ячейка (один ключ)"""
    positions: dict[Hashable, list[tuple[int, int]]] = {}
    for r, line in enumerate(lines):
        for c, (key, _) in enumerate(line):
            if key is not None:
                positions.setdefault(key, []).append((r, c))
    spans = []
    for cells in positions.values():
        rows, cols = {r for r, _ in cells}, {c for _, c in cells}
        if len(cells) > 1:
            spans.append(CellSpan(min(rows), min(cols), len(rows), len(cols)))
    return sorted(spans, key=lambda s: (s.row, s.col))
