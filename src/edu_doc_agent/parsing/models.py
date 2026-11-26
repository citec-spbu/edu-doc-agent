"""Результат парсинга: документ -> блоки -> Markdown

Контракт выхода (на него опирается модуль извлечения, extraction/rpd.py)

Основной формат: Markdown из `ParsedDocument.to_markdown()`. Блоки (`blocks`, `to_dict()`) —
та же информация в структурированном виде. Формат зафиксирован тестом `test_markdown_contract`
(tests/unit/parsing/test_parsing.py); менять его — только вместе с модулем извлечения.

- Блоки идут в порядке документа и разделены пустой строкой. Колонтитулы, картинки и сноски
  в результат не входят.
- Заголовок: «#» × уровень (1–6), пробел, номер автонумерации Word (если есть), пробел, текст:
  `# Раздел 1. Характеристики`, `## 1.3. Перечень результатов обучения`, `### Аннотация`.
  Номер, набранный вручную, остаётся частью текста.
- Пункт списка: номер автонумерации («1.», «а)») или «-», пробел, текст. Вложенность — 3 пробела
  на уровень, не глубже чем на 1 от предыдущего пункта.
- Таблица: pipe-таблица; первая строка таблицы документа — шапка, за ней `| --- | … |`.
  Все строки одной длины. Объединение по вертикали — значение повторено в каждой строке,
  по горизонтали — значение в первой ячейке, остальные пустые. Перенос строки в ячейке — `<br>`,
  символ «|» — `\\|`.
- Объединённые ячейки таблицы — в `Block.spans` (только в блоках и `to_dict()`, в Markdown их нет):
  для каждой объединённой ячейки — якорь (`row`, `col` — индексы в `rows`) и размер
  (`row_span`, `col_span`). Так пустая ячейка из объединения отличима от пустого значения,
  а повтор из объединения — от настоящего дубля.
- `Block.grid_widths` хранит ширины базовой сетки Word (`tblGrid`) в twips.
  Вместе со `spans` они позволяют сопоставить ячейки при разных границах шапки и тела.
  Геометрия доступна в блоках и `to_dict()`, но не в обычном Markdown.
- Абзац — текст как есть; если он начинается с одного из символов `# * + - > | =`,
  перед ним ставится `\\`, чтобы Markdown не принял его за разметку.
- Поля для заполнения (4 и более «_» подряд) → «…»; абзацы-линии целиком из `_ - – — . … = *`
  убираются. Остальной текст (многоточия, «----» в таблицах) не меняется.

XLSX: каждый видимый лист — заголовок 1-го уровня (имя листа) и таблица по правилам выше.
Скрытые листы, строки и столбцы, а также столбцы без единого значения пропускаются.
Значения: формулы — результат из кэша Excel, числа без «.0», даты — `ДД.ММ.ГГГГ`.

PDF (только с текстовым слоем): абзацы и заголовки восстанавливаются по вёрстке, таблицы —
по рамкам (продолжение у границы страниц с теми же столбцами приклеивается), колонтитулы,
номера страниц и сноски убираются. Номер в начале абзаца остаётся частью текста. Подробности
и ограничения — в `parsing/pdf_parser.py`.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

_MD_LINE_START = ("#", "*", "+", "-", ">", "|", "=")


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"


@dataclass
class CellSpan:
    """Объединенная ячейка таблицы: левый верхний угол (якорь) в `Block.rows` и размер"""

    row: int
    col: int
    row_span: int = 1
    col_span: int = 1


@dataclass
class Block:
    kind: BlockKind
    text: str = ""
    level: int = 0  # заголовок
    number: str = ""  # нумерация
    rows: list[list[str]] = field(default_factory=list)  # таблица
    spans: list[CellSpan] = field(default_factory=list)  # таблица: объединенные ячейки
    grid_widths: list[int] = field(default_factory=list)  # ширины tblGrid в twips (1/1440 дюйма)

    def to_markdown(self) -> str:
        if self.kind == BlockKind.HEADING:
            return "#" * min(max(self.level, 1), 6) + " " + f"{self.number} {self.text}".strip()
        if self.kind == BlockKind.LIST_ITEM:
            return f"{self.number or '-'} {_escape(self.text)}"
        if self.kind == BlockKind.TABLE:
            header, *body = self.rows
            lines = [_md_row(header), _md_row(["---"] * len(header))]
            return "\n".join(lines + [_md_row(row) for row in body])
        return _escape(self.text)


def _escape(text: str) -> str:
    return "\\" + text if text.startswith(_MD_LINE_START) else text


def _md_row(cells: list[str]) -> str:
    cells = [c.replace("|", r"\|").replace("\n", "<br>") for c in cells]
    return "| " + " | ".join(cells) + " |"


@dataclass
class ParsedDocument:
    source: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def tables(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == BlockKind.TABLE]

    def to_markdown(self) -> str:
        parts = []
        depth = -1
        for b in self.blocks:
            if b.kind == BlockKind.LIST_ITEM:
                depth = min(b.level, depth + 1)
                parts.append("   " * depth + b.to_markdown())
            else:
                depth = -1
                parts.append(b.to_markdown())
        return "\n\n".join(parts) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseParser(ABC):
    """Парсер конкретного формата"""

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument: ...
