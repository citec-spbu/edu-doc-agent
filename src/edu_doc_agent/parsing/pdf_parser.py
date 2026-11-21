"""PDF (с текстовым слоем) -> блоки в порядке документа

В PDF нет абзацев и стилей — только строки с координатами и шрифтом. Абзацы восстанавливаются
по вёрстке, а превращает их в блоки `parsing.common`, как и для DOCX:

- колонтитулы — строки у верхнего и нижнего края, повторяющиеся на большинстве страниц
  (цифры не учитываются: «Страница 2 из 14»), и одиночные номера страниц — убираются;
- сноски убираются: строки мельче основного шрифта после черты (текстовой «_____» или
  нарисованной короткой линией у левого поля) и абзацы-сноски в конце документа «*(1) …»
  (ГАРАНТ), «<1> …» (КонсультантПлюс);
- новый абзац начинается с отступа первой строки, после увеличенного интервала, при смене
  шрифта (размер, жирность) или после короткой строки, закончившейся «.», «:» или «;»;
  абзац продолжается через границу страниц;
- таблицы с рамками — `pdfplumber.find_tables`; объединённые ячейки определяются по рамкам
  в пределах страницы. Таблица на следующей странице приклеивается как продолжение, только
  если она первая на странице и начинается сверху, предыдущая упирается в низ своей
  страницы, а границы столбцов совпадают; повторённая шапка отбрасывается;
- в ячейках слово, разорванное без дефиса («Информационно-коммуник» + «ационные»), склеивается,
  только если вся строка — одно слово.

Ограничения: таблицы без рамок не распознаются (придут абзацами); две самостоятельные таблицы
с одинаковыми столбцами, из которых первая кончается внизу страницы, а вторая открывает
следующую, склеятся; скан без текстового слоя не читается — `NoTextLayerError` (нужен OCR).
"""

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import pdfplumber
from pdfplumber.page import Page

from edu_doc_agent.parsing.common import (
    BULLET,
    Cell,
    clean_text,
    is_rule,
    paragraph_block,
    table_block,
)
from edu_doc_agent.parsing.models import BaseParser, Block, ParsedDocument

_EDGE_ZONE = 0.1  # доля высоты страницы сверху и снизу, где ищем колонтитулы
_PAGE_NUMBER = re.compile(r"[-–—\s]*\d{1,4}[-–—\s]*")
_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi|полужирн", re.IGNORECASE)
_FOOTNOTE_RULE = re.compile(r"_{5,}")
# сноска в конце документа: «*(1) См. статью…» (ГАРАНТ), «<1> …» (КонсультантПлюс)
_FOOTNOTE_MARK = re.compile(r"^(?:\*\(\d+\)|<\d+>)\s")
# начало новой строки таблицы: «ОПК-4.», «12.»
_ITEM_START = re.compile(r"^(?:[А-ЯЁA-Z]{1,6}-\d|\d+[.)]\s)")
# слово, разорванное без дефиса в узкой ячейке: «Информационно-коммуник» + «ационные»
_CUT_WORD = re.compile(r"-[а-яё]{1,8}$")
_MIN_SIZE = 4.0  # pt: текст мельче — служебный/невидимый (метки на обложке)
_SNAP = 2.0  # pt: координаты рамок таблицы ближе этого считаются одной линией
_COLUMN_TOLERANCE = 3.0  # pt: столбцы продолжения таблицы совпадают с точностью до этого
# таблица продолжается на следующей странице, только если упирается в низ страницы,
# а продолжение начинается сверху (доли высоты страницы)
_TABLE_ENDS_BELOW = 0.7
_TABLE_STARTS_ABOVE = 0.3


class NoTextLayerError(ValueError):
    """В PDF нет текстового слоя (скан) — нужен OCR"""


@dataclass
class _Line:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    bold: bool
    page: int


@dataclass
class _Table:
    page: int
    top: float
    bottom: float
    height: float  # высота страницы
    xs: list[float]  # границы столбцов
    grid: list[list[Cell]] = field(default_factory=list)


class PdfParser(BaseParser):
    def parse(self, path: Path) -> ParsedDocument:
        with pdfplumber.open(path) as pdf:
            if not any(page.chars for page in pdf.pages):
                raise NoTextLayerError(f"{path.name}: нет текстового слоя, нужен OCR")
            pages = [_read_page(page, i) for i, page in enumerate(pdf.pages)]
            heights = [float(page.height) for page in pdf.pages]
        lines = _drop_footnotes(_drop_running_lines([lines for lines, _ in pages], heights))
        tables = [tables for _, tables in pages]
        return ParsedDocument(source=str(path), blocks=_blocks(lines, tables))


# Страница -> строки и таблицы


def _read_page(page: Page, index: int) -> tuple[list[_Line], list[_Table]]:
    found = page.find_tables()
    boxes = [t.bbox for t in found]

    def outside_tables(obj: dict) -> bool:
        cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
        return not any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)

    text_page = page.filter(outside_tables) if boxes else page
    lines = []
    for raw in text_page.extract_text_lines(return_chars=True, strip=True):
        chars = [c for c in raw["chars"] if c["text"].strip() and c["size"] >= _MIN_SIZE]
        if not chars or not raw["text"].strip():
            continue
        lines.append(
            _Line(
                text=raw["text"],
                x0=raw["x0"],
                x1=raw["x1"],
                top=raw["top"],
                bottom=raw["bottom"],
                size=round(median(c["size"] for c in chars), 1),
                bold=sum(_is_bold(c["fontname"]) for c in chars) >= 0.8 * len(chars),
                page=index,
            )
        )
    lines += _footnote_rules(page, index, boxes)
    lines.sort(key=lambda ln: ln.top)
    tables = [_table(page, t, index) for t in found]
    return lines, [t for t in tables if t is not None]


def _footnote_rules(page: Page, index: int, boxes: list[tuple]) -> list[_Line]:
    """Черта перед сносками, нарисованная графикой, -> строка «_____» (как текстовая черта).

    Черта — короткая горизонтальная линия у левого поля в нижней половине страницы, вне таблиц.
    """
    width, height = float(page.width), float(page.height)
    thin_rects = [r for r in page.rects if r["bottom"] - r["top"] <= 1.5]
    rules = []
    for obj in [*page.lines, *thin_rects]:
        if abs(obj["bottom"] - obj["top"]) > 1.5:
            continue  # не горизонтальная
        length = obj["x1"] - obj["x0"]
        cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
        if (
            30 <= length <= 0.6 * width
            and obj["x0"] < 0.4 * width
            and cy > 0.5 * height
            and not any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)
        ):
            rules.append(
                _Line("_____", obj["x0"], obj["x1"], obj["top"], obj["bottom"], 0.0, False, index)
            )
    return rules


def _is_bold(fontname: str) -> bool:
    # Имена шрифтов из ГАРАНТ бывают байтами cp1251 в виде строки: "b'...\\xcf\\xee...'"
    if fontname.startswith(("b'", 'b"')):
        try:
            fontname = ast.literal_eval(fontname).decode("cp1251")
        except (ValueError, SyntaxError, UnicodeDecodeError):
            pass
    return bool(_BOLD_FONT.search(fontname))


def _table(page: Page, found, index: int) -> _Table | None:
    """Таблица pdfplumber -> сетка (рамка ячейки, текст).

    Границы столбцов и строк — все края ячеек; позицию сетки занимает ячейка, рамка которой
    накрывает её центр. Так объединённая ячейка получает один ключ на все свои позиции.
    """
    cells = [c for c in found.cells if c]
    if not cells:
        return None
    xs = _snap(sorted({c[0] for c in cells} | {c[2] for c in cells}))
    ys = _snap(sorted({c[1] for c in cells} | {c[3] for c in cells}))
    texts = {c: _cell_text(page, c) for c in cells}
    grid = []
    for top, bottom in zip(ys, ys[1:], strict=False):
        cy = (top + bottom) / 2
        row = []
        for left, right in zip(xs, xs[1:], strict=False):
            cx = (left + right) / 2
            cell = next((c for c in cells if c[0] <= cx <= c[2] and c[1] <= cy <= c[3]), None)
            # координаты — в пределах страницы: ключ с номером страницы, чтобы одинаково
            # расположенные ячейки на двух страницах склеенной таблицы не стали «объединением»
            row.append(((index, *cell), texts[cell]) if cell else ((index, top, left), ""))
        grid.append(row)
    _, top, _, bottom = found.bbox
    return _Table(page=index, top=top, bottom=bottom, height=float(page.height), xs=xs, grid=grid)


def _snap(values: list[float]) -> list[float]:
    out: list[float] = []
    for v in values:
        if not out or v - out[-1] > _SNAP:
            out.append(v)
    return out


def _cell_text(page: Page, bbox: tuple) -> str:
    """Текст ячейки: строки, перенесённые вёрсткой, склеиваются в одну (как в абзаце)."""
    lines = page.crop(bbox).extract_text_lines(strip=True)
    return clean_text(_join([ln["text"] for ln in lines], narrow=True))


# Колонтитулы и номера страниц


def _drop_running_lines(pages: list[list[_Line]], heights: list[float]) -> list[list[_Line]]:
    def in_edge(line: _Line) -> bool:
        h = heights[line.page]
        return line.top < h * _EDGE_ZONE or line.bottom > h * (1 - _EDGE_ZONE)

    def key(line: _Line) -> str:
        return re.sub(r"\d+", "#", line.text)

    def countable(line: _Line) -> bool:  # черта сносок повторяется на страницах, но нужна
        return in_edge(line) and not _FOOTNOTE_RULE.fullmatch(line.text)

    seen = Counter(k for lines in pages for k in {key(ln) for ln in lines if countable(ln)})
    threshold = max(2, (len(pages) + 1) // 2)
    running = {k for k, n in seen.items() if n >= threshold} if len(pages) > 1 else set()
    # строка колонтитула, повторённая в теле (обложка), тоже служебная
    return [
        [
            ln
            for ln in lines
            if _FOOTNOTE_RULE.fullmatch(ln.text)
            or (key(ln) not in running and not (in_edge(ln) and _PAGE_NUMBER.fullmatch(ln.text)))
        ]
        for lines in pages
    ]


def _drop_footnotes(pages: list[list[_Line]]) -> list[list[_Line]]:
    """Черта «_____» и идущие за ней строки мельче основного шрифта — сноски."""
    sizes = Counter(ln.size for lines in pages for ln in lines)
    if not sizes:
        return pages
    body = sizes.most_common(1)[0][0]
    out = []
    for lines in pages:
        kept, skipping = [], False
        for ln in lines:
            if _FOOTNOTE_RULE.fullmatch(ln.text):
                skipping = True
                continue
            if skipping and ln.size < body - 0.5:
                continue
            skipping = False
            kept.append(ln)
        out.append(kept)
    return out


# Строки и таблицы -> блоки


@dataclass
class _Layout:
    """Вёрстка документа: основной шрифт, левое поле, отступ абзаца, интервал."""

    size: float
    left: float
    indent: float
    right: float
    leading: float

    @classmethod
    def of(cls, pages: list[list[_Line]]) -> "_Layout":
        lines = [ln for page in pages for ln in page]
        size = Counter(ln.size for ln in lines).most_common(1)[0][0]
        body = [ln for ln in lines if abs(ln.size - size) < 0.6] or lines
        left = Counter(round(ln.x0) for ln in body).most_common(1)[0][0]
        indents = [round(ln.x0) for ln in body if left + 8 < ln.x0 < left + 80]
        indent = Counter(indents).most_common(1)[0][0] if indents else left + 20
        right = sorted(ln.x1 for ln in body)[int(len(body) * 0.9)]
        steps = [
            b.top - a.top
            for page in pages
            for a, b in zip(page, page[1:], strict=False)
            if abs(a.size - size) < 0.6 and abs(b.size - size) < 0.6 and b.top > a.top
        ]
        return cls(size, left, indent, right, median(steps) if steps else size * 1.2)

    def indented(self, line: _Line) -> bool:
        return line.x0 > (self.left + self.indent) / 2

    def centered(self, line: _Line) -> bool:
        middle = (self.left + self.right) / 2
        return line.x0 > self.left + 8 and abs((line.x0 + line.x1) / 2 - middle) < 12

    def short(self, line: _Line) -> bool:
        return line.x1 < self.right - 3 * self.size


def _blocks(pages: list[list[_Line]], tables: list[list[_Table]]) -> list[Block]:
    if not any(pages) and not any(tables):
        return []
    layout = _Layout.of(pages) if any(pages) else None
    blocks: list[Block] = []
    paragraph: list[_Line] = []
    open_table: _Table | None = None

    def flush() -> None:
        if paragraph:
            block = _paragraph(paragraph, layout)
            if block is not None:
                blocks.append(block)
            paragraph.clear()

    def close_table() -> None:
        nonlocal open_table
        if open_table is not None:
            block = table_block(open_table.grid)
            if block is not None:
                blocks.append(block)
            open_table = None

    for index, (lines, page_tables) in enumerate(zip(pages, tables, strict=True)):
        items: list[_Line | _Table] = sorted([*lines, *page_tables], key=lambda it: it.top)
        for pos, item in enumerate(items):
            if isinstance(item, _Table):
                flush()
                if _continues(open_table, item, pos, index):
                    _append_continuation(open_table, item)
                else:
                    close_table()
                    open_table = item
                continue
            close_table()
            if paragraph and _starts_paragraph(paragraph[-1], item, layout):
                flush()
            paragraph.append(item)
    flush()
    close_table()
    return blocks


def _continues(previous: _Table | None, table: _Table, position: int, page: int) -> bool:
    """Таблица — продолжение предыдущей: первая на следующей странице, предыдущая упирается
    в низ своей страницы, эта начинается сверху, границы столбцов те же.

    Две самостоятельные таблицы подряд (вторая со своей шапкой) этим условиям не отвечают,
    если первая кончается выше низа страницы или столбцы у них разные.
    """
    return (
        previous is not None
        and previous.page == page - 1
        and position == 0
        and previous.bottom > _TABLE_ENDS_BELOW * previous.height
        and table.top < _TABLE_STARTS_ABOVE * table.height
        and len(previous.xs) == len(table.xs)
        and all(abs(a - b) <= _COLUMN_TOLERANCE for a, b in zip(previous.xs, table.xs, strict=True))
    )


def _append_continuation(table: _Table, part: _Table) -> None:
    """Приклеить продолжение таблицы со следующей страницы.

    Повторённая шапка отбрасывается. Если строка разорвана страницей (в продолжении первый
    столбец пуст, а текст не начинается с нового пункта «ОПК-4.», «12.»), её хвост дописывается
    в последнюю строку таблицы.
    """
    rows = part.grid
    if rows and [t for _, t in rows[0]] == [t for _, t in table.grid[0]]:
        rows = rows[1:]
    if rows and _is_row_tail(table.grid[-1], rows[0]):
        table.grid[-1] = [
            (key, _join([text, more], narrow=True) if more else text)
            for (key, text), (_, more) in zip(table.grid[-1], rows[0], strict=True)
        ]
        rows = rows[1:]
    table.grid.extend(rows)
    table.page, table.bottom, table.height = part.page, part.bottom, part.height


def _is_row_tail(last: list[Cell], first: list[Cell]) -> bool:
    texts = [t for _, t in first]
    filled = [(i, t) for i, t in enumerate(texts) if t]
    return (
        bool(filled)
        and not texts[0]
        and all(last[i][1] for i, _ in filled)
        and not _ITEM_START.match(filled[0][1])
    )


def _starts_paragraph(prev: _Line, line: _Line, layout: _Layout) -> bool:
    if abs(prev.size - line.size) > 0.6 or prev.bold != line.bold:
        return True
    leading = layout.leading * line.size / layout.size  # крупный шрифт — шире интервал
    if line.page == prev.page and line.top - prev.top > 1.5 * leading:
        return True
    if layout.centered(prev) and layout.centered(line) or line.size >= layout.size + 2:
        return False  # многострочный заголовок: по центру или крупным шрифтом
    if layout.indented(line):
        return True
    ended = prev.text.endswith((".", ":", ";"))
    # «- с требованиями…» в начале строки после незаконченной — тире, а не маркер списка
    return ended and (layout.short(prev) or bool(BULLET.match(line.text)))


def _paragraph(lines: list[_Line], layout: _Layout) -> Block | None:
    raw = _join([ln.text for ln in lines])
    if is_rule(raw):
        return None
    text = clean_text(raw)
    if not text or _FOOTNOTE_MARK.match(text):  # сноска в конце документа
        return None
    bold = all(ln.bold for ln in lines) or lines[0].size >= layout.size + 2
    return paragraph_block(text, bold=bold)


def _join(parts: list[str], narrow: bool = False) -> str:
    """Склейка строк абзаца: перенос «научно-» + «исследовательский» — без пробела, дефис
    остаётся (в нормативных текстах это почти всегда составное слово).

    В узких ячейках (`narrow`) слово бывает разорвано без дефиса: «Информационно-коммуник» +
    «ационные». Так рвётся только слово шире ячейки, поэтому разрыв признаём, лишь когда вся
    строка — одно слово; «в очно-заочной» + «форме обучения» склеиваются через пробел.
    """
    out, last = "", ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        glue = out.endswith("-") and len(out) > 1 and out[-2].isalpha()
        glue = glue or (narrow and " " not in last and bool(_CUT_WORD.search(last)))
        if glue and part[:1].islower():
            out += part
        else:
            out = f"{out} {part}" if out else part
        last = part
    return out
