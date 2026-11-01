"""Разбор Markdown: таблицы с привязкой к разделу и очистка текста ячеек."""

import re
from dataclasses import dataclass

_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


_SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?(?:\s|$)")


@dataclass
class MdSection:
    number: str | None  # «3.1.3» из заголовка «3.1.3. Методика…»; None, если номера нет
    heading: str
    text: str  # тело раздела без таблиц; номера строк совпадают с исходным файлом
    line: int  # 1-based номер строки заголовка
    level: int = 0  # число «#» в заголовке
    full_text: str = ""  # то же, но строки таблиц развёрнуты в текст «ячейка | ячейка»


@dataclass
class MdTable:
    rows: list[list[str]]
    row_lines: list[int]  # 1-based номер строки md для каждой строки rows
    section: str | None


def clean_text(text: str) -> str:
    """Убирает `<br>`-разметку и нормализует пробелы и тире.

    `<br>` (так парсер docx записывает перенос строки в ячейке) превращается в «\\n».
    Сами коды (в том числе нумерацию ПКП) не меняем — только символы-тире и пробелы.
    """
    text = _BR_RE.sub("\n", text)
    text = text.replace(" ", " ").translate(_DASHES)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\s*\n\s*", "\n", text).strip()


def split_row(line: str) -> list[str]:
    s = line.strip()
    s = s.removeprefix("|")
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)]


def _is_separator(cells: list[str]) -> bool:
    return all(not c or _SEPARATOR_CELL_RE.match(c) for c in cells) and any(cells)


def parse_tables(text: str) -> list[MdTable]:
    """Находит все pipe-таблицы; строки-разделители пропускает."""
    tables: list[MdTable] = []
    section: str | None = None
    rows: list[list[str]] = []
    row_lines: list[int] = []
    table_section: str | None = None

    def flush() -> None:
        nonlocal rows, row_lines
        if rows:
            tables.append(MdTable(rows=rows, row_lines=row_lines, section=table_section))
        rows, row_lines = [], []

    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("|"):
            if not rows:
                table_section = section
            cells = split_row(line)
            if not _is_separator(cells):
                rows.append(cells)
                row_lines.append(lineno)
            continue
        flush()
        m = _HEADING_RE.match(line)
        if m:
            section = clean_text(m.group(2))
    flush()
    return tables


def table_row_text(line: str) -> str:
    """Строка таблицы как обычный текст: ячейки через « | », без пустых и подряд повторённых
    (так объединённая ячейка попадает в md несколько раз). Разделитель шапки даёт пустую строку."""
    cells = split_row(line)
    if _is_separator(cells):
        return ""
    texts = [" ".join(clean_text(c).split()) for c in cells]
    kept = [c for i, c in enumerate(texts) if c and (i == 0 or c != texts[i - 1])]
    return " | ".join(kept)


def split_sections(text: str) -> list[MdSection]:
    """Делит md на разделы по заголовкам; строки таблиц в `text` заменяются пустыми.

    Пустые строки сохраняют нумерацию: смещение в `MdSection.text` легко перевести
    в номер строки исходного файла (`line + text.count("\\n", 0, offset) + 1`).
    В `full_text` таблицы развёрнуты в текст, строк по-прежнему столько же.
    """
    sections: list[MdSection] = []
    current: tuple[str, int, int] | None = None
    body: list[str] = []
    full: list[str] = []

    def flush() -> None:
        if current is None:
            return
        heading, line, level = current
        m = _SECTION_NUMBER_RE.match(heading)
        sections.append(
            MdSection(
                number=m.group(1) if m else None,
                heading=heading,
                text="\n".join(body),
                line=line,
                level=level,
                full_text="\n".join(full),
            )
        )

    for lineno, line in enumerate(text.splitlines(), start=1):
        m = _HEADING_RE.match(line)
        if m:
            flush()
            current, body, full = (clean_text(m.group(2)), lineno, len(m.group(1))), [], []
        elif current is not None:
            if line.lstrip().startswith("|"):
                body.append("")
                full.append(table_row_text(line))
            else:
                body.append(clean_text(line))
                full.append(body[-1])
    flush()
    return sections
