"""Исправленный документ: правки изменения переносятся в исходный DOCX (ADR-0003, «Генерация»).

Новый документ собирается из исходного файла, а не из шаблона: структура, оформление и
нетронутый текст остаются как были. Место правки находит цитата источника значения
(`SourceRef.quote` из извлечения): абзац или строка таблицы. Правку, место которой однозначно
не найдено, код не угадывает: она попадает в список «внести вручную».
"""

import io
import re
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from docx import Document
from docx.table import Table, _Row
from docx.text.paragraph import Paragraph
from pydantic import BaseModel

from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.source import SourceRef

_PATH_PART = re.compile(r"^\d+$")


@dataclass(frozen=True)
class Patch:
    label: str  # что правится, для пользователя: «credits», «competences/5»
    op: str  # set / remove / append
    old: Any
    new: Any
    quote: str | None  # фрагмент документа со значением
    reason: str | None = None  # правка не переносится: почему её вносить вручную


@dataclass
class PatchResult:
    content: bytes
    applied: list[str]
    manual: list[str]  # правки, которые нужно внести в документ вручную, с причиной


def source_for(data: BaseModel, path: str) -> SourceRef | None:
    """Источник значения по пути правки: ближайший `source` по пути или `sources[поле]`."""
    parts = path.split("/")
    sources = getattr(data, "sources", None)
    found = sources.get(parts[0]) if isinstance(sources, dict) else None
    value: Any = data
    for part in parts:
        try:
            value = value[int(part)] if _PATH_PART.match(part) else getattr(value, part)
        except (AttributeError, IndexError, KeyError, TypeError):
            break
        source = getattr(value, "source", None)
        if isinstance(source, SourceRef):
            found = source
    return found


def _text(value: Any) -> list[str]:
    """Варианты записи значения в документе: 6.0 → «6»; 4.5 → «4.5», «4,5»."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int | float):
        text = f"{value:g}"
        return [text, text.replace(".", ",")] if "." in text else [text]
    return [str(value)] if isinstance(value, str) and value.strip() else []


def _paragraphs(doc) -> Iterator[Paragraph]:
    yield from doc.paragraphs
    for table in _tables(doc):
        for row in table.rows:
            for cell in _cells(row):
                yield from cell.paragraphs


def _tables(container) -> Iterator[Table]:
    for table in container.tables:
        yield table
        for row in table.rows:
            for cell in _cells(row):
                yield from _tables(cell)


def _cells(row: _Row) -> list:
    """Ячейки строки без повторов объединённых ячеек."""
    out, seen = [], set()
    for cell in row.cells:
        if id(cell._tc) not in seen:
            seen.add(id(cell._tc))
            out.append(cell)
    return out


def _row_cells(quote: str) -> list[str] | None:
    """Ячейки строки таблицы из цитаты Markdown «| ИТОГО | 56 | | 2 |»."""
    if not quote.lstrip().startswith("|"):
        return None
    return [normalize(c) for c in quote.strip().strip("|").split("|")]


def _find_row(doc, cells: list[str]) -> _Row | None:
    want = [c for c in cells if c]
    for table in _tables(doc):
        for row in table.rows:
            have = [normalize(c.text) for c in _cells(row)]
            if [c for c in have if c] == want:
                return row
    return None


def _find_paragraph(doc, quote: str) -> Paragraph | None:
    needle = normalize(quote.strip().strip("|"))
    if not needle:
        return None
    return next((p for p in _paragraphs(doc) if needle in normalize(p.text)), None)


def _replace(paragraph: Paragraph, old: str, new: str) -> bool:
    """Последнее вхождение `old` в абзаце → `new`; оформление прогона сохраняется, если
    значение целиком в одном прогоне."""
    for run in reversed(paragraph.runs):
        if old in run.text:
            head, _, tail = run.text.rpartition(old)
            run.text = head + new + tail
            return True
    text = paragraph.text
    if old not in text:
        return False
    head, _, tail = text.rpartition(old)
    runs = paragraph.runs
    runs[0].text = head + new + tail
    for run in runs[1:]:
        run.text = ""
    return True


def _token(value: str, text: str) -> bool:
    """`value` в тексте целым словом: «УК-12» — да в «УК-12 способен», нет в «УК-12.1»."""
    return re.search(rf"(?<![\w.-]){re.escape(value)}(?![\w-]|\.\d)", text) is not None


def _element(place: _Row | Paragraph | list[Paragraph]):
    if isinstance(place, list):
        return place[0]._p
    return place._tr if isinstance(place, _Row) else place._p


def _delete(element) -> None:
    element.getparent().remove(element)


def _lines(text: str) -> list[str]:
    return [normalize(line) for line in text.splitlines() if line.strip()]


def _find_block(doc, text: str) -> list[Paragraph] | None:
    """Подряд идущие абзацы, которые в точности составляют многострочный текст (раздел целей)."""
    lines = _lines(text)
    body = [p for p in doc.paragraphs if p.text.strip()]
    for i in range(len(body) - len(lines) + 1):
        if [normalize(p.text) for p in body[i : i + len(lines)]] == lines:
            return body[i : i + len(lines)]
    return None


def _locate(doc, patch: Patch) -> _Row | Paragraph | list[Paragraph] | str:
    """Место правки в документе или причина, почему его нет."""
    if patch.reason:
        return patch.reason
    if patch.op == "set" and isinstance(patch.old, str) and len(_lines(patch.old)) > 1:
        block = _find_block(doc, patch.old)
        return block or "многострочный текст в документе не совпадает с данными: замените вручную"
    if patch.op == "append":
        return "новый элемент добавляется вручную: место в документе не определено"
    if not patch.quote:
        return "в данных нет места значения в документе"
    cells = _row_cells(patch.quote)
    place = (_find_row(doc, cells) if cells else None) or _find_paragraph(doc, patch.quote)
    return place if place is not None else "место значения в документе не найдено"


def _write_block(block: list[Paragraph], text: str) -> None:
    """Новый текст вместо абзацев: первый абзац сохраняет оформление, лишние удаляются,
    недостающие добавляются копией первого."""
    lines = [line for line in text.splitlines() if line.strip()] or [""]
    first = block[0]
    _set(first, lines[0])
    anchor = first._p
    for line in lines[1:]:
        copy = deepcopy(first._p)
        anchor.addnext(copy)
        anchor = copy
        _set(Paragraph(copy, first._parent), line)
    for paragraph in block[1:]:
        _delete(paragraph._p)


def _set(paragraph: Paragraph, text: str) -> None:
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    runs[0].text = text
    for run in runs[1:]:
        run.text = ""


def _apply(place: _Row | Paragraph | list[Paragraph], patch: Patch) -> str | None:
    """Правка на найденном месте; None — перенесена, иначе причина, почему нужно вручную."""
    if isinstance(place, list):
        _write_block(place, str(patch.new))
        return None
    row = place if isinstance(place, _Row) else None
    paragraph = place if isinstance(place, Paragraph) else None
    if patch.op == "remove":
        _delete(row._tr if row is not None else paragraph._p)
        return None
    olds, new = _text(patch.old), _text(patch.new)
    if not olds or not new:
        return "значение сложного вида переносится вручную"
    if row is not None:
        cells = _cells(row)
        hits = [c for c in cells if c.text.strip() in olds]
        if hits and (len(hits) == 1 or patch.label.endswith("credits")):
            cell = hits[-1]  # з.е. — последний столбец строки таблицы трудоёмкости
            return (
                None if _replace(cell.paragraphs[0], cell.text.strip(), new[0]) else "не заменено"
            )
        # значение — часть текста ячейки («УК-12 способен …»): ячейка, где оно одно целым
        # словом (не «УК-12.1» индикатора)
        within = [(c, p, o) for c in cells for p in c.paragraphs for o in olds if _token(o, p.text)]
        if len({id(c._tc) for c, _, _ in within}) == 1 and not hits:
            _, paragraph, old = within[0]
            return None if _replace(paragraph, old, new[0]) else "не заменено"
        if not within and any(_token(new[0], c.text) for c in cells):
            return None  # уже внесено другой правкой: тот же код в той же строке
        return "значение в строке таблицы не определяется однозначно"
    for old in olds:
        if _replace(paragraph, old, new[0]):
            return None
    if _token(new[0], paragraph.text):
        return None  # уже внесено другой правкой
    return "прежнее значение в найденном месте документа не найдено"


def patch_docx(content: bytes, patches: list[Patch]) -> PatchResult:
    doc = Document(io.BytesIO(content))
    applied, manual = [], []
    # места находятся по исходному тексту до правок: вторая правка той же строки таблицы
    # не теряет её из-за первой; удаления — после замен
    places = [(p, _locate(doc, p)) for p in sorted(patches, key=lambda p: p.op == "remove")]
    removed: set[int] = set()  # несколько удалений одной строки удаляют её один раз
    for patch, place in places:
        element = None if isinstance(place, str) else _element(place)
        if patch.op == "remove" and element is not None and id(element) in removed:
            applied.append(patch.label)
            continue
        reason = place if isinstance(place, str) else _apply(place, patch)
        if patch.op == "remove" and reason is None and element is not None:
            removed.add(id(element))
        if reason is None:
            applied.append(patch.label)
        else:
            manual.append(f"{patch.label}: {reason}")
    out = io.BytesIO()
    doc.save(out)
    return PatchResult(out.getvalue(), applied, manual)
