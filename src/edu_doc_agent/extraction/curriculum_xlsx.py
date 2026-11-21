"""Семантические таблицы учебного плана поверх структурированного XLSX-парсера.

Один лист может содержать титул, компетенции, нагрузку и ГИА. Сохраняем координаты
и объединения до выделения этих таблиц; предметные модели и проверки общие с DOCX.
"""

import re

from edu_doc_agent.extraction.curriculum import (
    CurriculumPlan,
    _extract_attestation,
    _extract_competences,
    _extract_entries,
    _extract_title,
    _number,
    _ref,
    check_plan,
)
from edu_doc_agent.extraction.md_utils import MdTable, clean_text
from edu_doc_agent.parsing import Block, BlockKind, CellSpan, ParsedDocument


def _kind(row: list[str]) -> str | None:
    text = " ".join(row).lower()
    if "код компетенции" in text and "наименование" in text and "код блока" not in text:
        return "competences"
    if "итоговой аттестации" in text and "компетенций" in text:
        return "attestation"
    if "код блока" in text and "аудиторная работа" in text:
        return "entries"
    return None


def _tables(block: Block, sheet: str, plan: CurriculumPlan) -> list[tuple[str, MdTable]]:
    """Разделы внутри листа и повторные шапки; между шапками контекст не сбрасывается."""
    numbers = block.source_rows
    if len(numbers) != len(block.rows):
        plan.warnings.append(
            f"{sheet}: нет исходных координат строк XLSX, использован индекс таблицы"
        )
        numbers = list(range(1, len(block.rows) + 1))
    result = []
    start, kind = 0, "title"

    def flush(end: int) -> None:
        if end <= start:
            return
        spans = [
            CellSpan(s.row - start, s.col, min(s.row_span, end - s.row), s.col_span)
            for s in block.spans
            if start <= s.row < end
        ]
        rows = [[clean_text(c) for c in row] for row in block.rows[start:end]]
        result.append(
            (
                kind,
                MdTable(
                    rows=rows,
                    row_lines=numbers[start:end],
                    section=sheet,
                    spans=spans,
                    row_quotes=[" | ".join(row) for row in block.rows[start:end]],
                ),
            )
        )

    for idx, row in enumerate(block.rows):
        first = clean_text(row[0]) if row else ""
        if re.match(r"^раздел\s+\d", first, re.I):
            flush(idx)
            start, kind = idx, "title"
        elif (new_kind := _kind(row)) and new_kind != kind:
            flush(idx)
            start, kind = idx, new_kind
    flush(len(block.rows))
    return result


def _merge_translations(table: MdTable, name_col: int) -> MdTable:
    """Английская строка относится к записи только при общем вертикальном объединении.

    Одинаковый код блока сам по себе не повод склеить соседние дисциплины.
    Остальные ячейки продолжения должны быть пустыми или повторять исходные.
    """
    merged_rows: dict[int, int] = {}
    merged_names: dict[int, int] = {}
    for span in table.spans or []:
        if span.row_span > 1:
            for row in range(span.row, span.row + span.row_span):
                target = merged_names if span.col == name_col else merged_rows
                target[row] = max(target.get(row, 0), span.row + span.row_span)
    rows, lines, quotes = [], [], []
    anchor = -1
    for idx, row in enumerate(table.rows):
        name = row[name_col]
        is_english = bool(re.search(r"[a-zA-Z]{2,}", name)) and not re.search(r"[а-яА-ЯёЁ]", name)
        same_cells = rows and all(
            not c or c == rows[-1][col] for col, c in enumerate(row) if col != name_col
        )
        if same_cells and idx < merged_names.get(anchor, 0) and name == rows[-1][name_col]:
            continue
        if (
            rows
            and is_english
            and idx < merged_rows.get(anchor, 0)
            and not re.match(r"^\[\d+\]", name)
            and same_cells
        ):
            rows[-1][name_col] += "\n" + name
        else:
            rows.append(row.copy())
            lines.append(table.row_lines[idx])
            quotes.append(table.row_quotes[idx])
            anchor = idx
    return MdTable(rows=rows, row_lines=lines, section=table.section, row_quotes=quotes)


def _title(table: MdTable, plan: CurriculumPlan) -> None:
    _extract_title("", [table], plan)
    if plan.program is not None:
        return
    for idx, row in enumerate(table.rows):
        if not row or not row[0].lower().startswith("по уровню") or idx == 0:
            continue
        start = idx - 1
        previous = " ".join(c for c in table.rows[start] if c)
        if (
            re.search(r"[a-zA-Z]{2,}", previous)
            and not re.search(r"[а-яА-ЯёЁ]", previous)
            and start > 0
        ):
            start -= 1
        plan.program = " ".join(c for r in table.rows[start:idx] for c in r if c)
        plan.sources["program"] = _ref(table, start, plan.file)
        return


def _choice_groups(table: MdTable, name_col: int) -> MdTable:
    """Многострочная шапка выбора задаёт группу, а не несколько фиктивных дисциплин."""
    rows, lines, quotes = [], [], []
    choice: int | None = None
    for idx, row in enumerate(table.rows):
        name = row[name_col]
        if name.lower().startswith("дисциплины по выбору"):
            choice = len(rows)
            rows.append(["Блок дисциплин по выбору"] + [""] * (len(row) - 1))
            lines.append(table.row_lines[idx])
            quotes.append(table.row_quotes[idx])
            continue
        if choice is not None:
            if (
                re.match(r"^\[\d+\]", name)
                or any(_number(c) is not None for c in row if c != name)
                or re.search(r"семестр|год обучения|часть|профиль|траектория", row[0], re.I)
            ):
                choice = None
            else:
                if name and not name.startswith("("):
                    rows[choice][0] += ": " + name.split("\n")[0]
                continue
        rows.append(row)
        lines.append(table.row_lines[idx])
        quotes.append(table.row_quotes[idx])
    return MdTable(rows=rows, row_lines=lines, section=table.section, row_quotes=quotes)


def extract_xlsx_plan(document: ParsedDocument, file: str) -> CurriculumPlan:
    plan = CurriculumPlan(file=file, warnings=list(document.warnings))
    sheet = ""
    for block in document.blocks:
        if block.kind == BlockKind.HEADING:
            sheet = block.text
        elif block.kind == BlockKind.TABLE:
            for kind, table in _tables(block, sheet, plan):
                if kind == "title":
                    _title(table, plan)
                elif kind == "competences":
                    _extract_competences(table, file, plan)
                elif kind == "entries":
                    name_col = next(
                        (i for i, c in enumerate(table.rows[0]) if "наименование" in c.lower()),
                        None,
                    )
                    if name_col is None:
                        plan.warnings.append(f"{sheet}: в шапке дисциплин нет наименования")
                        continue
                    table = _choice_groups(_merge_translations(table, name_col), name_col)
                    if not _extract_entries(table, file, plan):
                        plan.warnings.append(f"{sheet}: не распознана шапка нагрузки дисциплин")
                elif kind == "attestation":
                    header = table.rows[0]
                    cols = [
                        next((i for i, c in enumerate(header) if term in c.lower()), None)
                        for term in ("код блока", "трудо", "процедуры", "компетенций")
                    ]
                    if None in cols:
                        plan.warnings.append(f"{sheet}: неполная шапка итоговой аттестации")
                        continue
                    table = _merge_translations(table, cols[2])
                    # ГИА занимает другую сетку, чем дисциплины на том же листе.
                    table.rows = [[row[c] for c in cols] for row in table.rows]
                    keep = [
                        i
                        for i, row in enumerate(table.rows)
                        if not row[0].lower().startswith("код блока")
                    ]
                    table = MdTable(
                        rows=[table.rows[0], *(table.rows[i] for i in keep)],
                        row_lines=[table.row_lines[0], *(table.row_lines[i] for i in keep)],
                        section=table.section,
                        row_quotes=[table.row_quotes[0], *(table.row_quotes[i] for i in keep)],
                    )
                    _extract_attestation(table, file, plan)
    if not plan.competences:
        plan.warnings.append("не найдена таблица компетенций (1.1)")
    if not plan.entries:
        plan.warnings.append("не найдена таблица дисциплин (раздел 2)")
    check_plan(plan)
    return plan
