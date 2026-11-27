"""Семантические таблицы учебного плана поверх структурированного XLSX-парсера.

Один лист может содержать титул, компетенции, нагрузку и ГИА. Сохраняем координаты
и объединения до выделения этих таблиц; предметные модели и проверки общие с DOCX.
"""

import re

from edu_doc_agent.extraction.curriculum import (
    CurriculumPlan,
    _codes,
    _entry_columns,
    _extract_attestation,
    _extract_competences,
    _extract_entries,
    _extract_title,
    _find_plan_header,
    _number,
    _one_line,
    _plan_headers,
    _ref,
    check_plan,
)
from edu_doc_agent.extraction.md_utils import MdTable, clean_text
from edu_doc_agent.parsing import Block, BlockKind, CellSpan, ParsedDocument


def _kind(row: list[str]) -> str | None:
    cells = [_one_line(clean_text(c)).lower() for c in row]

    def has(term: str) -> bool:
        return any(c.startswith(term) for c in cells)

    if has("код компетенции") and has("наименование") and not has("код блока"):
        return "competences"
    if _attestation_columns(row) is not None:
        return "attestation"
    if has("код блока") and has("аудиторная работа"):
        return "entries"
    return None


def _attestation_columns(row: list[str]) -> list[int] | None:
    cells = [_one_line(clean_text(c)).lower() for c in row]
    cols = [
        next((i for i, c in enumerate(cells) if c.startswith(prefix)), None)
        for prefix in (
            "код блока",
            "трудо",
            "форма и наименование процедуры",
            "перечень кодов компетенций",
        )
    ]
    return None if None in cols else cols


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
        if (
            _one_line(first).lower().startswith("код блока")
            and any(_one_line(c).lower().startswith("форма и наименование процедуры") for c in row)
            and _attestation_columns(row) is None
        ):
            plan.warnings.append(
                f"{sheet}, строка {numbers[idx]}: неполная шапка итоговой аттестации"
            )
        if re.match(r"^раздел\s+\d", first, re.I):
            flush(idx)
            start, kind = idx, "title"
        elif (new_kind := _kind(row)) and new_kind != kind:
            flush(idx)
            start, kind = idx, new_kind
    flush(len(block.rows))
    return result


def _merge_translations(
    table: MdTable, name_col: int, block_col: int = 0, columns: set[int] | None = None
) -> MdTable:
    """Перевод связан объединением предметного поля, а не только общего блока.

    Одинаковый код блока сам по себе не повод склеить соседние дисциплины.
    Остальные ячейки продолжения внутри таблицы должны быть пустыми или повторять
    исходные. Заметки за пределами её столбцов остаются только в цитате источника.
    """
    merged_cells: dict[tuple[int, int], tuple[int, int]] = {}
    merged_names: dict[int, int] = {}
    for span in table.spans or []:
        if span.row_span > 1:
            for row in range(span.row, span.row + span.row_span):
                if span.col == name_col:
                    merged_names[row] = span.row + span.row_span
                elif span.col != block_col:
                    merged_cells[row, span.col] = (span.row, span.row + span.row_span)
    rows, lines, quotes, end_lines = [], [], [], []
    anchor = -1
    for idx, row in enumerate(table.rows):
        relevant = columns if columns is not None else set(range(len(row)))
        name = row[name_col]
        is_english = bool(re.search(r"[a-zA-Z]{2,}", name)) and not re.search(r"[а-яА-ЯёЁ]", name)
        same_cells = rows and all(
            not row[col] or row[col] == rows[-1][col] for col in relevant if col != name_col
        )
        shared = {
            col
            for col in relevant
            if (idx, col) in merged_cells
            and merged_cells[idx, col] == merged_cells.get((anchor, col))
        }
        continuation = shared and all(
            not row[col] or col in {name_col, block_col} or col in shared for col in relevant
        )
        repeated_name = rows and idx < merged_names.get(anchor, 0) and name == rows[-1][name_col]
        translation = rows and is_english and continuation and not re.match(r"^\[\d+\]", name)
        if same_cells and (repeated_name or translation):
            if translation and not repeated_name:
                rows[-1][name_col] += "\n" + name
            quotes[-1] += "\n" + table.row_quotes[idx]
            end_lines[-1] = table.row_lines[idx]
        else:
            rows.append(row.copy())
            lines.append(table.row_lines[idx])
            quotes.append(table.row_quotes[idx])
            end_lines.append(table.row_end_lines[idx] if table.row_end_lines else None)
            anchor = idx
    return MdTable(
        rows=rows,
        row_lines=lines,
        section=table.section,
        row_quotes=quotes,
        row_end_lines=end_lines,
    )


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


def _choice_groups(table: MdTable, cols: dict[str, int]) -> MdTable:
    """Многострочная шапка выбора задаёт группу, а не несколько фиктивных дисциплин."""
    name_col = cols["name"]
    rows, lines, quotes, end_lines = [], [], [], []
    choice: int | None = None
    for idx, row in enumerate(table.rows):
        name = row[name_col]
        if name.lower().startswith("дисциплины по выбору"):
            choice = len(rows)
            rows.append(["Блок дисциплин по выбору"] + [""] * (len(row) - 1))
            lines.append(table.row_lines[idx])
            quotes.append(table.row_quotes[idx])
            end_lines.append(table.row_end_lines[idx])
            continue
        if choice is not None:
            choice_limit = "credits" in cols and re.match(
                r"^(?:от|до)\s+\d", row[cols["credits"]], re.I
            )
            if not choice_limit and (
                re.match(r"^\[\d+\]", name)
                or (
                    row[cols.get("block", 0)].lower().startswith("блок")
                    and name
                    and (
                        ("competences" in cols and _codes(row[cols["competences"]]))
                        or (
                            "control" in cols
                            and re.search(
                                r"\b(?:зач[её]т|экзамен|текущий контроль)\b",
                                row[cols["control"]],
                                re.I,
                            )
                        )
                    )
                )
                or any(_number(c) is not None for c in row if c != name)
                or re.search(r"семестр|год обучения|часть|профиль|траектория", row[0], re.I)
            ):
                choice = None
            else:
                if name and not name.startswith("(") and re.search(r"[а-яА-ЯёЁ]", name):
                    rows[choice][0] += ": " + name.split("\n")[0]
                continue
        rows.append(row)
        lines.append(table.row_lines[idx])
        quotes.append(table.row_quotes[idx])
        end_lines.append(table.row_end_lines[idx])
    return MdTable(
        rows=rows,
        row_lines=lines,
        section=table.section,
        row_quotes=quotes,
        row_end_lines=end_lines,
    )


def _prepare_entries(table: MdTable) -> MdTable:
    """Обработка переводов и групп в раскладке каждой повторной шапки."""
    starts = {0}
    for idx in _plan_headers(table):
        start = idx - 1 if idx > 0 and _kind(table.rows[idx - 1]) == "entries" else idx
        starts.add(start)
    boundaries = sorted(starts) + [len(table.rows)]
    result = MdTable(rows=[], row_lines=[], section=table.section)
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        chunk = MdTable(
            rows=table.rows[start:end],
            row_lines=table.row_lines[start:end],
            section=table.section,
            row_quotes=table.row_quotes[start:end],
            spans=[
                CellSpan(s.row - start, s.col, min(s.row_span, end - s.row), s.col_span)
                for s in table.spans or []
                if start <= s.row < end
            ],
        )
        header = _find_plan_header(chunk)
        if header is not None:
            cols, contact, self_study = _entry_columns(header[1])
            if "name" in cols:
                chunk = _choice_groups(
                    _merge_translations(
                        chunk,
                        cols["name"],
                        cols.get("block", 0),
                        set(cols.values()) | contact.keys() | self_study.keys(),
                    ),
                    cols,
                )
        result.rows.extend(chunk.rows)
        result.row_lines.extend(chunk.row_lines)
        result.row_quotes.extend(chunk.row_quotes)
        result.row_end_lines.extend(chunk.row_end_lines or [None] * len(chunk.rows))
    return result


def extract_xlsx_plan(document: ParsedDocument, file: str) -> CurriculumPlan:
    plan = CurriculumPlan(file=file, warnings=list(document.warnings))
    sheet = ""
    seen_sections: dict[tuple, str] = {}
    for block in document.blocks:
        if block.kind == BlockKind.HEADING:
            sheet = block.text
        elif block.kind == BlockKind.TABLE:
            for kind, table in _tables(block, sheet, plan):
                if kind != "title":
                    fingerprint = (
                        kind,
                        tuple(tuple(row) for row in table.rows),
                        tuple(sorted((s.row, s.col, s.row_span, s.col_span) for s in table.spans)),
                    )
                    previous = seen_sections.setdefault(fingerprint, sheet)
                    if previous != sheet:
                        plan.warnings.append(
                            f"Листы «{previous}» и «{sheet}»: полностью совпадает раздел {kind}; "
                            "проверьте, какой лист рабочий"
                        )
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
                    table = _prepare_entries(table)
                    if not _extract_entries(table, file, plan):
                        plan.warnings.append(f"{sheet}: не распознана шапка нагрузки дисциплин")
                elif kind == "attestation":
                    cols = _attestation_columns(table.rows[0])
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
                        row_end_lines=[
                            table.row_end_lines[0],
                            *(table.row_end_lines[i] for i in keep),
                        ],
                    )
                    _extract_attestation(table, file, plan)
    if not plan.competences:
        plan.warnings.append("не найдена таблица компетенций (1.1)")
    if not plan.entries:
        plan.warnings.append("не найдена таблица дисциплин (раздел 2)")
    check_plan(plan)
    return plan
