"""Извлечение сущностей из Markdown РПД/РПП (вывод docx-парсера) регулярными выражениями.

Формат входа — как у `parsing.ParsedDocument.to_markdown()`: заголовки `#`, таблицы pipe-формата
с прямоугольной сеткой (объединённые ячейки дают пустые), перенос строки в ячейке — `<br>`.

Что берём:
- титул: название, рег. номер, трудоёмкость, тип документа;
- таблица 1.3: компетенции, индикаторы, результаты обучения;
- таблица 3.1.4: индикатор → контрольно-измерительные материалы (оценочные средства);
- таблица форм контроля: текущий контроль и промежуточная аттестация по семестрам;
- цели, активные формы, часы по темам, шкалы и билеты, КИМ, разработчики, направление и ООП,
  данные РПП (вид, профстандарты, компетенции по группам) — см. `sections.py`;
- свободный текст (пререквизиты, формы контроля, трудоёмкость, семестры) — см. `freetext.py`.

Всё, что не удалось разобрать правилами, не выдумываем, а пропускаем и пишем в `warnings`
(кандидаты на LLM-извлечение).
"""

import re
from pathlib import Path

from edu_doc_agent.extraction.freetext import extract_free_text
from edu_doc_agent.extraction.llm_extract import extract_with_llm
from edu_doc_agent.extraction.md_utils import MdTable, clean_text, parse_tables
from edu_doc_agent.extraction.models import (
    AssessmentLink,
    Attestation,
    Competence,
    ControlForm,
    Indicator,
    LearningOutcome,
    OutcomeKind,
    RPDExtraction,
    SourceRef,
)
from edu_doc_agent.extraction.patterns import (
    ATTESTATION_KIND_RE,
    BULLET_RE,
    COMPETENCE_CELL_RE,
    COMPETENCE_RE,
    CREDITS_RE,
    INDICATOR_RE,
    INDICATOR_SUFFIX_RE,
    INTERMEDIATE_RE,
    KIND_BY_LETTER,
    KIND_CANONICAL,
    KIND_HEADING_RE,
    KIND_SPLIT_RE,
    KIND_WORD_RE,
    LANGUAGE_LINE_RE,
    LEVEL_RE,
    NAME_THEN_CODE_RE,
    OUTCOME_RE,
    REG_NUMBER_RE,
    SEMESTER_ROW_RE,
    STUDY_FORM_RE,
    SUFFIX_OUTCOME_RE,
    TRAJECTORY_ROW_RE,
    code_key,
)
from edu_doc_agent.extraction.sections import extract_sections
from edu_doc_agent.extraction.workload_table import check_workload, extract_workload_table
from edu_doc_agent.llm import LLMClient

_TITLE_SCAN_LINES = 40
_HEADING_MARK_RE = re.compile(r"^#+\s*")


def extract_rpd_file(path: Path, llm: LLMClient | None = None) -> RPDExtraction:
    return extract_rpd(path.read_text(encoding="utf-8"), file=path.name, llm=llm)


def extract_rpd(text: str, file: str, llm: LLMClient | None = None) -> RPDExtraction:
    """Правила и Yargy всегда; смысловые поля через LLM — только если передан клиент.

    Без `llm` работа полностью офлайн и детерминирована. С `llm` добавляется `result.llm`
    (пререквизиты, активные формы, правила оценивания с цитатами), а его предупреждения и
    расхождения с правилами попадают в `result.warnings`.
    """
    result = RPDExtraction(file=file)
    _extract_title_block(text, result)

    tables = parse_tables(text)
    outcome_tables = [t for t in tables if "планируемые результаты" in _header(t)]
    assessment_tables = [
        t for t in tables if "измерительные" in _header(t) and t not in outcome_tables
    ]
    for table in outcome_tables:
        _extract_outcome_table(table, file, result)
    for table in assessment_tables:
        _extract_assessment_table(table, file, result)
    for table in tables:
        _extract_control_forms(table, file, result)
        extract_workload_table(table, file, result)
    check_workload(result)
    extract_free_text(text, file, result)
    extract_sections(text, tables, file, result)

    for comp in result.competences:
        if not comp.name:
            result.warnings.append(f"у компетенции {comp.code} не найдено название")
    if not outcome_tables:
        result.warnings.append("не найдена таблица результатов обучения (1.3)")
    if not assessment_tables:
        result.warnings.append("не найдена таблица оценочных средств (3.1.4)")
    if not result.control_forms:
        result.warnings.append("не найдены формы контроля по семестрам")
    if not result.workload_table:
        result.warnings.append("не найдена таблица трудоёмкости по видам занятий")
    if llm is not None:
        result.llm = extract_with_llm(text, file, llm)
        result.warnings.extend(f"LLM: {w}" for w in result.llm.warnings)
        _cross_check_llm(result)
    return result


def _cross_check_llm(result: RPDExtraction) -> None:
    """Расхождения между LLM и правилами: повод для ручной проверки, а не для исправления.

    Если общий объём в 1.4 не назван, сравнивается сумма часов форм, но только когда часы
    есть у каждой формы: иначе сумма неполная и расхождение было бы ложным.
    """
    assert result.llm is not None
    forms = result.llm.active_forms
    stated = result.llm.active_total_hours
    label = "объём активных форм в 1.4"
    if stated is None and forms and all(f.hours is not None for f in forms):
        stated = sum(f.hours or 0 for f in forms)
        label = "сумма часов активных форм в 1.4"
    table = {w.active_hours for w in result.workload_table if not w.is_total and w.active_hours}
    if stated is not None and table and stated not in table:
        result.warnings.append(
            f"LLM: {label} ({stated:g} ч) не совпадает с таблицей 2.1 "
            f"({', '.join(f'{h:g}' for h in sorted(table))})"
        )


# --- титул ---------------------------------------------------------------------------------


def _extract_title_block(text: str, result: RPDExtraction) -> None:
    head = text.splitlines()[:_TITLE_SCAN_LINES]
    plain = clean_text("\n".join(head))
    if m := CREDITS_RE.search(plain):
        result.credits = float(m.group(1).replace(",", "."))
    if m := REG_NUMBER_RE.search(plain):
        result.registration_number = m.group(1)

    lines: list[str] = []
    for raw in head:
        line = _HEADING_MARK_RE.sub("", clean_text(raw))  # в md титул идёт заголовками «###»
        if LANGUAGE_LINE_RE.match(line):
            break
        if line:
            lines.append(line)
    idx = next((i for i, ln in enumerate(lines) if _squash(ln) == "РАБОЧАЯПРОГРАММА"), None)
    if idx is None or idx + 1 >= len(lines):
        return
    rest = lines[idx + 1 :]
    if "дисциплины" in rest[0].lower():
        result.doc_kind = "discipline"
    elif _squash(rest[0]) == "ПРАКТИКИ":
        result.doc_kind = "practice"
    else:
        return
    if len(rest) > 1:
        result.title = rest[1]
    if len(rest) > 2:
        result.title_en = rest[2]


def _one_line(s: str) -> str:
    return " ".join(s.split())


def _squash(s: str) -> str:
    return "".join(s.split()).upper()


# --- таблицы ------------------------------------------------------------------------------


def _header(table: MdTable) -> str:
    """Текст первых строк таблицы в нижнем регистре, для опознания таблицы по заголовку."""
    return _one_line(clean_text(" ".join(c for row in table.rows[:3] for c in row))).lower()


def _ref(table: MdTable, row_idx: int, file: str) -> SourceRef:
    return SourceRef(file=file, section=table.section, line=table.row_lines[row_idx])


def _extract_outcome_table(table: MdTable, file: str, result: RPDExtraction) -> None:
    competences = {c.code: c for c in result.competences}
    seen_indicators = {(i.code, i.competence_code) for i in result.indicators}
    current_code: str | None = None
    prev_out = ""
    group: list[LearningOutcome] = []  # результаты текущей (возможно, объединённой) ячейки

    header = _outcome_header(table)
    if header is None:
        return
    header_idx, comp_col, out_col, ind_col = header

    for idx, row in enumerate(table.rows):
        if idx <= header_idx:
            continue

        def cell(col: int | None, row: list[str] = row) -> str:
            return row[col] if col is not None and col < len(row) else ""

        comp_cell, out_cell, ind_cell = cell(comp_col), cell(out_col), cell(ind_col)
        # ячейка результатов, объединённая по вертикали на несколько компетенций, повторяется
        # парсером в каждой строке: результаты берём один раз, а повторные строки только
        # добавляют свою компетенцию к уже извлечённым
        out_key = _one_line(clean_text(out_cell))
        repeated = bool(out_key) and out_key == prev_out
        if repeated:
            out_cell = ""
        prev_out = out_key or prev_out
        if not out_cell.strip() and not ind_cell.strip() and not repeated:
            continue
        ref = _ref(table, idx, file)

        if parsed := _parse_competence_cell(comp_cell):
            current_code, name = parsed
            known = competences.get(current_code)
            if known is None:
                competences[current_code] = Competence(code=current_code, name=name, source=ref)
            elif known.name is None and name:
                known.name = name

        for code, text, level in _parse_indicators(ind_cell):
            if (code, current_code) not in seen_indicators:
                seen_indicators.add((code, current_code))
                result.indicators.append(
                    Indicator(
                        code=code, competence_code=current_code, text=text, level=level, source=ref
                    )
                )
        if repeated:
            for outcome in group:
                if current_code and current_code not in outcome.competence_codes:
                    outcome.competence_codes.append(current_code)
        elif out_key:
            group = []
        for kind, code, text in _parse_outcomes(out_cell):
            outcome = LearningOutcome(
                kind=kind,
                code=code,
                indicator_code=_indicator_of(code),
                competence_codes=[current_code] if current_code else [],
                text=text,
                source=ref,
            )
            group.append(outcome)
            result.learning_outcomes.append(outcome)
    result.competences = list(competences.values())


def _outcome_header(table: MdTable) -> tuple[int, int | None, int, int | None] | None:
    """Строка-заголовок таблицы 1.3 и индексы столбцов: компетенция, результаты, индикатор."""
    for i, row in enumerate(table.rows[:4]):
        low = [_one_line(clean_text(c)).lower() for c in row]
        outs = [j for j, c in enumerate(low) if "планируемые результаты" in c]
        if not outs:
            continue
        comps = [
            j
            for j, c in enumerate(low)
            if "компетенц" in c and "планируемые" not in c and "индикатор" not in c
        ]
        comp = next((j for j in comps if "код" in low[j]), comps[0] if comps else None)
        ind = next((j for j, c in enumerate(low) if "индикатор" in c), None)
        return i, comp, outs[0], ind
    return None


def _parse_competence_cell(cell: str) -> tuple[str, str | None] | None:
    text = clean_text(cell)
    if m := COMPETENCE_CELL_RE.match(text):
        return m.group(1), _one_line(clean_text(m.group(2) or "")) or None
    if m := NAME_THEN_CODE_RE.match(text):
        return m.group(2), _one_line(clean_text(m.group(1))) or None
    return None


def _parse_indicators(cell: str) -> list[tuple[str, str, str | None]]:
    text = clean_text(cell)
    if suffixed := _parse_suffix_indicators(text):
        return suffixed
    outcome_spans = [m.span() for m in OUTCOME_RE.finditer(text)]
    found = [
        m
        for m in INDICATOR_RE.finditer(text)
        if not any(s <= m.start() < e for s, e in outcome_spans)
    ]
    items = []
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        body = _one_line(text[m.end() : end]).lstrip(".:;, ").strip()
        body, level = _split_level(body)
        items.append((m.group(0), body.rstrip(". ;").strip(), level))
    # «ОПК-1.1; ОПК-1.2: текст» — у первого кода текста нет, берём общий
    for i in range(len(items) - 2, -1, -1):
        if not items[i][1]:
            items[i] = (items[i][0], items[i + 1][1], items[i + 1][2])
    return items


def _parse_suffix_indicators(text: str) -> list[tuple[str, str, str | None]]:
    """Формат «текст (КОД)», по одному индикатору в строке; иначе пустой список."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    items = []
    for line in lines:
        m = INDICATOR_SUFFIX_RE.match(line)
        if not m or INDICATOR_RE.search(m.group("text")):  # код ещё и в начале — формат «КОД текст»
            return []
        body, level = _split_level(_one_line(m.group("text")))
        items.append((m.group("code"), body.rstrip(". ;").strip(), level))
    return items


def _split_level(body: str) -> tuple[str, str | None]:
    if lm := LEVEL_RE.search(body):
        return body[: lm.start()].strip(), lm.group(1).capitalize()
    return body, None


def _indicator_of(outcome_code: str | None) -> str | None:
    if outcome_code and (m := INDICATOR_RE.match(outcome_code)):
        return m.group(0)
    return None


def _parse_outcomes(cell: str) -> list[tuple[OutcomeKind | None, str | None, str]]:
    text = clean_text(cell)
    heads = list(KIND_HEADING_RE.finditer(text))
    groups: list[tuple[OutcomeKind | None, str]] = []
    if not heads:
        groups.append((None, text))
    else:
        if text[: heads[0].start()].strip():
            groups.append((None, text[: heads[0].start()]))
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            groups.append((KIND_CANONICAL[h.group(1).lower()], text[h.end() : end]))  # type: ignore[arg-type]

    out: list[tuple[OutcomeKind | None, str | None, str]] = []
    for kind, body in groups:
        for part in BULLET_RE.split(_join_wrapped_lines(body)):
            for chunk in part.split("\n"):  # пункты в ячейке разделены переносами строк
                for item in _split_by_codes(chunk):
                    out.extend(_parse_outcome_item(item, kind))
    return out


def _join_wrapped_lines(body: str) -> str:
    """Склеивает пункт, перенесённый на следующую строку после запятой.

    Запятая в конце строки — перенос внутри пункта, только если она одна среди строк,
    кончающихся иначе («…задач,» / «…данных;»). Если запятой кончаются несколько строк подряд
    («…сети,» / «…фильтры,»), это перечисление, и каждая строка остаётся отдельным пунктом.
    """
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    enders = {ln[-1] for ln in lines[:-1] if ln[-1] in ",;."}
    if "," not in enders or enders == {","}:
        return body

    def comma(i: int) -> bool:
        return 0 <= i < len(lines) and lines[i].endswith(",")

    merged: list[str] = []
    carry = ""
    for i, line in enumerate(lines):
        line = f"{carry} {line}".strip()
        isolated = comma(i) and not comma(i - 1) and not comma(i + 1) and i < len(lines) - 1
        if isolated:
            carry = line
            continue
        merged.append(line)
        carry = ""
    return "\n".join(merged)


def _split_by_codes(chunk: str) -> list[str]:
    """Режет кусок на пункты: по кодам результатов, а без кодов — по словам «Знать/Уметь»."""
    suffix = list(SUFFIX_OUTCOME_RE.finditer(chunk))
    if len(suffix) > 1:  # «текст (КОД); текст (КОД)»
        cuts = [m.end() for m in suffix[:-1]]
    else:  # «КОД. текст КОД. текст»
        leading = [
            m.start()
            for m in OUTCOME_RE.finditer(chunk)
            if not chunk[: m.start()].rstrip().endswith("(")
        ]
        cuts = [p for p in leading if p > 0] if len(leading) > 1 else []
    if not cuts and not OUTCOME_RE.search(chunk):
        return [p for p in KIND_SPLIT_RE.split(chunk) if p.strip()]
    pieces, prev = [], 0
    for c in cuts:
        pieces.append(chunk[prev:c])
        prev = c
    pieces.append(chunk[prev:])
    return [p for p in pieces if p.strip()]


def _parse_outcome_item(
    item: str, kind: OutcomeKind | None
) -> list[tuple[OutcomeKind | None, str | None, str]]:
    """Один пункт → по записи на каждый код в нём (обычно один; без кода — одна запись)."""
    codes: list[str] = []
    if m := SUFFIX_OUTCOME_RE.search(item):
        codes = OUTCOME_RE.findall(m.group(1))
        item = item[: m.start()] + item[m.end() :]
    elif m := OUTCOME_RE.search(item):
        codes = [m.group(0)]
        item = item[: m.start()] + item[m.end() :]
    text = " ".join(item.split()).strip(" ;,:").lstrip(". ").strip()
    if len(text) < 3:
        return []
    if not codes:
        return [_outcome_tuple(kind, None, text)]
    return [_outcome_tuple(kind, " ".join(code.split()), text) for code in codes]


def _outcome_tuple(
    kind: OutcomeKind | None, code: str | None, text: str
) -> tuple[OutcomeKind | None, str | None, str]:
    if code:
        kind = kind or KIND_BY_LETTER.get(code[-3])  # type: ignore[assignment]
    if kind is None and (m := KIND_WORD_RE.match(text)):
        kind = KIND_CANONICAL[m.group(1).lower()]  # type: ignore[assignment]
    # «Знает алгоритмы…» при kind=знает: вид уже записан отдельно, в тексте оставляем суть
    if kind and (m := KIND_WORD_RE.match(text)) and KIND_CANONICAL[m.group(1).lower()] == kind:
        rest = text[m.end() :].lstrip(" :.,;").strip()
        if len(rest) >= 3:
            text = rest
    return kind, code, text


def _extract_assessment_table(table: MdTable, file: str, result: RPDExtraction) -> None:
    for idx, row in enumerate(table.rows):
        if not row:
            continue
        number = clean_text(row[0]).rstrip(".)")
        filled = [clean_text(c) for c in row[1:] if clean_text(c)]
        if number and not number.isdigit():
            continue  # заголовок и прочие служебные строки
        if len(filled) >= 2:
            indicator_text, materials = _one_line(" ".join(filled[:-1])), filled[-1]
        elif len(filled) == 1 and (
            INDICATOR_RE.search(filled[0]) or COMPETENCE_RE.match(filled[0])
        ):
            indicator_text, materials = filled[0], ""  # КИМ в строке не указаны
        else:
            continue
        indicators = _parse_indicators(indicator_text)
        competence = None if indicators else _parse_competence_cell(indicator_text)
        if not number and not indicators:
            continue  # строка без номера — продолжение, только если в ней есть индикатор
        # в одной строке может быть несколько индикаторов с общими материалами
        for code, text in [(c, t) for c, t, _ in indicators] or [(None, indicator_text)]:
            if competence:  # строка относится к компетенции целиком, а не к индикатору
                code, text = None, competence[1]
            result.assessments.append(
                AssessmentLink(
                    indicator_code=code,
                    competence_code=_known_competence(competence[0], result)
                    if competence
                    else None,
                    indicator_text=text,
                    materials_raw=materials,
                    materials=[p.strip() for p in re.split(r"[;\n]", materials) if p.strip()],
                    source=_ref(table, idx, file),
                )
            )


def _known_competence(code: str, result: RPDExtraction) -> str:
    """Код компетенции в написании из таблицы 1.3 (с учётом «03» и «3»), иначе как в строке."""
    key = code_key(code)
    return next((c.code for c in result.competences if code_key(c.code) == key), code)


def _extract_control_forms(table: MdTable, file: str, result: RPDExtraction) -> None:
    in_forms = False
    trajectory: str | None = None
    study_form: str | None = None
    for idx, row in enumerate(table.rows):
        if any("формы текущего контроля" in _one_line(clean_text(c)).lower() for c in row):
            in_forms = True
            continue
        if not in_forms:
            continue
        first = _one_line(clean_text(row[0])) if row else ""
        if TRAJECTORY_ROW_RE.search(first):  # новая траектория: форма обучения задаётся заново
            trajectory, study_form = first, None
            continue
        if (fm := STUDY_FORM_RE.search(first)) and "обучени" in first.lower():
            study_form = fm.group(1).lower()
            continue
        m = SEMESTER_ROW_RE.match(first)
        if not m:
            continue
        values: list[str] = []
        for c in row[1:]:
            c = clean_text(c)
            if c and c not in values:
                values.append(c)
        intermediate_raw = "; ".join(v for v in values if INTERMEDIATE_RE.match(v)) or None
        current_raw = "; ".join(v for v in values if not INTERMEDIATE_RE.match(v)) or None
        form = ControlForm(
            semester=int(m.group(1)),
            trajectory=trajectory,
            study_form=study_form,
            current=_split_items(current_raw),
            intermediate=_parse_attestations(intermediate_raw),
            current_raw=current_raw,
            intermediate_raw=intermediate_raw,
            source=_ref(table, idx, file),
        )
        key = _form_key(form)
        if all(_form_key(f) != key for f in result.control_forms):
            result.control_forms.append(form)


def _form_key(f: ControlForm) -> tuple:
    return (f.trajectory, f.study_form, f.semester, f.current_raw, f.intermediate_raw)


def _split_items(raw: str | None) -> list[str]:
    """Режет «a, b; c» по запятым, «;» и переносам строк вне скобок; повторы убирает."""
    if not raw:
        return []
    items, depth, buf = [], 0, ""
    for ch in raw:
        depth += (ch == "(") - (ch == ")")
        if ch in ",;\n" and depth <= 0:
            items.append(buf)
            buf = ""
        else:
            buf += ch
    items.append(buf)
    out: list[str] = []
    for item in (i.strip() for i in items):
        if item and item not in out:
            out.append(item)
    return out


def _parse_attestations(raw: str | None) -> list[Attestation]:
    """«зачёт, устно, ...; экзамен, устно» → [зачёт[устно, ...], экзамен[устно]]."""
    if not raw:
        return []
    found = list(ATTESTATION_KIND_RE.finditer(raw))
    result = []
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(raw)
        tail = raw[m.end() : end].strip(" ,;:")
        details = _split_items(tail.replace("(", ", ").replace(")", ""))
        kind = " ".join(m.group(0).lower().replace("ё", "е").split()).replace("зачет", "зачёт")
        result.append(Attestation(kind=kind, details=details))
    return result
