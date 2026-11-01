"""Сущности из разделов РПД/РПП вне таблиц 1.3, 2.1 и 3.1.4: всё, что берётся регулярками.

Всё разбирается регулярками по номерам и заголовкам разделов (см. ADR-0002, «Какие сущности
извлекаем»):

- титул: язык обучения, код дисциплины;
- 1.1 цели; 1.4 активные и интерактивные формы с часами;
- 2.2 часы по темам (и сверка с таблицей 2.1);
- 3.1.3 шкалы оценивания (A–F, баллы) и структура билета; 3.1.4 вопросы к экзамену и примеры задач;
- раздел 4 разработчики; направление, ООП и профили;
- РПП: вид, тип и способ практики, профстандарты (1.6), компетенции по группам (1.7),
  соответствие компетенций трудовым функциям (1.8).

Что не нашлось, не выдумываем: пропускаем без предупреждения, если раздела нет в шаблоне
(в РПП нет 2.2), и с предупреждением, если раздел есть, а разобрать его не удалось.
"""

import re

from edu_doc_agent.extraction.md_utils import MdSection, MdTable, clean_text, split_sections
from edu_doc_agent.extraction.models import (
    ActiveForm,
    Competence,
    ControlMaterial,
    Developer,
    GradeScaleRow,
    PracticeInfo,
    ProfStandard,
    ProgramInfo,
    RPDExtraction,
    SourceRef,
    StandardLink,
    Ticket,
    TicketPart,
    TopicHours,
)
from edu_doc_agent.extraction.patterns import (
    COMPETENCE_RE,
    LANGUAGE_LINE_RE,
    TRAJECTORY_ROW_RE,
)

_CHECKED = "🗹☑☒✓✔■\uf0fe"  # \uf0fe — галочка из шрифта Wingdings
_UNCHECKED = "□☐\uf0a8"
_TITLE_SCAN_LINES = 40
_NESTED_RE = re.compile(r"^\d+(?:\.\d+)+\.?\s")

_HOURS_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:час\w*|ч\b\.?)", re.I)
_SEMESTER_RE = re.compile(r"Семестр\s*(\d+)", re.I)
_NUMBERED_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.+)$")
_NUMBER_WORDS = {
    "один": 1,
    "одна": 1,
    "одну": 1,
    "одно": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
}
_TICKET_RE = re.compile(
    r"[Бб]илет\w*\s+(?:содержит|состоит\s+из|включает(?:\s+в\s+себя)?)\s+(?P<body>[^.\n]+)"
)
_TICKET_PART_RE = re.compile(r"^(?P<n>\d+|" + "|".join(_NUMBER_WORDS) + r")\s+(?P<item>.+)$", re.I)
_DIRECTION_RE = re.compile(r"^Направление:\s*(\d\d\.\d\d\.\d\d)\s*(?:[«\"](.+?)[»\"]?)?\s*$", re.M)
_PROGRAM_RE = re.compile(r"^(?:ООП|Образовательная программа):\s*(.+?)\s*$", re.M)
_PROFILE_RE = re.compile(r"Профиль\s+(.+?)[»\"]?\s*$", re.I)
_STANDARD_RE = re.compile(
    r"^[-*•]?\s*(?:Код\s+)?(\d\d\.\d{3})\s*(?:Профессиональный стандарт\s*)?[«\"](.+?)[»\"]"
)
_FUNCTION_RE = re.compile(r"^[-*•]\s*(?:\d\d\.\d{3}\.)?([A-ZА-Я]\.?\d[\w./]*)\s+(.+?)[.,;]?\s*$")
_LINK_STANDARD_RE = re.compile(r"ПС\s*(\d\d\.\d{3})\s*:\s*([^;\n]+)")
_LINK_FUNCTION_RE = re.compile(r"(\d\d\.\d{3})\.([A-ZА-Я]\.?\d[\w]*)")  # «06.001.A.3»
_LATIN = str.maketrans("АВСЕНКМОРТХ", "ABCEHKMOPTX")  # в документах функции пишут и кириллицей
_COMPETENCE_ITEM_RE = re.compile(rf"^[-*•]\s*({COMPETENCE_RE.pattern})[.:]?\s+(.+?)\.?\s*$")

# вид занятий из заголовка/ячейки → общее название; нужно для сверки 2.2 и 2.1
KIND_ALIASES = {
    "лекц": "лекции",
    "семинар": "семинары",
    "практическ": "практические занятия",
    "лаборатор": "лабораторные работы",
    "консультац": "консультации",
}


def competence_group(code: str) -> str | None:
    """Группа компетенции по коду: УК, УНК, УКБ, ОПК или ПК (ПКП и ПКА входят в ПК)."""
    if code.startswith("ПКП-УНК"):
        return "УНК"
    for prefix in ("УКБ", "УК", "ОПК"):
        if code.startswith(prefix + "-"):
            return prefix
    return "ПК" if code.startswith(("ПКП", "ПКА")) else None


def kind_alias(kind: str) -> str | None:
    low = kind.lower()
    return next((v for k, v in KIND_ALIASES.items() if k in low), None)


def extract_sections(text: str, tables: list[MdTable], file: str, result: RPDExtraction) -> None:
    lines = text.splitlines()
    sections = split_sections(text)
    _extract_title_fields(lines, result)
    _extract_goals(sections, file, result)
    _extract_active_forms(sections, file, result)
    _extract_topic_hours(tables, lines, file, result)
    for table in tables:
        _extract_grade_scale(table, file, result)
        _extract_developers(table, file, result)
        _extract_standard_links(table, file, result)
    _extract_tickets(sections, file, result)
    _extract_control_materials(sections, file, result)
    _extract_program(text, tables, file, result)
    _extract_practice(sections, file, result)
    _extract_prof_standards(sections, file, result)
    _extract_competence_lists(sections, file, result)
    for comp in result.competences:
        comp.group = comp.group or competence_group(comp.code)


def _ref(section: MdSection | None, line: int, file: str) -> SourceRef:
    return SourceRef(file=file, section=section.heading if section else None, line=line)


def _find(sections: list[MdSection], number: str) -> MdSection | None:
    return next((s for s in sections if s.number == number), None)


def _body(section: MdSection) -> list[tuple[int, str]]:
    return [(section.line + 1 + i, ln) for i, ln in enumerate(section.text.split("\n")) if ln]


def _one_line(s: str) -> str:
    return " ".join(s.split())


# --- титул -------------------------------------------------------------------------------


def _extract_title_fields(lines: list[str], result: RPDExtraction) -> None:
    result.discipline_code = result.registration_number
    head = [_one_line(clean_text(ln)).lstrip("# ").strip() for ln in lines[:_TITLE_SCAN_LINES]]
    for i, line in enumerate(head):
        if LANGUAGE_LINE_RE.match(line):
            result.language = next((ln for ln in head[i + 1 :] if ln), None)
            return


# --- 1.1, 1.4 ----------------------------------------------------------------------------


def _extract_goals(sections: list[MdSection], file: str, result: RPDExtraction) -> None:
    section = _find(sections, "1.1")
    if section and (body := _body(section)):
        result.goals = "\n".join(ln for _, ln in body)


_OF_THEM_RE = re.compile(r"из них\s+(\d+(?:[.,]\d+)?)\s*(?:час\w*|ч\b\.?)", re.I)
_TOTAL_WORD_RE = re.compile(r"составля|объ[её]м", re.I)


def _extract_active_forms(sections: list[MdSection], file: str, result: RPDExtraction) -> None:
    section = _find(sections, "1.4")
    if section is None or "активн" not in section.heading.lower():
        return
    for lineno, line in _body(section):
        ref = _ref(section, lineno, file)
        for sentence in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])", line):
            matches = list(_HOURS_RE.finditer(sentence))
            if not matches:
                continue
            # «16 часов, из них 2 часа в активной форме»: в активной форме только 2 часа
            if of_them := _OF_THEM_RE.search(sentence):
                hours = float(of_them.group(1).replace(",", "."))
                result.active_forms.append(
                    ActiveForm(text=sentence.strip(), hours=hours, source=ref)
                )
                continue
            colon = sentence.find(":")
            has_total = len(matches) > 1 and colon > 0 and _TOTAL_WORD_RE.search(sentence[:colon])
            # «составляет 40 часов: семинары — 16 часов, практические — 24 часа»: общий объём
            # и состав; «Лекции, 8 ч Лабораторных работ, 50 ч.» — по записи на форму
            start = 0
            for i, m in enumerate(matches):
                text = sentence[start : m.end()].strip(" ,;:")
                result.active_forms.append(
                    ActiveForm(
                        text=text,
                        hours=float(m.group(1).replace(",", ".")),
                        is_total=bool(has_total and i == 0),
                        source=ref,
                    )
                )
                start = m.end()
                if has_total and i == 0:
                    start = colon + 1


# --- 2.2 часы по темам -------------------------------------------------------------------


def _columns(row: list[str]) -> dict[str, int]:
    low = [_one_line(clean_text(c)).lower() for c in row]
    cols: dict[str, int] = {}
    for j, c in enumerate(low):
        if "наименование темы" in c:
            cols["topic"] = j
        elif "вид" in c and "занят" in c:
            cols["kind"] = j
        elif "количество часов" in c or c == "часы":
            cols["hours"] = j
        elif c.startswith("№"):
            cols["number"] = j
    return cols


def _extract_topic_hours(
    tables: list[MdTable], lines: list[str], file: str, result: RPDExtraction
) -> None:
    per_table: list[tuple[int, dict[str, float]]] = []  # (семестр, сумма по виду занятий)
    for table in tables:
        cols = _columns(table.rows[0]) if table.rows else {}
        if not {"topic", "kind", "hours"} <= cols.keys():
            continue
        semester = _semester_before(lines, table.row_lines[0])
        sums: dict[str, float] = {}
        for idx, row in enumerate(table.rows[1:], start=1):
            cell = [clean_text(c) if c else "" for c in row]
            if max(cols.values()) >= len(cell):
                continue
            try:
                hours = float(cell[cols["hours"]].replace(",", "."))
            except ValueError:
                continue
            if not cell[cols["topic"]] or not cell[cols["kind"]]:
                continue
            number = cell[cols["number"]] if "number" in cols else ""
            row_kind = _one_line(cell[cols["kind"]])
            result.topic_hours.append(
                TopicHours(
                    semester=semester,
                    number=number or None,
                    topic=_one_line(cell[cols["topic"]]),
                    kind=row_kind,
                    hours=hours,
                    source=SourceRef(file=file, section=table.section, line=table.row_lines[idx]),
                )
            )
            if kind := kind_alias(row_kind):
                sums[kind] = sums.get(kind, 0) + hours
        if semester is not None:
            per_table.append((semester, sums))
    _check_topic_hours(per_table, result)


def _semester_before(lines: list[str], first_row_line: int) -> int | None:
    """«Период обучения (модуль): Семестр 1» над таблицей: смотрим ближайшие непустые строки."""
    seen = 0
    for ln in reversed(lines[: first_row_line - 1]):
        if not ln.strip():
            continue
        if ln.lstrip().startswith(("|", "#")):  # дошли до другой таблицы или заголовка раздела
            break
        if m := _SEMESTER_RE.search(ln):
            return int(m.group(1))
        seen += 1
        if seen == 2:
            break
    return None


def _check_topic_hours(
    per_table: list[tuple[int, dict[str, float]]], result: RPDExtraction
) -> None:
    """Часы по темам (2.2) должны сходиться с таблицей 2.1 по виду занятий в каждом семестре.

    Траекторий бывает несколько, и у каждой своя таблица 2.2, поэтому достаточно совпадения
    с любой из строк 2.1 этого семестра.
    """
    kinds = {(sem, kind) for sem, sums in per_table for kind in sums}
    for semester, kind in sorted(kinds):
        topics = {sums[kind] for sem, sums in per_table if sem == semester and kind in sums}
        values = {
            r.contact_hours[kind]
            for r in result.workload_table
            if r.semester == semester and kind in r.contact_hours
        }
        if values and not topics & values:
            result.warnings.append(
                f"часы по темам (2.2), семестр {semester}, {kind}: "
                f"{', '.join(f'{t:g}' for t in sorted(topics))} ч, "
                f"в таблице 2.1 — {', '.join(f'{v:g}' for v in sorted(values))}"
            )


# --- 3.1.3 шкалы и билеты ----------------------------------------------------------------


def _attestation_name(header: str) -> str:
    m = re.search(r"(?:при проведении|за)\s+(.+)$", header.lower())
    name = (m.group(1) if m else header.lower()).replace("ё", "е").strip()
    if name.startswith("зач"):
        return "зачёт с оценкой" if "оценк" in name else "зачёт"
    return "экзамен" if name.startswith("экзамен") else name


def _extract_grade_scale(table: MdTable, file: str, result: RPDExtraction) -> None:
    if not table.rows:
        return
    low = [_one_line(clean_text(c)).lower() for c in table.rows[0]]
    ects = next((j for j, c in enumerate(low) if "ects" in c), None)
    # столбец оценки: «Оценка СПбГУ», «Оценка при проведении зачёта», «Оценка за экзамен»
    grade = next((j for j, c in enumerate(low) if c.startswith("оценка") and "ects" not in c), None)
    if ects is None or grade is None:
        return
    attestation = _attestation_name(low[grade])
    # критерий — последний столбец, а если последний занят оценкой, то первый из остальных
    # («Процент выполнения задач | Оценка ECTS | Оценка при проведении зачёта»)
    rest = [j for j in range(len(low)) if j not in (ects, grade)]
    if not rest:
        return
    criteria = rest[-1] if rest[-1] > grade else rest[0]
    for idx, row in enumerate(table.rows[1:], start=1):
        if len(row) <= max(ects, grade, criteria):
            continue
        g, e, c = (_one_line(clean_text(row[j])) for j in (grade, ects, criteria))
        if g and c:
            result.grade_scales.append(
                GradeScaleRow(
                    attestation=attestation,
                    grade=g,
                    ects=e or None,
                    criteria=c,
                    source=SourceRef(file=file, section=table.section, line=table.row_lines[idx]),
                )
            )


def _extract_tickets(sections: list[MdSection], file: str, result: RPDExtraction) -> None:
    for section in sections:
        for lineno, line in _body(section):
            if not (m := _TICKET_RE.search(line)):
                continue
            parts = []
            for chunk in re.split(r",\s*|\s+и\s+", m.group("body")):
                if pm := _TICKET_PART_RE.match(chunk.strip()):
                    n = pm.group("n").lower()
                    parts.append(
                        TicketPart(
                            count=int(n) if n.isdigit() else _NUMBER_WORDS[n],
                            item=pm.group("item").strip(),
                        )
                    )
            if parts:
                text = m.group(0).strip()
                result.tickets.append(
                    Ticket(text=text, parts=parts, source=_ref(section, lineno, file))
                )


# --- КИМ: вопросы и примеры задач ---------------------------------------------------------


def _material_kind(heading: str) -> str | None:
    low = heading.lower()
    if "перечень вопросов" in low or low.startswith("вопросы"):
        return "вопросы к зачёту" if "зач" in low else "вопросы к экзамену"
    if low.startswith("примеры задач"):
        return "примеры задач"
    if low.startswith("контрольная работа"):
        return "контрольная работа"
    return None


def _extract_control_materials(sections: list[MdSection], file: str, result: RPDExtraction) -> None:
    for section in sections:
        kind = _material_kind(section.heading)
        if kind is None:
            continue
        body = _body(section)
        items = [m.group(1).strip() for _, ln in body if (m := _NUMBERED_RE.match(ln))]
        items = items or [ln for _, ln in body]
        if items:
            result.control_materials.append(
                ControlMaterial(
                    kind=kind,  # type: ignore[arg-type]
                    title=section.heading,
                    items=items,
                    source=_ref(section, section.line, file),
                )
            )


# --- раздел 4 ----------------------------------------------------------------------------


def _extract_developers(table: MdTable, file: str, result: RPDExtraction) -> None:
    if not table.rows:
        return
    low = [_one_line(clean_text(c)).lower() for c in table.rows[0]]
    cols = {
        field: next((j for j, c in enumerate(low) if key in c), None)
        for field, key in (
            ("name", "фамилия"),
            ("degree", "степень"),
            ("title", "звание"),
            ("position", "должность"),
            ("contacts", "контакт"),
        )
    }
    if cols["name"] is None or cols["position"] is None:
        return

    def cell(row: list[str], field: str) -> str | None:
        j = cols[field]
        return _one_line(clean_text(row[j])) or None if j is not None and j < len(row) else None

    for idx, row in enumerate(table.rows[1:], start=1):
        if name := cell(row, "name"):
            result.developers.append(
                Developer(
                    name=name,
                    degree=cell(row, "degree"),
                    title=cell(row, "title"),
                    position=cell(row, "position"),
                    contacts=cell(row, "contacts"),
                    source=SourceRef(file=file, section=table.section, line=table.row_lines[idx]),
                )
            )


# --- направление, ООП, профили -----------------------------------------------------------


def _extract_program(text: str, tables: list[MdTable], file: str, result: RPDExtraction) -> None:
    info = ProgramInfo()
    if m := _DIRECTION_RE.search(text):
        info.direction_code = m.group(1)
        info.direction = _one_line(m.group(2) or "").rstrip("»") or None
        info.source = SourceRef(file=file, line=text.count("\n", 0, m.start()) + 1)
    if m := _PROGRAM_RE.search(text):
        info.program = m.group(1)
    for table in tables:
        for row in table.rows:
            first = _one_line(clean_text(row[0])) if row else ""
            if TRAJECTORY_ROW_RE.search(first) and (pm := _PROFILE_RE.search(first)):
                profile = pm.group(1).strip("«» ")
                if profile not in info.profiles:
                    info.profiles.append(profile)
    if info.direction_code or info.program or info.profiles:
        result.program = info


# --- РПП ---------------------------------------------------------------------------------


def _checked_lines(body: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Отмеченные пункты. Без чекбоксов — первая строка раздела.

    В части документов значок отмеченного пункта при конвертации теряется, а у остальных
    остаётся «□». Тогда отмеченным считаем пункт без значка.
    """
    # вложенные пункты («1.3.1. Дополнительные характеристики…») к выбору не относятся
    nested = next((i for i, (_, ln) in enumerate(body) if _NESTED_RE.match(ln)), len(body))
    body = [(n, ln) for n, ln in body[:nested] if not ln.startswith("(")]  # пояснение в скобках
    if not any(ln[:1] in _UNCHECKED for _, ln in body):
        return [(n, ln) for n, ln in body if ln[:1] in _CHECKED] or body[:1]
    marked = [(n, ln) for n, ln in body if ln[:1] in _CHECKED]
    return marked or [(n, ln) for n, ln in body if ln[:1] not in _UNCHECKED]


def _extract_practice(sections: list[MdSection], file: str, result: RPDExtraction) -> None:
    kind_section = next((s for s in sections if re.match(r"1\.2\.?\s+вид", s.heading, re.I)), None)
    type_section = _find(sections, "1.2.1")
    method_section = next(
        (s for s in sections if re.match(r"1\.3\.?\s+способ", s.heading, re.I)), None
    )
    anchor = kind_section or type_section or method_section
    if anchor is None:
        return

    def first(section: MdSection | None) -> str | None:
        picked = _checked_lines(_body(section)) if section else []
        return picked[0][1].lstrip(_CHECKED + _UNCHECKED).strip() if picked else None

    kind = first(kind_section)
    result.practice = PracticeInfo(
        kind=kind.split(",")[0].strip() if kind else None,
        type=first(type_section),
        method=first(method_section),
        source=_ref(anchor, anchor.line, file),
    )


def _extract_prof_standards(sections: list[MdSection], file: str, result: RPDExtraction) -> None:
    section = _find(sections, "1.6")
    if section is None:
        return
    current: ProfStandard | None = None
    for lineno, line in _body(section):
        if m := _STANDARD_RE.match(line):
            current = ProfStandard(
                code=m.group(1), name=m.group(2), source=_ref(section, lineno, file)
            )
            result.prof_standards.append(current)
        elif current and (fm := _FUNCTION_RE.match(line)):
            current.functions.append(f"{fm.group(1).translate(_LATIN)} {fm.group(2)}")


def _extract_competence_lists(sections: list[MdSection], file: str, result: RPDExtraction) -> None:
    """РПП 1.7: компетенции перечислены списками, а не таблицей 1.3."""
    known = {c.code for c in result.competences}
    in_list = False
    for section in sections:
        if section.number:  # заголовки групп («Универсальные компетенции (УК)») без номера
            in_list = section.number.split(".")[:2] == ["1", "7"]
        if not in_list:
            continue
        for lineno, line in _body(section):
            if (m := _COMPETENCE_ITEM_RE.match(line)) and m.group(1) not in known:
                known.add(m.group(1))
                result.competences.append(
                    Competence(code=m.group(1), name=m.group(2), source=_ref(section, lineno, file))
                )


def _extract_standard_links(table: MdTable, file: str, result: RPDExtraction) -> None:
    if not table.rows:
        return
    header = _one_line(clean_text(" ".join(table.rows[0]))).lower()
    if "трудовые функции" not in header or "компетенц" not in header or len(table.rows[0]) < 2:
        return
    for idx, row in enumerate(table.rows[1:], start=1):
        codes = COMPETENCE_RE.findall(clean_text(row[0]))
        cell = clean_text(row[1])
        functions = {
            m.group(1): [f.strip().translate(_LATIN) for f in m.group(2).split(",") if f.strip()]
            for m in _LINK_STANDARD_RE.finditer(cell)
        }
        for m in _LINK_FUNCTION_RE.finditer(cell):
            found = functions.setdefault(m.group(1), [])
            if (code := m.group(2).translate(_LATIN)) not in found:
                found.append(code)
        if codes and functions:
            result.standard_links.append(
                StandardLink(
                    competence_codes=codes,
                    functions=functions,
                    source=SourceRef(file=file, section=table.section, line=table.row_lines[idx]),
                )
            )
