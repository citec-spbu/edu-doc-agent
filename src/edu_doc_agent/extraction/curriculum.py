"""Извлечение сущностей из учебного плана (docx → Markdown): компетенции, дисциплины, ГИА.

Тот же подход, что и для РПД (ADR-0002): числа, коды и названия берутся детерминированно из
таблиц, у каждой сущности есть источник, расхождения идут в `warnings`. Таблица раздела 2 имеет
двухуровневую шапку («Аудиторная работа» / «Самостоятельная работа» над видами занятий) и
служебные строки контекста: год, семестр, часть периода, профиль, траектория, группа блоков.
"""

import re
from pathlib import Path

from pydantic import Field

from edu_doc_agent.extraction.md_utils import MdTable, clean_text, parse_tables
from edu_doc_agent.extraction.models import _Model
from edu_doc_agent.extraction.patterns import COMPETENCE_RE, code_key
from edu_doc_agent.extraction.source import SourceRef, attach_quotes

HOURS_PER_CREDIT = 36
_EPS = 1e-9

_DISCIPLINE_CODE_RE = re.compile(r"^\[(\d+)\]\s*")
_SEMESTER_RE = re.compile(r"семестр\s+(\d+)", re.I)
_YEAR_RE = re.compile(r"^(\d+)\s+год обучения", re.I)
_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
_REG_NUMBER_RE = re.compile(r"^\d+/\d+/\d+$")

# заголовки групп столбцов второй строки шапки → поле
_CONTACT_GROUP = "аудиторная работа"
_SELF_GROUP = "самостоятельная работа"
_ACTIVE_HEADER = "объём занятий в активных"


class PlanCompetence(_Model):
    """Строка таблицы 1.1: компетенция ОП и её формулировка."""

    code: str
    name: str
    source: SourceRef


class PlanEntry(_Model):
    """Дисциплина, практика или иной элемент плана из таблицы раздела 2."""

    block: str | None = Field(default=None, description="«Блок.1.дисц», «Блок.2..прки», …")
    code: str | None = Field(default=None, description="Код в квадратных скобках: «001856»")
    name: str
    name_en: str | None = None
    credits: float | None = Field(default=None, description="Зачётные единицы")
    competences: list[str] = Field(default_factory=list)
    control: list[str] = Field(
        default_factory=list, description="Текущий контроль и формы аттестации: «зачёт», «экзамен»"
    )
    contact_hours: dict[str, float] = Field(default_factory=dict)
    self_study_hours: dict[str, float] = Field(default_factory=dict)
    active_hours: float | None = Field(
        default=None, description="Объём занятий в активных и интерактивных формах"
    )
    semester: int | None = None
    year: int | None = None
    part: str | None = Field(default=None, description="«Базовая часть периода обучения», …")
    group: str | None = Field(default=None, description="«Блок дисциплин Иностранный язык»")
    profile: str | None = None
    trajectory: str | None = None
    source: SourceRef

    @property
    def total_hours(self) -> float:
        return sum(self.contact_hours.values()) + sum(self.self_study_hours.values())


class FinalAttestation(_Model):
    """Строка таблицы раздела 3: процедура ГИА и проверяемые ею компетенции."""

    block: str | None = None
    credits: float | None = None
    name: str
    name_en: str | None = None
    competences: list[str] = Field(default_factory=list)
    source: SourceRef


class CurriculumPlan(_Model):
    file: str
    sources: dict[str, SourceRef] = Field(
        default_factory=dict, description="Источники полей титула"
    )
    program: str | None = Field(default=None, description="Название ОП из титула")
    level: str | None = None
    direction: str | None = None
    study_form: str | None = None
    languages: str | None = None
    duration: str | None = None
    registration_number: str | None = Field(default=None, description="«26/5003/1»")
    competences: list[PlanCompetence] = Field(default_factory=list)
    entries: list[PlanEntry] = Field(default_factory=list)
    final_attestation: list[FinalAttestation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --- вспомогательное ------------------------------------------------------------------------


def _cell(row: list[str], col: int | None) -> str:
    return clean_text(row[col]) if col is not None and col < len(row) else ""


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _number(text: str) -> float | None:
    text = text.strip()
    return float(text.replace(",", ".")) if _NUMBER_RE.match(text) else None


def _codes(text: str) -> list[str]:
    return [m.group(0) for m in COMPETENCE_RE.finditer(text)]


def _split_name(cell: str) -> tuple[str | None, str, str | None]:
    """«[001856] Математический анализ I\\nMathematical Analysis I» → (код, имя, имя_en)."""
    text = cell.strip()
    code = None
    if m := _DISCIPLINE_CODE_RE.match(text):
        code, text = m.group(1), text[m.end() :]
    lines = [_one_line(ln) for ln in text.split("\n") if ln.strip()]
    if not lines:
        return code, "", None
    boundary = next(
        (
            i
            for i, line in enumerate(lines[1:], 1)
            if re.search(r"[a-zA-Z]{2,}", line) and not re.search(r"[а-яА-ЯёЁ]", line)
        ),
        len(lines),
    )
    return code, " ".join(lines[:boundary]), " ".join(lines[boundary:]) or None


def _ref(table: MdTable, idx: int, file: str) -> SourceRef:
    return SourceRef(file=file, section=table.section, line=table.row_lines[idx])


def _is_context_row(row: list[str]) -> bool:
    """Служебная строка: заполнена только первая ячейка («С01. Семестр 1», «Профиль …»)."""
    return bool(row) and bool(row[0].strip()) and not any(c.strip() for c in row[1:])


# --- таблица 1.1 ----------------------------------------------------------------------------


def _is_competence_table(table: MdTable) -> bool:
    head = [c.lower() for c in table.rows[0]] if table.rows else []
    return len(head) >= 2 and "код компетенции" in head[0] and "наименование" in head[1]


def _extract_competences(table: MdTable, file: str, plan: CurriculumPlan) -> None:
    for idx, row in enumerate(table.rows[1:], start=1):
        codes = _codes(_cell(row, 0))
        name = _one_line(_cell(row, 1))
        if codes and name:
            plan.competences.append(
                PlanCompetence(code=codes[0], name=name, source=_ref(table, idx, file))
            )


# --- таблица раздела 2 ----------------------------------------------------------------------


def _find_plan_header(table: MdTable) -> tuple[int, dict[int, tuple[str, str]]] | None:
    """Индекс строки с видами занятий и карта столбец → (группа, название)."""
    for i, row in enumerate(table.rows):
        low = [clean_text(c).lower() for c in row]
        if "лекции" in low and any("код блока" in c for c in low):
            groups: dict[int, str] = {}
            current = ""
            above = [clean_text(c).lower() for c in table.rows[i - 1]] if i > 0 else []
            for col in range(len(row)):
                if col < len(above) and above[col]:
                    current = above[col]
                groups[col] = current
            return i, {
                col: (groups[col], _one_line(low[col])) for col in range(len(row)) if low[col]
            }
    return None


def _group_kind(group: str) -> str | None:
    if group.startswith(_CONTACT_GROUP):
        return "contact"
    if group.startswith(_SELF_GROUP):
        return "self"
    return None


def _extract_entries(table: MdTable, file: str, plan: CurriculumPlan) -> bool:
    found = _find_plan_header(table)
    if found is None:
        return False
    header_idx, columns = found
    cols: dict[str, int] = {}
    contact: dict[int, str] = {}
    self_study: dict[int, str] = {}
    for col, (group, name) in columns.items():
        if name.startswith("код блока"):
            cols["block"] = col
        elif name.startswith("трудоёмкость") or name.startswith("трудоемкость"):
            cols["credits"] = col
        elif name.startswith("код компетенции"):
            cols["competences"] = col
        elif name.startswith("наименование"):
            cols["name"] = col
        elif name.startswith("виды текущего контроля"):
            cols["control"] = col
        elif name.startswith(_ACTIVE_HEADER):
            cols["active"] = col
        elif _group_kind(group) == "contact":
            contact[col] = name
        elif _group_kind(group) == "self":
            self_study[col] = name
    if "name" not in cols:
        return False

    semester = year = None
    part = group = profile = trajectory = None
    for idx, row in enumerate(table.rows):
        if idx <= header_idx:
            continue
        first = _one_line(_cell(row, 0))
        if first.lower().startswith("код блока"):  # шапка повторяется на каждой странице
            continue
        if _is_context_row(row):
            low = first.lower()
            if m := _YEAR_RE.match(first):
                year = int(m.group(1))
            elif m := _SEMESTER_RE.search(first):
                semester, part, group, profile, trajectory = int(m.group(1)), None, None, None, None
            elif "часть" in low or low.startswith("факультатив"):
                part, group, profile, trajectory = first, None, None, None
            elif low.startswith("блок(и) дисциплин"):
                group, profile, trajectory = None, None, None
            elif low.startswith("блок дисциплин"):
                group, profile, trajectory = first, None, None
            elif low.startswith("профиль"):
                profile, trajectory = first, None
            elif low.startswith("траектория"):
                trajectory = first
            continue
        code, name, name_en = _split_name(_cell(row, cols["name"]))
        block = _one_line(_cell(row, cols.get("block"))) or None
        if not name or not (code or (block and block.lower().startswith("блок"))):
            continue  # заголовки групп выбора («Дисциплины по выбору: …») — не элементы плана
        control = [
            c.strip().lower()
            for c in re.split(r"[,;]", _cell(row, cols.get("control")))
            if c.strip()
        ]
        entry = PlanEntry(
            block=block,
            code=code,
            name=name,
            name_en=name_en,
            credits=_number(_cell(row, cols.get("credits"))),
            competences=_codes(_cell(row, cols.get("competences"))),
            control=control,
            contact_hours=_hours(row, contact),
            self_study_hours=_hours(row, self_study),
            active_hours=_number(_cell(row, cols.get("active"))),
            semester=semester,
            year=year,
            part=part,
            group=group,
            profile=profile,
            trajectory=trajectory,
            source=_ref(table, idx, file),
        )
        plan.entries.append(entry)
    return True


def _hours(row: list[str], columns: dict[int, str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for col, name in columns.items():
        if (value := _number(_cell(row, col))) is not None:
            result[name] = value
    return result


# --- таблица раздела 3 (ГИА) ----------------------------------------------------------------


def _is_attestation_table(table: MdTable) -> bool:
    head = " ".join(table.rows[0]).lower() if table.rows else ""
    return "итоговой аттестации" in head and "компетенций" in head


def _extract_attestation(table: MdTable, file: str, plan: CurriculumPlan) -> None:
    for idx, row in enumerate(table.rows[1:], start=1):
        if _is_context_row(row) or len(row) < 4:
            continue
        _, name, name_en = _split_name(_cell(row, 2))
        codes = _codes(_cell(row, 3))
        if name:
            if not codes:
                plan.warnings.append(f"ГИА, строка {table.row_lines[idx]}: не указаны компетенции")
            plan.final_attestation.append(
                FinalAttestation(
                    block=_one_line(_cell(row, 0)) or None,
                    credits=_number(_cell(row, 1)),
                    name=name,
                    name_en=name_en,
                    competences=codes,
                    source=_ref(table, idx, file),
                )
            )


# --- титул ----------------------------------------------------------------------------------

_TITLE_LABELS = {
    "по уровню": "level",
    "по направлению": "direction",
    "форма обучения": "study_form",
    "язык(и) обучения": "languages",
    "срок обучения": "duration",
}


def _extract_title(text: str, tables: list[MdTable], plan: CurriculumPlan) -> None:
    for table in tables[:3]:
        for idx, row in enumerate(table.rows):
            cells = [_one_line(clean_text(c)) for c in row]
            non_empty = [c for c in cells if c]
            if len(non_empty) >= 2:
                label = non_empty[0].lower().rstrip(":")
                for prefix, field in _TITLE_LABELS.items():
                    if label.startswith(prefix) and getattr(plan, field) is None:
                        setattr(plan, field, non_empty[1])
                        plan.sources[field] = _ref(table, idx, plan.file)
            for c in non_empty:
                if _REG_NUMBER_RE.match(c) and plan.registration_number is None:
                    plan.registration_number = c
                    plan.sources["registration_number"] = _ref(table, idx, plan.file)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|") and "по уровню" in line.lower():
            for j in range(i - 1, -1, -1):
                prev = lines[j]
                if prev.strip() and not prev.startswith("#"):
                    plan.program = _one_line(clean_text(prev))
                    plan.sources["program"] = SourceRef(file=plan.file, section="Титул", line=j + 1)
                    return
            return


# --- сверки ---------------------------------------------------------------------------------


def check_plan(plan: CurriculumPlan) -> None:
    """Часы против з.е. и коды компетенций против таблицы 1.1; расхождения — в `warnings`."""
    known = {code_key(c.code) for c in plan.competences}
    for entry in plan.entries:
        where = f"{entry.code or entry.name} (строка {entry.source.line})"
        if entry.credits:  # 0 з.е. — занятия вне зачётных единиц
            if not entry.contact_hours and not entry.self_study_hours:
                plan.warnings.append(f"план, {where}: не найдены часы при ненулевых з.е.")
            expected = entry.credits * HOURS_PER_CREDIT
            if abs(entry.total_hours - expected) > _EPS:
                plan.warnings.append(
                    f"план, {where}: {entry.total_hours:g} ч не равно "
                    f"{entry.credits:g} з.е. × {HOURS_PER_CREDIT}"
                )
        if known:
            for code in entry.competences:
                if code_key(code) not in known:
                    plan.warnings.append(f"план, {where}: компетенция {code} нет в таблице 1.1")
    if known:
        for att in plan.final_attestation:
            for code in att.competences:
                if code_key(code) not in known:
                    plan.warnings.append(f"ГИА, {att.name}: компетенция {code} нет в таблице 1.1")


# --- точка входа ----------------------------------------------------------------------------


def extract_plan(text: str, file: str = "") -> CurriculumPlan:
    """Учебный план из Markdown (`parse_document(...).to_markdown()`)."""
    plan = CurriculumPlan(file=file)
    tables = parse_tables(text)
    _extract_title(text, tables, plan)
    for table in tables:
        if not table.rows:
            continue
        if _is_competence_table(table):
            _extract_competences(table, file, plan)
        elif _is_attestation_table(table):
            _extract_attestation(table, file, plan)
        else:
            _extract_entries(table, file, plan)
    if not plan.competences:
        plan.warnings.append("не найдена таблица компетенций (1.1)")
    if not plan.entries:
        plan.warnings.append("не найдена таблица дисциплин (раздел 2)")
    check_plan(plan)
    attach_quotes(plan, text)
    return plan


def extract_plan_file(path: Path) -> CurriculumPlan:
    return extract_plan(path.read_text(encoding="utf-8"), file=path.name)
