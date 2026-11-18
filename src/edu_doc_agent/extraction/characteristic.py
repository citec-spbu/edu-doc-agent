"""Извлечение сущностей из общей характеристики ОП (docx → Markdown).

Детерминированная часть подхода ADR-0002: титульные реквизиты, перечни по разделам 1.x,
виды деятельности с кодами ОКВЭД и задачами, профстандарты, компетенции из разделов 2.x,
таблица соответствия компетенций трудовым функциям (раздел 3). Свободный текст (аннотация,
миссия, разделы 4–9) сохраняется как есть для последующего LLM-разбора. У каждой сущности
есть источник, а расхождения внутри документа идут в `warnings`.

Номера разделов 2.x в версиях документа разные (в одной УНК вынесены в отдельный раздел 2.2),
поэтому группа компетенций определяется по тексту заголовка, а не по номеру.
"""

import re
from pathlib import Path

from pydantic import Field

from edu_doc_agent.extraction.characteristic_llm import (
    CharacteristicLLM,
    extract_characteristic_llm,
)
from edu_doc_agent.extraction.md_utils import MdSection, clean_text, parse_tables, split_sections
from edu_doc_agent.extraction.models import _Model
from edu_doc_agent.extraction.patterns import COMPETENCE_CODE, code_key
from edu_doc_agent.extraction.source import SourceRef, attach_quotes
from edu_doc_agent.llm import LLMClient

_COMPETENCE_LINE_RE = re.compile(rf"^({COMPETENCE_CODE})[.:]?\s+(\S.*)$")
_STANDARD_RE = re.compile(r"^Код\s+(\d{2}\.\d{3})\s+Профессиональный стандарт\s+«(.+?)»\s*(.*)$")
_OKVED_RE = re.compile(r"^Код ОКВЭД\s+(\S+)\s+[-–]\s+(.*?)[;.]?$")
_AREA_RE = re.compile(r"^(\d{2})\s+(\S.*?)[;.]?$")
# «06.001.A.3», «06.015.А.4» (кириллическая А), «06.042.B/04.7»
_FUNCTION_RE = re.compile(r"^(\d{2}\.\d{3})\.([A-ZА-Я])(?:/\d+)?\.(\d+)\s+(\S.*?)\.?$")
_HEADING_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s*")
_HEADING_MARK_RE = re.compile(r"^#+\s*")
_LATIN_BY_CYRILLIC = str.maketrans("АВСЕНКМОРТХ", "ABCEHKMOPTX")

# заголовок раздела 2.x → группа компетенций; порядок важен: более частные признаки раньше
_GROUPS = (
    ("надпрофессиональных", "универсальные надпрофессиональные"),
    ("общепрофессиональных", "общепрофессиональные"),  # тоже «предусмотренных ФГОС»
    ("образовательным стандартом спбгу", "универсальные (СПбГУ)"),
    ("предусмотренных фгос", "универсальные (ФГОС)"),
    ("академическую", "профессиональные академические"),
    ("практическую", "профессиональные практические"),
)


class TextItem(_Model):
    text: str
    source: SourceRef


class ProfessionalArea(_Model):
    code: str
    name: str
    source: SourceRef


class OkvedCode(_Model):
    code: str
    name: str
    source: SourceRef


class ActivityKind(_Model):
    """Вид деятельности: коды ОКВЭД из 1.4 и задачи из 1.5 с тем же названием."""

    name: str
    okved: list[OkvedCode] = Field(default_factory=list)
    tasks: list[TextItem] = Field(default_factory=list)
    source: SourceRef


class ProfessionalStandard(_Model):
    code: str = Field(description="«06.001»")
    name: str
    details: str | None = Field(default=None, description="Реквизиты приказа, как в тексте")
    source: SourceRef


class CharCompetence(_Model):
    code: str
    name: str
    group: str = Field(description="«универсальные (ФГОС)», «профессиональные практические», …")
    section: str | None = Field(default=None, description="Номер раздела 2.x в этом документе")
    source: SourceRef


class LaborFunction(_Model):
    code: str = Field(description="Как в тексте: «06.015.А.4» (бывает кириллическая А)")
    standard_code: str
    function_letter: str = Field(description="Буква обобщённой трудовой функции, латиницей")
    level: int
    name: str
    source: SourceRef


class CompetenceMapping(_Model):
    """Строка таблицы раздела 3: компетенция и её трудовые функции."""

    competence_code: str
    functions: list[LaborFunction] = Field(default_factory=list)
    source: SourceRef


class TextSection(_Model):
    number: str | None
    title: str
    text: str
    source: SourceRef


class ProgramCharacteristic(_Model):
    file: str
    sources: dict[str, SourceRef] = Field(
        default_factory=dict, description="Источники полей титула, аннотации и миссии"
    )
    qualification_sources: list[SourceRef] = Field(default_factory=list)
    llm: CharacteristicLLM | None = None
    group_code: str | None = Field(default=None, description="«02.00.00»")
    group_name: str | None = None
    direction_code: str | None = Field(default=None, description="«02.03.02»")
    direction: str | None = None
    level: str | None = None
    profile: str | None = None
    profile_en: str | None = None
    program_name: str | None = None
    program_name_en: str | None = None
    program_code: str | None = Field(default=None, description="Шифр программы: «СВ.5003.2026»")
    study_form: str | None = None
    languages: str | None = None
    duration: str | None = None
    annotation: str | None = None
    mission: str | None = None
    qualification: list[str] = Field(default_factory=list)
    areas: list[ProfessionalArea] = Field(default_factory=list)
    objects: list[TextItem] = Field(default_factory=list)
    activities: list[ActivityKind] = Field(default_factory=list)
    standards: list[ProfessionalStandard] = Field(default_factory=list)
    employers: list[TextItem] = Field(default_factory=list)
    competences: list[CharCompetence] = Field(default_factory=list)
    mappings: list[CompetenceMapping] = Field(default_factory=list)
    sections: list[TextSection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --- вспомогательное ---------------------------------------------------------------------------


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _lines(section: MdSection) -> list[tuple[int, str]]:
    """Непустые строки тела раздела с номером строки md (без таблиц)."""
    return [
        (section.line + 1 + i, _one_line(line))
        for i, line in enumerate(section.text.split("\n"))
        if line.strip()
    ]


def _ref(section: MdSection | None, line: int, file: str) -> SourceRef:
    return SourceRef(file=file, section=section.heading if section else None, line=line)


def _bullet(text: str) -> str | None:
    return text[1:].strip() if text.startswith(("-", "•")) else None


def _find(sections: list[MdSection], number: str) -> MdSection | None:
    return next((s for s in sections if s.number == number), None)


def function_key(code: str) -> str:
    """Код трудовой функции с латинскими буквами: «06.015.А.4» (кириллица) = «06.015.A.4»."""
    return code.upper().translate(_LATIN_BY_CYRILLIC)


# --- титул ------------------------------------------------------------------------------------


def _extract_title(text: str, file: str, doc: ProgramCharacteristic) -> None:
    lines = text.splitlines()
    end = next((i for i, ln in enumerate(lines) if re.match(r"^#\s+1\.", ln)), len(lines))
    head = [(i + 1, _HEADING_MARK_RE.sub("", clean_text(ln))) for i, ln in enumerate(lines[:end])]
    head = [(n, ln) for n, ln in head if ln]

    def value_after(prefix: str) -> tuple[int, str] | None:
        for idx, (_, ln) in enumerate(head):
            if ln.startswith(prefix) and idx + 1 < len(head):
                value = head[idx + 1]
                if re.match(r"^\d{2}\.\d{2}\.\d{2}\s+\S", value[1]):
                    return value
                doc.warnings.append(f"строка {value[0]}: некорректный код после «{ln}»")
        return None

    def split_code(value: str) -> tuple[str, str]:
        code, _, name = value.partition(" ")
        return code, name.strip()

    if found := value_after("Код, наименование укрупненной группы"):
        doc.group_code, doc.group_name = split_code(found[1])
        for field in ("group_code", "group_name"):
            doc.sources[field] = SourceRef(file=file, section="Титул", line=found[0])
    if found := value_after("Код, наименование направления"):
        doc.direction_code, doc.direction = split_code(found[1])
        for field in ("direction_code", "direction"):
            doc.sources[field] = SourceRef(file=file, section="Титул", line=found[0])
    for n, ln in head:
        for prefix, attr in (
            ("Уровень ", "level"),
            ("Шифр программы ", "program_code"),
            ("Форма(ы) обучения:", "study_form"),
            ("Язык(и) обучения:", "languages"),
            ("Срок(и) обучения:", "duration"),
        ):
            if ln.startswith(prefix) and getattr(doc, attr) is None:
                setattr(doc, attr, ln[len(prefix) :].strip() or None)
                if getattr(doc, attr) is not None:
                    doc.sources[attr] = SourceRef(file=file, section="Титул", line=n)
    for table in parse_tables("\n".join(lines[:end])):
        if len(table.rows) < 2 or len(table.rows[0]) < 2:
            continue
        header = _one_line(clean_text(table.rows[0][0])).lower()
        ru, en = (_one_line(clean_text(c)) or None for c in table.rows[1][:2])
        if header.startswith("направленность"):
            doc.profile, doc.profile_en = ru, en
            fields = ("profile", "profile_en")
        elif header.startswith("наименование программы"):
            doc.program_name, doc.program_name_en = ru, en
            fields = ("program_name", "program_name_en")
        else:
            continue
        for field in fields:
            if getattr(doc, field) is not None:
                doc.sources[field] = SourceRef(file=file, section="Титул", line=table.row_lines[1])


# --- разделы 1.x ------------------------------------------------------------------------------


def _extract_part1(sections: list[MdSection], file: str, doc: ProgramCharacteristic) -> None:
    if s := _find(sections, "1.1"):
        doc.qualification = [t.rstrip(".;") for _, t in _lines(s)]
        doc.qualification_sources = [_ref(s, n, file) for n, _ in _lines(s)]
    if s := _find(sections, "1.2"):
        for n, t in _lines(s):
            if m := _AREA_RE.match(t):
                doc.areas.append(
                    ProfessionalArea(code=m.group(1), name=m.group(2), source=_ref(s, n, file))
                )
    if s := _find(sections, "1.3"):
        for n, t in _lines(s):
            if (item := _bullet(t)) is not None:
                doc.objects.append(TextItem(text=item, source=_ref(s, n, file)))
    _extract_activities(sections, file, doc)
    if s := _find(sections, "1.6"):
        for n, t in _lines(s):
            if m := _STANDARD_RE.match(t):
                details = m.group(3).strip().rstrip(".").strip("() ") or None
                doc.standards.append(
                    ProfessionalStandard(
                        code=m.group(1), name=m.group(2), details=details, source=_ref(s, n, file)
                    )
                )
    if s := _find(sections, "1.7"):
        for n, t in _lines(s):
            doc.employers.append(TextItem(text=t.rstrip(";."), source=_ref(s, n, file)))


def _extract_activities(sections: list[MdSection], file: str, doc: ProgramCharacteristic) -> None:
    kinds: dict[str, ActivityKind] = {}

    def kind_of(name: str, section: MdSection, line: int) -> ActivityKind:
        key = _one_line(name).casefold()
        if key not in kinds:
            kinds[key] = ActivityKind(name=name, source=_ref(section, line, file))
        return kinds[key]

    if s := _find(sections, "1.4"):
        current: ActivityKind | None = None
        for n, t in _lines(s):
            if m := _OKVED_RE.match(t):
                if current is not None:
                    current.okved.append(
                        OkvedCode(code=m.group(1), name=m.group(2), source=_ref(s, n, file))
                    )
            elif t.endswith(":"):
                current = kind_of(t.rstrip(":"), s, n)
    if s := _find(sections, "1.5"):
        current = None
        for n, t in _lines(s):
            if t.endswith(":") and not t.startswith("-"):
                current = kind_of(t.rstrip(":"), s, n)
            elif current is not None and (item := _bullet(t)) is not None:
                current.tasks.append(TextItem(text=item, source=_ref(s, n, file)))
    doc.activities = list(kinds.values())


# --- разделы 2.x и 3 --------------------------------------------------------------------------


def _group_of(heading: str) -> str | None:
    low = _HEADING_NUMBER_RE.sub("", heading).lower()
    if not low.startswith("перечень") or "компетенций" not in low:
        return None
    return next((label for key, label in _GROUPS if key in low), None)


def _extract_competences(sections: list[MdSection], file: str, doc: ProgramCharacteristic) -> None:
    for s in sections:
        if not (s.number or "").startswith("2.") or (group := _group_of(s.heading)) is None:
            continue
        for n, t in _lines(s):
            if m := _COMPETENCE_LINE_RE.match(t):
                doc.competences.append(
                    CharCompetence(
                        code=m.group(1),
                        name=m.group(2).rstrip(".").strip(),
                        group=group,
                        section=s.number,
                        source=_ref(s, n, file),
                    )
                )
            elif re.match(r"^(?:УК[А-Я]*|ОПК|ПК[А-Я]*|УНК)[\s\d-]", t):
                doc.warnings.append(f"строка {n}: не разобрана компетенция «{t[:60]}»")


def _extract_mapping(text: str, file: str, doc: ProgramCharacteristic) -> None:
    for table in parse_tables(text):
        head = _one_line(clean_text(" ".join(table.rows[0]))).lower() if table.rows else ""
        if not ("перечень компетенций" in head and "трудовые функции" in head):
            continue
        for idx, row in enumerate(table.rows[1:], start=1):
            if len(row) < 2 or not (code := _one_line(clean_text(row[0]))):
                continue
            ref = SourceRef(file=file, section=table.section, line=table.row_lines[idx])
            mapping = CompetenceMapping(competence_code=code, source=ref)
            for part in clean_text(row[1]).split("\n"):
                if m := _FUNCTION_RE.match(_one_line(part)):
                    mapping.functions.append(
                        LaborFunction(
                            code=_one_line(part).split(" ", 1)[0],
                            standard_code=m.group(1),
                            function_letter=function_key(m.group(2)),
                            level=int(m.group(3)),
                            name=m.group(4),
                            source=ref,
                        )
                    )
                elif part.strip():
                    doc.warnings.append(
                        f"строка {ref.line}: не разобрана трудовая функция «{part[:50]}»"
                    )
            doc.mappings.append(mapping)


# --- свободный текст --------------------------------------------------------------------------


def _extract_text_parts(sections: list[MdSection], file: str, doc: ProgramCharacteristic) -> None:
    for s in sections:
        body = "\n".join(_one_line(t) for _, t in _lines(s))
        if s.level == 3 and s.heading.startswith("Аннотация"):
            doc.annotation = body or None
            if body:
                doc.sources["annotation"] = _ref(s, _lines(s)[0][0], file)
        elif s.level == 3 and s.heading.startswith("Миссия"):
            doc.mission = body or None
            if body:
                doc.sources["mission"] = _ref(s, _lines(s)[0][0], file)
        elif s.level == 1 and s.number and int(s.number.split(".")[0]) >= 4:
            doc.sections.append(
                TextSection(
                    number=s.number, title=s.heading, text=body, source=_ref(s, s.line, file)
                )
            )


# --- сверки -----------------------------------------------------------------------------------


def check_characteristic(doc: ProgramCharacteristic) -> None:
    """Согласованность внутри документа; расхождения — в `warnings`, ничего не правим."""
    seen: dict[str, CharCompetence] = {}
    for c in doc.competences:
        if (key := code_key(c.code)) in seen:
            doc.warnings.append(f"компетенция {c.code} (строка {c.source.line}): повтор кода")
        seen[key] = c
    mapped = {code_key(m.competence_code) for m in doc.mappings}
    for m in doc.mappings:
        if code_key(m.competence_code) not in seen:
            doc.warnings.append(
                f"раздел 3, строка {m.source.line}: компетенции {m.competence_code} нет в разделе 2"
            )
        if not m.functions:
            doc.warnings.append(
                f"раздел 3, строка {m.source.line}: у {m.competence_code} нет трудовых функций"
            )
    if doc.mappings:
        for c in doc.competences:
            if c.group.startswith("профессиональные") and code_key(c.code) not in mapped:
                doc.warnings.append(
                    f"компетенция {c.code} (строка {c.source.line}): нет в таблице раздела 3"
                )
    standards = {s.code for s in doc.standards}
    if standards:
        for m in doc.mappings:
            for f in m.functions:
                if f.standard_code not in standards:
                    doc.warnings.append(
                        f"раздел 3, строка {f.source.line}: профстандарта {f.standard_code} "
                        "нет в разделе 1.6"
                    )
    for a in doc.activities:
        if a.okved and not a.tasks:
            doc.warnings.append(f"вид деятельности «{a.name}»: нет задач в разделе 1.5")
        if a.tasks and not a.okved:
            doc.warnings.append(f"вид деятельности «{a.name}»: нет кодов ОКВЭД в разделе 1.4")


# --- точка входа ------------------------------------------------------------------------------


def extract_characteristic(
    text: str, file: str = "", *, llm: LLMClient | None = None
) -> ProgramCharacteristic:
    """Характеристика из Markdown; смысловые поля извлекаются при переданном LLM-клиенте."""
    doc = ProgramCharacteristic(file=file)
    sections = split_sections(text)
    _extract_title(text, file, doc)
    _extract_part1(sections, file, doc)
    _extract_competences(sections, file, doc)
    _extract_mapping(text, file, doc)
    _extract_text_parts(sections, file, doc)
    if doc.direction_code is None:
        doc.warnings.append("не найдено направление подготовки в титуле")
    if not doc.competences:
        doc.warnings.append("не найдены перечни компетенций (раздел 2)")
    if not doc.mappings:
        doc.warnings.append("не найдена таблица соответствия трудовым функциям (раздел 3)")
    check_characteristic(doc)
    if llm is not None:
        doc.llm = extract_characteristic_llm(text, file, llm)
        doc.warnings.extend(doc.llm.warnings)
    attach_quotes(doc, text)
    return doc


def extract_characteristic_file(
    path: Path, *, llm: LLMClient | None = None
) -> ProgramCharacteristic:
    return extract_characteristic(path.read_text(encoding="utf-8"), file=path.name, llm=llm)
