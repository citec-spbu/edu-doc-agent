"""Единицы проверки: что получает правило по `applies_to` (ADR-0003, «Каталог правил»).

- `course`: одна РПД или РПП (`RPDExtraction`);
- `documents`: дисциплина вместе со строками учебного плана о ней (`CourseDocuments`);
- `program`: все документы программы (`ProgramDocuments`).
"""

from dataclasses import dataclass, field
from typing import Any

from edu_doc_agent.extraction.characteristic import ProgramCharacteristic
from edu_doc_agent.extraction.ckmv import CkmvDocument
from edu_doc_agent.extraction.curriculum import CurriculumPlan, PlanEntry
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.matrix import CompetenceMatrix
from edu_doc_agent.extraction.models import RPDExtraction


@dataclass(frozen=True)
class CourseDoc:
    document_id: str
    data: RPDExtraction
    filename: str
    text: str = ""


@dataclass(frozen=True)
class ProgramDocuments:
    courses: tuple[CourseDoc, ...] = ()
    plan: CurriculumPlan | None = None
    characteristic: ProgramCharacteristic | None = None
    ckmv: CkmvDocument | None = None
    matrix: CompetenceMatrix | None = None
    texts: dict[str, str] = field(default_factory=dict)  # имя файла → Markdown
    # файлы каждого вида документа программы («plan» → …): проверяется последний загруженный,
    # несколько файлов одного вида — замечание
    files_by_kind: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class CourseDocuments:
    course: RPDExtraction
    plan: CurriculumPlan | None
    plan_entries: tuple[PlanEntry, ...]  # строки плана этой дисциплины (по семестрам, профилям)
    text: str = ""  # Markdown документа дисциплины: для проверки структуры разделов


@dataclass(frozen=True)
class Unit:
    """Единица проверки правила: документ дисциплины или программа целиком."""

    document_id: str | None  # None: программа
    course: str | None  # название дисциплины для протокола
    data: Any  # RPDExtraction | CourseDocuments | ProgramDocuments
    files: tuple[str, ...]  # файлы, из которых можно цитировать


def plan_entries_for(course: RPDExtraction, plan: CurriculumPlan | None) -> tuple[PlanEntry, ...]:
    """Строки плана дисциплины: по коду дисциплины, без кода — по названию."""
    if plan is None:
        return ()
    if course.discipline_code:
        found = tuple(e for e in plan.entries if e.code == course.discipline_code)
        if found:
            return found
    if not course.title:
        return ()
    title = normalize(course.title)
    return tuple(e for e in plan.entries if normalize(e.name).startswith(title))


def _plan_file(program: ProgramDocuments) -> tuple[str, ...]:
    return (program.plan.file,) if program.plan else ()


def units(applies_to: str, doc_kinds: list[str], program: ProgramDocuments) -> list[Unit]:
    if applies_to == "program":
        files = tuple(program.texts)
        return [Unit(None, None, program, files)]
    out = []
    for c in program.courses:
        # вид документа не определён: правило применяется, чтобы не потерять проверку
        if c.data.doc_kind is not None and c.data.doc_kind not in doc_kinds:
            continue
        if applies_to == "course":
            out.append(Unit(c.document_id, c.data.title, c.data, (c.filename,)))
        else:
            entries = plan_entries_for(c.data, program.plan)
            data = CourseDocuments(c.data, program.plan, entries, c.text)
            out.append(Unit(c.document_id, c.data.title, data, (c.filename, *_plan_file(program))))
    return out
