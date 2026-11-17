"""Извлечение сущностей из ЦКМВ (docx → Markdown): роли, компетенции, индикаторы, результаты.

Документ — одна четырёхстолбцовая таблица: роль → компетенция → индикатор (с уровнем) → результаты
обучения «З-/У-/В-». Роль и компетенция заполнены только в первой строке группы, остальные ячейки
пусты (вертикальное объединение); несколько результатов могут лежать в одной ячейке через `<br>`.
Группа компетенций («… базового уровня», «… топ-уровня») задаётся строкой-шапкой с заголовками
«Индикаторы …» и «Результаты обучения …». Как и в РПД (ADR-0002), всё детерминированно, у каждой
сущности есть источник, а расхождения контрактов идут в `warnings`.
"""

import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from edu_doc_agent.extraction.md_utils import MdTable, clean_text, parse_tables
from edu_doc_agent.extraction.models import _Model
from edu_doc_agent.extraction.patterns import COMPETENCE_CODE, INDICATOR_CODE, code_key
from edu_doc_agent.extraction.source import SourceRef, attach_quotes

OutcomeLetter = Literal["З", "У", "В"]
_KIND_BY_LETTER = {"З": "знает", "У": "умеет", "В": "владеет"}
_LEVELS = ("Базовый", "Средний", "Продвинутый")

_COMPETENCE_RE = re.compile(rf"^({COMPETENCE_CODE})\.?\s*(.*)$", re.S)
_LEVEL_RE = re.compile(rf"\s*\(({'|'.join(_LEVELS)})\)\s*$")
_INDICATOR_RE = re.compile(rf"^({INDICATOR_CODE})\.?\s*(.*)$", re.S)
# З — знает, У — умеет, В — владеет; встречается опечатка «3-1» (цифра вместо З)
_OUTCOME_RE = re.compile(rf"^({INDICATOR_CODE})\.?\s*([ЗУВ3])-(\d+)\.?\s*(.*)$", re.S)


class CkmvOutcome(_Model):
    """Результат обучения по индикатору: «ПКП-МО-1.1 З-1. Знает …»."""

    code: str = Field(description="Как в тексте, без хвостовой точки: «ПКП-МО-1.1 З-1»")
    letter: OutcomeLetter
    number: int
    kind: Literal["знает", "умеет", "владеет"]
    text: str
    source: SourceRef


class CkmvIndicator(_Model):
    code: str
    name: str
    level: str | None = Field(default=None, description="Базовый / Средний / Продвинутый")
    outcomes: list[CkmvOutcome] = Field(default_factory=list)
    source: SourceRef


class CkmvCompetence(_Model):
    code: str = Field(description="Как в ЦКМВ: «ПКП-2-ППК-Р2», в плане этот код — «ПКП-ППК-Р2»")
    name: str
    group: str | None = Field(default=None, description="«Профессиональные компетенции базового…»")
    role_section: str | None = Field(
        default=None, description="«Основная профессиональная роль», «Универсальные компетенции»"
    )
    role: str | None = Field(default=None, description="Название роли, если она задана")
    indicators: list[CkmvIndicator] = Field(default_factory=list)
    source: SourceRef


class CkmvDocument(_Model):
    file: str
    competences: list[CkmvCompetence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def indicators(self) -> list[CkmvIndicator]:
        return [i for c in self.competences for i in c.indicators]

    @property
    def outcomes(self) -> list[CkmvOutcome]:
        return [o for i in self.indicators for o in i.outcomes]


# --- разбор -----------------------------------------------------------------------------------


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _cell(row: list[str], col: int) -> str:
    return clean_text(row[col]) if col < len(row) else ""


def _is_group_header(row: list[str]) -> bool:
    return _cell(row, 2).startswith("Индикаторы") and _cell(row, 3).startswith("Результаты")


def _ref(table: MdTable, idx: int, file: str) -> SourceRef:
    return SourceRef(file=file, section=table.section, line=table.row_lines[idx])


def _is_ckmv_table(table: MdTable) -> bool:
    return bool(table.rows) and len(table.rows[0]) == 4 and _is_group_header(table.rows[0])


def _parse_indicator(text: str) -> tuple[str, str, str | None] | None:
    text = _one_line(text)
    level = None
    if m := _LEVEL_RE.search(text):
        level, text = m.group(1), text[: m.start()]
    if not (m := _INDICATOR_RE.match(text)):
        return None
    return m.group(1), m.group(2).strip().rstrip(".").strip(), level


def _extract_table(table: MdTable, file: str, doc: CkmvDocument) -> None:
    role_section = role = group = None
    competence: CkmvCompetence | None = None
    indicator: CkmvIndicator | None = None
    last_competence = last_indicator = ""

    for idx, row in enumerate(table.rows):
        c0, c1, c2, c3 = (_cell(row, i) for i in range(4))
        ref = _ref(table, idx, file)
        if c0:
            only_label = not (c1 or c2 or c3)
            if c0.endswith(":") or only_label:
                role_section, role = _one_line(c0).rstrip(":"), None
            else:
                role = _one_line(c0)
        if _is_group_header(row):
            group, competence, indicator = _one_line(c1), None, None
            last_competence = last_indicator = ""
            continue
        if c1 and _one_line(c1) != last_competence:
            last_competence = _one_line(c1)
            if m := _COMPETENCE_RE.match(last_competence):
                competence = CkmvCompetence(
                    code=m.group(1),
                    name=m.group(2).strip(),
                    group=group,
                    role_section=role_section,
                    role=role,
                    source=ref,
                )
                doc.competences.append(competence)
                indicator = None
                last_indicator = ""
            else:
                competence = indicator = None
                doc.warnings.append(f"строка {ref.line}: не разобрана компетенция «{c1[:60]}»")
        if c2 and _one_line(c2) != last_indicator:
            last_indicator = _one_line(c2)
            parsed = _parse_indicator(last_indicator)
            if parsed is None or competence is None:
                indicator = None
                doc.warnings.append(f"строка {ref.line}: не разобран индикатор «{c2[:60]}»")
            else:
                code, name, level = parsed
                indicator = CkmvIndicator(code=code, name=name, level=level, source=ref)
                competence.indicators.append(indicator)
        pending = ""
        for part in (p.strip() for p in c3.split("\n") if p.strip()):
            if _OUTCOME_RE.match(part):
                if pending:
                    _add_outcome(pending, indicator, ref, doc)
                pending = part
            elif pending:
                pending += " " + part
            else:
                _add_outcome(part, indicator, ref, doc)
        if pending:
            _add_outcome(pending, indicator, ref, doc)


def _add_outcome(
    part: str, indicator: CkmvIndicator | None, ref: SourceRef, doc: CkmvDocument
) -> None:
    part = _one_line(part)
    m = _OUTCOME_RE.match(part)
    if m is None or indicator is None:
        doc.warnings.append(f"строка {ref.line}: не разобран результат «{part[:60]}»")
        return
    ind_code, letter, number, text = m.groups()
    letter = "З" if letter == "3" else letter
    indicator.outcomes.append(
        CkmvOutcome(
            code=f"{ind_code} {letter}-{number}",
            letter=letter,  # type: ignore[arg-type]
            number=int(number),
            kind=_KIND_BY_LETTER[letter],  # type: ignore[arg-type]
            text=text.strip(),
            source=ref,
        )
    )


# --- сверки -----------------------------------------------------------------------------------


def _base_code(indicator_code: str) -> str:
    return indicator_code.rsplit(".", 1)[0]


def check_ckmv(doc: CkmvDocument) -> None:
    """Согласованность кодов и видов результатов; расхождения — в `warnings`, ничего не правим."""
    for comp in doc.competences:
        if not comp.indicators:
            doc.warnings.append(
                f"компетенция {comp.code} (строка {comp.source.line}): нет индикаторов"
            )
        for ind in comp.indicators:
            where = f"индикатор {ind.code} (строка {ind.source.line})"
            if code_key(_base_code(ind.code)) != code_key(comp.code):
                doc.warnings.append(f"{where}: код не относится к компетенции {comp.code}")
            if ind.level is None:
                doc.warnings.append(f"{where}: не указан уровень")
            if not ind.outcomes:
                doc.warnings.append(f"{where}: нет результатов обучения")
            seen: set[tuple[str, int]] = set()
            for out in ind.outcomes:
                owner = out.code.rsplit(" ", 1)[0]
                if code_key(owner) != code_key(ind.code):
                    doc.warnings.append(
                        f"результат {out.code} (строка {out.source.line}): код не относится "
                        f"к индикатору {ind.code}"
                    )
                if (out.letter, out.number) in seen:
                    doc.warnings.append(f"результат {out.code} (строка {out.source.line}): повтор")
                seen.add((out.letter, out.number))
                # «Способен …», «Понимает …» букве не противоречат: тревожит чужой глагол тройки
                first = out.text.lower().split(" ", 1)[0]
                if first in _KIND_BY_LETTER.values() and first != out.kind:
                    doc.warnings.append(
                        f"результат {out.code} (строка {out.source.line}): буква "
                        f"«{out.letter}» ({out.kind}), а текст начинается с «{first}»"
                    )


# --- точка входа ------------------------------------------------------------------------------


def extract_ckmv(text: str, file: str = "") -> CkmvDocument:
    """ЦКМВ из Markdown (`parse_document(...).to_markdown()`)."""
    doc = CkmvDocument(file=file)
    tables = [t for t in parse_tables(text) if _is_ckmv_table(t)]
    for table in tables:
        _extract_table(table, file, doc)
    if not tables:
        doc.warnings.append("не найдена таблица ролей и компетенций")
    check_ckmv(doc)
    attach_quotes(doc, text)
    return doc


def extract_ckmv_file(path: Path) -> CkmvDocument:
    return extract_ckmv(path.read_text(encoding="utf-8"), file=path.name)
