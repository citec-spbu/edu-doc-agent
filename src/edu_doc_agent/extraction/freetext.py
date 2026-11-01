"""Извлечение сущностей из свободного текста РПД правилами Yargy.

Таблицы разбирает `rpd.py` регулярками. Здесь — то, что лежит в обычных абзацах:

- пререквизиты (раздел 1.2): дисциплины и области знаний;
- формы контроля и оценочные средства (разделы про контроль, 3.1.x): «2 контрольные работы»;
- трудоёмкость («18 зачётных единиц (648 академических часов)», «Семестр 2: 3 з.е. (108 часов)»);
- семестры («2-го, 3-го и 5-го семестров», «2-3 семестры»).

Yargy нужен ради морфологии: «контрольных работах» и «контрольные работы» — одно и то же,
а «зачётных единиц» не надо перечислять во всех падежах. Правила разбирают каждую строку отдельно:
абзац в md — это одна строка, поэтому перечисления не склеиваются между пунктами списка.
"""

import re
import warnings
from functools import cache

from yargy import Parser, not_, or_, rule
from yargy.interpretation import attribute, fact
from yargy.pipelines import morph_pipeline
from yargy.predicates import eq, gram, in_caseless, normalized
from yargy.predicates import type as token_type

from edu_doc_agent.extraction.md_utils import MdSection, split_sections
from edu_doc_agent.extraction.models import (
    ControlMention,
    Prerequisite,
    RPDExtraction,
    SourceRef,
    Workload,
)

# --- вспомогательные предикаты -----------------------------------------------------------------

INT = token_type("INT")
COMMA = eq(",")
COLON = eq(":")
LPAR, RPAR = eq("("), eq(")")
LQUOTE, RQUOTE = eq("«"), eq("»")
AND = or_(rule(eq("и")), rule(eq("а"), eq("также")))
SEPARATOR = or_(rule(COMMA, AND), rule(COMMA), rule(eq(";")), AND)

# --- формы контроля и оценочные средства -------------------------------------------------------

_INTERMEDIATE = "промежуточный"
_CURRENT = "текущий"
# каноническое название → (роль, формы слова; Yargy сам склоняет)
CONTROL_TERMS: dict[str, tuple[str, list[str]]] = {
    "экзамен": (_INTERMEDIATE, ["экзамен"]),
    "зачёт с оценкой": (_INTERMEDIATE, ["зачет с оценкой", "зачёт с оценкой"]),
    "зачёт": (_INTERMEDIATE, ["зачет", "зачёт"]),
    "контрольная работа": (_CURRENT, ["контрольная работа"]),
    "коллоквиум": (_CURRENT, ["коллоквиум"]),
    "тест": (_CURRENT, ["тест", "тестирование"]),
    "домашнее задание": (_CURRENT, ["домашнее задание", "домашняя работа"]),
    "лабораторная работа": (_CURRENT, ["лабораторная работа"]),
    "практическая работа": (_CURRENT, ["практическая работа", "практическое задание"]),
    "расчётно-графическая работа": (_CURRENT, ["расчетно-графическая работа"]),
    "курсовая работа": (_CURRENT, ["курсовая работа", "курсовой проект"]),
    "реферат": (_CURRENT, ["реферат"]),
    "доклад": (_CURRENT, ["доклад"]),
    "эссе": (_CURRENT, ["эссе"]),
    "собеседование": (_CURRENT, ["собеседование"]),
    "устный опрос": (_CURRENT, ["устный опрос"]),
    "командный проект": (_CURRENT, ["командный проект", "групповой проект"]),
}
_ROLE_BY_KIND = {kind: role for kind, (role, _) in CONTROL_TERMS.items()}

Control = fact("Control", ["kind", "count"])
Activity = fact("Activity", [attribute("names").repeatable()])
# «ретроспективный анализ достижений и проблем» — один пункт, поэтому «и» не разделитель
_ACTIVITY_ITEM = (
    not_(in_caseless({",", ";", ".", ":"})).repeatable(max=14).interpretation(Activity.names)
)


def _control_rule():
    terms = or_(
        *[
            morph_pipeline(forms).interpretation(Control.kind.const(kind))
            for kind, (_, forms) in CONTROL_TERMS.items()
        ]
    )
    return rule(INT.interpretation(Control.count.custom(int)).optional(), terms).interpretation(
        Control
    )


# --- пререквизиты ------------------------------------------------------------------------------

Names = fact("Names", [attribute("names").repeatable()])
_STOP = or_(
    in_caseless({",", ";", ".", ":", "«", "»", "(", ")"}),
    eq("и"),
    in_caseless(
        {
            "а",
            "также",
            "который",
            "которые",
            "которых",
            "необходимо",
            "для",
            "при",
            "должен",
            "были",
            "способны",
            "первую",
            "очередь",
            "умение",
            "уметь",
            "владеть",
            "или",
            "либо",
        }
    ),
)
_PLAIN_ITEM = not_(_STOP).repeatable(max=6).interpretation(Names.names)
_QUOTED_ITEM = rule(
    LQUOTE, not_(or_(LQUOTE, RQUOTE)).repeatable(max=12).interpretation(Names.names), RQUOTE
)
_PLAIN_LIST = rule(_PLAIN_ITEM, rule(SEPARATOR, _PLAIN_ITEM).repeatable().optional())
_DISCIPLINE_WORD = or_(normalized("дисциплина"), normalized("курс"))
_KNOWLEDGE_WORD = or_(
    normalized("знание"), normalized("умение"), normalized("навык"), normalized("подготовка")
)
# «знания по дисциплинам: …» — это список дисциплин, а не область знаний
_AREA_ITEM = rule(
    not_(or_(_STOP, _DISCIPLINE_WORD)), not_(_STOP).repeatable(max=5).optional()
).interpretation(Names.names)
_AREA_LIST = rule(_AREA_ITEM, rule(SEPARATOR, _PLAIN_ITEM).repeatable().optional())
# без предлога «по»: «Знание основ комбинаторного анализа»; «по» и «в» тут значат другое правило
_BARE_AREA_ITEM = rule(
    not_(or_(_STOP, _DISCIPLINE_WORD, eq("по"), eq("в"), eq("на"))),
    not_(_STOP).repeatable(max=5).optional(),
).interpretation(Names.names)
_BARE_AREA_LIST = rule(_BARE_AREA_ITEM, rule(SEPARATOR, _PLAIN_ITEM).repeatable().optional())
_IN_AREA = or_(
    rule(eq("по")), rule(eq("в"), normalized("область")), rule(eq("в"), normalized("сфера"))
)

# Группы правил: у каждой свой «вид» пререквизита
PREREQUISITE_GROUPS: dict[str, list] = {
    "дисциплина": [
        _QUOTED_ITEM,  # любое «название» в разделе 1.2 — дисциплина
        rule(_DISCIPLINE_WORD, COLON, _PLAIN_LIST),
        rule(eq("по"), _DISCIPLINE_WORD, _PLAIN_LIST),
        rule(_KNOWLEDGE_WORD, eq("по"), _DISCIPLINE_WORD, _PLAIN_LIST),
    ],
    "область знаний": [
        rule(_KNOWLEDGE_WORD, _IN_AREA, _AREA_LIST),
        # «Знание основ комбинаторного анализа, математического анализа.» — без «по»
        rule(normalized("знание"), _BARE_AREA_LIST),
    ],
}

# --- семестры ----------------------------------------------------------------------------------

SemList = fact("SemList", [attribute("nums").repeatable()])
SemRange = fact("SemRange", ["first", "last"])
_SEMESTER = normalized("семестр")
_ORDINAL = rule(
    INT.interpretation(SemList.nums),
    rule(eq("-"), in_caseless({"й", "го", "м", "ой", "е", "ый", "ая"})).optional(),
)
_SEM_LIST = rule(_ORDINAL, rule(SEPARATOR, _ORDINAL).repeatable().optional(), _SEMESTER)
_SEM_RANGE = or_(
    rule(
        INT.interpretation(SemRange.first.custom(int)),
        eq("-"),
        INT.interpretation(SemRange.last.custom(int)),
        _SEMESTER,
    ),
    rule(
        in_caseless({"с", "со"}),
        INT.interpretation(SemRange.first.custom(int)),
        eq("по"),
        INT.interpretation(SemRange.last.custom(int)),
        _SEMESTER,
    ),
)

# --- трудоёмкость ------------------------------------------------------------------------------

Load = fact("Load", ["semester", "credits", "hours"])


def _to_float(value: str) -> float:
    return float(value.replace(" ", "").replace(",", "."))


_CREDITS_NUM = rule(INT, rule(COMMA, INT).optional()).interpretation(Load.credits.custom(_to_float))
_CREDIT_UNIT = or_(
    rule(morph_pipeline(["зачетная единица", "зачётная единица"])),
    rule(eq("з"), eq("."), eq("е"), eq(".").optional()),
)
_HOURS = rule(
    INT.interpretation(Load.hours.custom(int)), gram("ADJF").optional(), normalized("час")
)
_SEM_PREFIX = rule(_SEMESTER, INT.interpretation(Load.semester.custom(int)), COLON.optional())
_CREDITS_WITH_UNIT = rule(_CREDITS_NUM, _CREDIT_UNIT)
_LOAD = or_(
    rule(
        _SEM_PREFIX.optional(),
        _CREDITS_WITH_UNIT,
        rule(LPAR.optional(), _HOURS, RPAR.optional()).optional(),
    ),
    rule(_SEM_PREFIX, _HOURS),
    rule(_HOURS, LPAR, _CREDITS_WITH_UNIT, RPAR),
).interpretation(Load)

# --- парсеры (строятся один раз: загрузка словаря pymorphy занимает около секунды) --------------


@cache
def _parsers() -> dict[str, Parser]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "control": Parser(_control_rule()),
            "activity": Parser(
                rule(
                    normalized("вид"),
                    COLON,
                    rule(
                        _ACTIVITY_ITEM,
                        rule(or_(rule(COMMA), rule(eq(";"))), _ACTIVITY_ITEM)
                        .repeatable()
                        .optional(),
                    ),
                ).interpretation(Activity)
            ),
            **{
                f"prereq:{kind}": Parser(or_(*rules).interpretation(Names))
                for kind, rules in PREREQUISITE_GROUPS.items()
            },
            "sem_list": Parser(_SEM_LIST.interpretation(SemList)),
            "sem_range": Parser(_SEM_RANGE.interpretation(SemRange)),
            "load": Parser(_LOAD),
        }


# --- разбор ------------------------------------------------------------------------------------

_SPACE_FIX = [
    (re.compile(r"\s+([,.;:)])"), r"\1"),
    (re.compile(r"([(])\s+"), r"\1"),
    (re.compile(r"\s*([+#/-])\s*"), r"\1"),
]
_CHECKED = "🗹☑☒✓✔■\uf0fe"  # \uf0fe — галочка из шрифта Wingdings
_UNCHECKED = "□☐\uf0a8"
_MIN_NAME_LEN = 3


def _detokenize(value: str) -> str:
    """Yargy склеивает токены пробелами («С + +»): возвращаем обычную типографику."""
    for pattern, repl in _SPACE_FIX:
        value = pattern.sub(repl, value)
    return value.strip(" ,;.:")


def _as_list(value: object) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


def _lines(section: MdSection) -> list[tuple[int, str]]:
    return [(section.line + 1 + i, line) for i, line in enumerate(section.text.split("\n")) if line]


def _is_control_section(section: MdSection) -> bool:
    number = section.number or ""
    if number.startswith("3.1.4"):  # примеры КИМ — не описание контроля
        return False
    if number.startswith("3.1.3"):
        return True
    heading = section.heading.casefold()
    return number.startswith("3.1") and ("контрол" in heading or "аттестац" in heading)


def _is_checked(line: str, position: int) -> bool:
    """Для строк с чекбоксами «🗹 зачет □ экзамен»: отмечен ли пункт перед `position`."""
    marks = [c for c in line[:position] if c in _CHECKED or c in _UNCHECKED]
    return not marks or marks[-1] in _CHECKED


def _ref(section: MdSection, line: int, file: str) -> SourceRef:
    return SourceRef(file=file, section=section.heading, line=line)


def extract_free_text(text: str, file: str, result: RPDExtraction) -> None:
    """Дополняет `result` сущностями из свободного текста; таблицы пропускает."""
    parsers = _parsers()
    sections = split_sections(text)
    title = (result.title or "").casefold()
    _extract_prerequisites(sections, parsers, file, title, result)
    _extract_control(sections, parsers, file, result)
    _extract_workload_and_semesters(sections, parsers, file, result)


def _extract_prerequisites(
    sections: list[MdSection],
    parsers: dict[str, Parser],
    file: str,
    title: str,
    result: RPDExtraction,
) -> None:
    seen: set[str] = set()
    for section in sections:
        if section.number != "1.2":
            continue
        for lineno, raw_line in _lines(section):
            # в части документов названия в прямых кавычках: "Основы программирования"
            line = re.sub(r'"([^"«»]+)"', r"«\1»", raw_line)
            for kind in PREREQUISITE_GROUPS:
                for match in parsers[f"prereq:{kind}"].findall(line):
                    for raw in _as_list(match.fact.names):
                        name = _detokenize(raw)
                        key = name.casefold()
                        if len(name) < _MIN_NAME_LEN or key in seen or key == title:
                            continue
                        seen.add(key)
                        result.prerequisites.append(
                            Prerequisite(name=name, kind=kind, source=_ref(section, lineno, file))
                        )


def _extract_control(
    sections: list[MdSection], parsers: dict[str, Parser], file: str, result: RPDExtraction
) -> None:
    found: dict[str, ControlMention] = {}
    for section in filter(_is_control_section, sections):
        for lineno, line in _lines(section):
            for match in parsers["control"].findall(line):
                if not _is_checked(line, match.span.start):
                    continue
                kind, count = match.fact.kind, match.fact.count
                if kind in found:
                    mention = found[kind]
                    mention.occurrences += 1
                    if count is not None:
                        mention.count = max(mention.count or 0, count)
                else:
                    found[kind] = ControlMention(
                        kind=kind,
                        role=_ROLE_BY_KIND[kind],
                        count=count,
                        source=_ref(section, lineno, file),
                    )
            for match in parsers["activity"].findall(line):
                for raw in _as_list(match.fact.names):
                    name = _detokenize(raw).casefold()
                    if len(name) >= _MIN_NAME_LEN and name not in found:
                        found[name] = ControlMention(
                            kind=name, role=_CURRENT, source=_ref(section, lineno, file)
                        )
    result.control_mentions.extend(found.values())


def _extract_workload_and_semesters(
    sections: list[MdSection], parsers: dict[str, Parser], file: str, result: RPDExtraction
) -> None:
    loads: dict[tuple, Workload] = {}
    semesters: set[int] = set()
    for section in sections:
        if not (section.number or "").startswith("2.1"):
            continue
        for lineno, line in _lines(section):
            for match in parsers["load"].findall(line):
                f = match.fact
                key = (f.semester, f.credits, f.hours)
                if key not in loads and (f.credits is not None or f.hours is not None):
                    loads[key] = Workload(
                        semester=f.semester,
                        credits=f.credits,
                        hours=f.hours,
                        source=_ref(section, lineno, file),
                    )
                if f.semester is not None:
                    semesters.add(f.semester)
            for match in parsers["sem_list"].findall(line):
                semesters.update(int(n) for n in _as_list(match.fact.nums))
            for match in parsers["sem_range"].findall(line):
                first, last = match.fact.first, match.fact.last
                if first <= last:
                    semesters.update(range(first, last + 1))
    result.workload.extend(loads.values())
    result.semesters = sorted(semesters)
