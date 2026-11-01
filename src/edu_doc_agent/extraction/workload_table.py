"""Таблица «Трудоёмкость, объёмы учебной работы»: часы по видам занятий в каждом семестре.

Заголовок двухуровневый: группа («Контактная работа» / «Аудиторная работа», «Самостоятельная
работа») и вид занятий под ней. Число и порядок столбцов в документах разные (17–36 из-за
объединённых ячеек), названия тоже («Трудоёмкость» / «Трудоёмкость, ед.», «Контактная» /
«Аудиторная», «лекции» / «Лекции»). Поэтому столбцы опознаём не по номеру и не по точному тексту,
а по смыслу заголовка: словарь Yargy сопоставляет ячейку виду занятий в любой форме слова.
"""

import re
import warnings
from functools import cache

from yargy import Parser, or_, rule
from yargy.pipelines import morph_pipeline
from yargy.predicates import eq, normalized
from yargy.predicates import type as token_type

from edu_doc_agent.extraction.md_utils import MdTable, clean_text
from edu_doc_agent.extraction.models import RPDExtraction, SourceRef, WorkloadRow
from edu_doc_agent.extraction.patterns import STUDY_FORM_RE, TRAJECTORY_ROW_RE

CONTACT = "контактная"
SELF_STUDY = "самостоятельная"
# каноническое название → формы слова для заголовка
CONTACT_FIELDS = {
    "лекции": ["лекция"],
    "семинары": ["семинар"],
    "консультации": ["консультация"],
    "практические занятия": ["практическое занятие"],
    "лабораторные работы": ["лабораторная работа"],
    "контрольные работы": ["контрольная работа"],
    "коллоквиумы": ["коллоквиум"],
    "текущий контроль": ["текущий контроль"],
    "промежуточная аттестация": ["промежуточная аттестация"],
    "итоговая аттестация": ["итоговая аттестация"],
}
SELF_STUDY_FIELDS = {
    "под руководством преподавателя": ["под руководством преподавателя"],
    "в присутствии преподавателя": ["в присутствии преподавателя"],
    # «сам.раб. с использованием методических материалов», «В т. ч. с использованием
    # учебно-методич. материалов» — слова после «с использованием» пишут и сокращают по-разному
    "с использованием методических материалов": ["с использованием"],
    "текущий контроль": ["текущий контроль"],
    "промежуточная аттестация": ["промежуточная аттестация"],
    "итоговая аттестация": ["итоговая аттестация"],
}
GROUPS = {
    CONTACT: ["контактная работа", "аудиторная работа"],
    SELF_STUDY: ["самостоятельная работа"],
}
ACTIVE = "активные формы"
CREDITS = "трудоёмкость"
_ACTIVE_FORMS = ["активные и интерактивные формы", "активная и интерактивная форма"]
_CREDIT_FORMS = ["трудоемкость", "трудоёмкость"]

HOURS_PER_CREDIT = 36
_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
_RANGE_RE = re.compile(r"^\d+-\d+$")  # «1-60» — размер группы, а не часы
_MIN_LEAVES = 2  # вид занятий + «Трудоёмкость»: так заголовок отличается от прочих таблиц
_EPS = 1e-9


@cache
def _parsers() -> dict[str, Parser]:
    """Парсеры строим один раз: загрузка словаря pymorphy занимает около секунды."""
    vocab: dict[str, list[str]] = {}
    for name, forms in {**CONTACT_FIELDS, **SELF_STUDY_FIELDS}.items():
        vocab.setdefault(name, forms)
    vocab[ACTIVE] = _ACTIVE_FORMS
    vocab[CREDITS] = _CREDIT_FORMS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsers = {name: Parser(morph_pipeline(forms)) for name, forms in vocab.items()}
        parsers.update({f"group:{g}": Parser(morph_pipeline(f)) for g, f in GROUPS.items()})
        parsers["semester"] = Parser(rule(normalized("семестр"), token_type("INT")))
        parsers["total"] = Parser(or_(rule(normalized("итого")), rule(eq("всего"))))
    return parsers


def _text(cell: str) -> str:
    return " ".join(clean_text(cell).split())


def _leaf(cell: str) -> str | None:
    """Вид занятий по тексту заголовка; один столбец — один вид."""
    text = _text(cell)
    if not text:
        return None
    parsers = _parsers()
    found = [
        (m.span.start, name)
        for name, parser in parsers.items()
        if not name.startswith(("group:", "semester", "total"))
        and (m := next(iter(parser.findall(text)), None)) is not None
    ]
    # «промежуточная аттестация (сам.раб.)» содержит один вид; при споре берём самый ранний
    return min(found)[1] if found else None


def _group(cell: str) -> str | None:
    text = _text(cell)
    for group in GROUPS:
        if next(iter(_parsers()[f"group:{group}"].findall(text)), None) is not None:
            return group
    return None


def _number(cell: str) -> float | None:
    text = _text(cell)
    return float(text.replace(",", ".")) if _NUMBER_RE.match(text) else None


def _find_header(table: MdTable) -> tuple[int, dict[int, tuple[str | None, str]]] | None:
    """Строка с видами занятий и для каждого столбца пара (группа, вид)."""
    for i, row in enumerate(table.rows[:6]):
        leaves = {j: leaf for j, cell in enumerate(row) if (leaf := _leaf(cell))}
        if CREDITS not in leaves.values() or len(leaves) < _MIN_LEAVES + 1:
            continue
        groups: dict[int, str] = {}
        for above in table.rows[:i]:
            groups.update({j: g for j, cell in enumerate(above) if (g := _group(cell))})
        columns: dict[int, tuple[str | None, str]] = {}
        current: str | None = None
        for j in range(len(row)):
            current = groups.get(j, current)
            if j in leaves:
                leaf = leaves[j]
                top_level = leaf in (ACTIVE, CREDITS)
                columns[j] = (None if top_level else current, leaf)
        return i, columns
    return None


def extract_workload_table(table: MdTable, file: str, result: RPDExtraction) -> bool:
    """Разбирает таблицу трудоёмкости; возвращает False, если это другая таблица."""
    header = _find_header(table)
    if header is None:
        return False
    header_idx, columns = header
    parsers = _parsers()
    trajectory: str | None = None
    study_form: str | None = None
    for idx in range(header_idx + 1, len(table.rows)):
        row = table.rows[idx]
        first = _text(row[0]) if row else ""
        if "формы текущего контроля" in first.lower():
            break
        if TRAJECTORY_ROW_RE.search(first):
            trajectory, study_form = first, None
            continue
        if (fm := STUDY_FORM_RE.search(first)) and "обучени" in first.lower():
            study_form = fm.group(1).lower()
            continue
        is_total = next(iter(parsers["total"].findall(first)), None) is not None
        sem = next(iter(parsers["semester"].findall(first)), None)
        if not is_total and sem is None:
            continue
        if any(_RANGE_RE.match(_text(c)) for c in row[1:]):
            continue
        result.workload_table.append(
            _build_row(
                row,
                columns,
                is_total,
                sem,
                trajectory,
                study_form,
                SourceRef(file=file, section=table.section, line=table.row_lines[idx]),
            )
        )
    return True


def _build_row(
    row: list[str],
    columns: dict[int, tuple[str | None, str]],
    is_total: bool,
    semester_match,
    trajectory: str | None,
    study_form: str | None,
    source: SourceRef,
) -> WorkloadRow:
    out = WorkloadRow(
        trajectory=trajectory,
        study_form=study_form,
        semester=None if is_total else int(semester_match.tokens[-1].value),
        is_total=is_total,
        source=source,
    )
    for j, (group, leaf) in columns.items():
        value = _number(row[j]) if j < len(row) else None
        if value is None:
            continue
        if leaf == CREDITS:
            out.credits = value
        elif leaf == ACTIVE:
            out.active_hours = value
        elif group == SELF_STUDY:
            out.self_study_hours[leaf] = value
        else:
            out.contact_hours[leaf] = value
    return out


def check_workload(result: RPDExtraction) -> None:
    """Сверка часов с зачётными единицами и ИТОГО с семестрами.

    Расхождения идут в `warnings` и не правятся: чаще всего ошибка в самом документе
    (в ИТОГО пропущен столбец или опечатка в часах).
    """
    block: list[WorkloadRow] = []  # семестры с прошлого ИТОГО: траекторий может быть несколько
    for row in result.workload_table:
        label = "ИТОГО" if row.is_total else f"семестр {row.semester}"
        if row.credits is not None and row.total_hours != row.credits * HOURS_PER_CREDIT:
            result.warnings.append(
                f"трудоёмкость, {label}: {row.total_hours:g} ч не равно "
                f"{row.credits:g} з.е. × {HOURS_PER_CREDIT}"
            )
        if not row.is_total:
            block.append(row)
            continue
        if block and row.credits is not None:
            credits = sum(r.credits or 0 for r in block)
            if abs(credits - row.credits) > _EPS:
                result.warnings.append(
                    f"трудоёмкость: з.е. по семестрам ({credits:g}) не равны ИТОГО "
                    f"({row.credits:g})"
                )
        block = []
