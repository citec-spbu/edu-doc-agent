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


def _cell_intervals(table: MdTable, row: int) -> dict[int, tuple[int, int]]:
    """Физические границы ячеек строки, включая пустые ячейки между значениями."""
    edges = [0]
    for width in table.grid_widths:
        edges.append(edges[-1] + width)
    intervals = {}
    col = 0
    while col < len(table.rows[row]):
        span = next(
            (s for s in table.spans or [] if s.row <= row < s.row + s.row_span and s.col == col),
            None,
        )
        end = col + (span.col_span if span else 1)
        intervals[col] = (edges[col], edges[end])
        col = end
    return intervals


def _aligned_columns(
    table: MdTable,
    header: int,
    row: int,
    columns: dict[int, tuple[str | None, str]],
    warnings: list[str],
) -> dict[int, tuple[str | None, str]]:
    """Монотонное сопоставление ячеек по перекрытию, без повторного использования.

    Сначала максимизируем число пар, затем суммарное перекрытие. Пустые ячейки
    участвуют: локальный максимум для одного числа может занять чужой столбец.
    Равнозначные варианты не разрешаем по значениям часов или ожидаемой сумме.
    """
    if not table.grid_widths or table.spans is None:
        return columns
    if len(table.grid_widths) != len(table.rows[row]) or any(w <= 0 for w in table.grid_widths):
        warnings.append(f"трудоёмкость, строка {table.row_lines[row]}: некорректная сетка tblGrid")
        return {}
    head_cells = _cell_intervals(table, header)
    data_cells = _cell_intervals(table, row)
    heads = [(col, head_cells[col]) for col in columns]
    cells = [(col, bounds) for col, bounds in data_cells.items() if col != 0]

    @cache
    def match(i: int, j: int) -> tuple[tuple[int, int], tuple[tuple[int, int], ...], bool]:
        if i == len(heads) or j == len(cells):
            return (0, 0), (), False
        candidates = [match(i + 1, j), match(i, j + 1)]
        hcol, (hstart, hend) = heads[i]
        dcol, (dstart, dend) = cells[j]
        overlap = min(hend, dend) - max(hstart, dstart)
        if overlap > 0:
            (count, width), pairs, ambiguous = match(i + 1, j + 1)
            candidates.append(((count + 1, width + overlap), ((hcol, dcol), *pairs), ambiguous))
        score = max(c[0] for c in candidates)
        best = [c for c in candidates if c[0] == score]
        pairs = best[0][1]
        return score, pairs, any(c[2] or c[1] != pairs for c in best)

    _, pairs, ambiguous = match(0, 0)
    where = f"трудоёмкость, строка {table.row_lines[row]}"
    if ambiguous:
        warnings.append(f"{where}: неоднозначное перекрытие ячеек с шапкой; часы не присвоены")
        return {}
    matched = {dcol for _, dcol in pairs}
    for col, _ in cells:
        if col not in matched and _number(table.rows[row][col]) is not None:
            warnings.append(f"{where}: число в столбце {col + 1} не сопоставлено с шапкой")
    return {dcol: columns[hcol] for hcol, dcol in pairs}


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
        aligned = _aligned_columns(table, header_idx, idx, columns, result.warnings)
        extracted = _build_row(
            row,
            aligned,
            is_total,
            sem,
            trajectory,
            study_form,
            SourceRef(file=file, section=table.section, line=table.row_lines[idx]),
        )
        result.workload_table.append(extracted)
        if extracted.credits is None:
            result.warnings.append(
                f"трудоёмкость, строка {table.row_lines[idx]}: не извлечены зачётные единицы; "
                "проверьте сетку таблицы, для Markdown без геометрии используйте исходный DOCX"
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


def _where(row: WorkloadRow) -> str:
    """Траектория, форма и строка md: иначе предупреждения разных траекторий неразличимы."""
    parts = [p for p in (row.trajectory, row.study_form) if p]
    if row.source.line is not None:
        parts.append(f"строка {row.source.line}")
    return f" [{', '.join(parts)}]" if parts else ""


def _kind_sums(rows: list[WorkloadRow], field: str) -> dict[str, float]:
    sums: dict[str, float] = {}
    for r in rows:
        for kind, hours in getattr(r, field).items():
            sums[kind] = sums.get(kind, 0) + hours
    return sums


def check_workload(result: RPDExtraction) -> None:
    """Сверка часов с з.е. и ИТОГО с семестрами, в том числе по видам контактной работы.

    Расхождения идут в `warnings` и не правятся: чаще всего ошибка в самом документе
    (в ИТОГО пропущен столбец или опечатка в часах).
    """
    block: list[WorkloadRow] = []  # семестры с прошлого ИТОГО: траекторий может быть несколько
    for row in result.workload_table:
        label = "ИТОГО" if row.is_total else f"семестр {row.semester}"
        if row.credits is not None and row.total_hours != row.credits * HOURS_PER_CREDIT:
            result.warnings.append(
                f"трудоёмкость, {label}{_where(row)}: {row.total_hours:g} ч не равно "
                f"{row.credits:g} з.е. × {HOURS_PER_CREDIT}"
            )
        if not row.is_total:
            block.append(row)
            continue
        if block and row.credits is not None:
            credits = sum(r.credits or 0 for r in block)
            if abs(credits - row.credits) > _EPS:
                result.warnings.append(
                    f"трудоёмкость{_where(row)}: з.е. по семестрам ({credits:g}) не равны ИТОГО "
                    f"({row.credits:g})"
                )
        if block:
            for field, label in (
                ("contact_hours", "контактная работа"),
                ("self_study_hours", "самостоятельная работа"),
            ):
                sums = _kind_sums(block, field)
                totals = _kind_sums([row], field)
                for kind in sums.keys() | totals.keys():
                    total = totals.get(kind, 0)
                    if abs(sums.get(kind, 0) - total) > _EPS:
                        result.warnings.append(
                            f"трудоёмкость{_where(row)}: {label}, {kind} по семестрам "
                            f"({sums.get(kind, 0):g} ч) не равны ИТОГО ({total:g} ч)"
                        )
        block = []
