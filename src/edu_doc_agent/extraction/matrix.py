"""Извлечение матрицы компетенций из xlsx: какие компетенции формирует каждая дисциплина.

Лист устроен так: две строки шапки (тип компетенции над кодом), затем строки дисциплин
«№, семестр, профиль, часть, з.е., дисциплина» с отметками 0/1 по столбцам компетенций и итоговые
строки «Итого (СУММ)», «Итого (Стандарт)», «Итого (ТОП)». Столбцы находятся по заголовкам, а не по
позициям. Коды компетенций хранятся как в таблице (в нумерованном диалекте `ПКП-2-ППК-Р2`);
сопоставление с другими документами — `codes.canonical_code`. Расхождения идут в `warnings`.
"""

from pathlib import Path
from typing import BinaryIO

from openpyxl import load_workbook
from pydantic import Field

from edu_doc_agent.extraction.models import _Model
from edu_doc_agent.extraction.patterns import COMPETENCE_RE, code_key
from edu_doc_agent.extraction.source import SourceRef

_META_HEADERS = {
    "№ строки": "number",
    "семестр": "semester",
    "профиль": "profile",
    "часть": "part",
    "зет": "credits",
    "дисциплина": "discipline",
}
_SUM_LABELS = {"сумм", "сумма", "всего"}
_SHARED_PROFILE = "база"  # общие для всех профилей строки


class MatrixColumn(_Model):
    code: str
    type: str | None = Field(default=None, description="Тип компетенции над кодом: УК, ОПК, ПКП, …")
    column: int = Field(description="Номер столбца листа, с 1")
    source: SourceRef | None = None


class MatrixRow(_Model):
    number: int
    semester: int | None = None
    profile: str | None = Field(default=None, description="«база», «стандарт», «топ»")
    part: str | None = Field(default=None, description="«базовая» / «вариативная»")
    credits: float | None = None
    discipline: str
    competences: list[str] = Field(
        default_factory=list, description="Коды компетенций с отметкой 1, в порядке столбцов"
    )
    source: SourceRef


class CompetenceMatrix(_Model):
    file: str
    sheet: str | None = None
    columns: list[MatrixColumn] = Field(default_factory=list)
    rows: list[MatrixRow] = Field(default_factory=list)
    totals: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="Итоговые строки: подпись → {код: сумма}"
    )
    total_sources: dict[str, SourceRef] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _text(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _number(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "") else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _find_header(rows: list[tuple]) -> int | None:
    for i, row in enumerate(rows):
        cells = [_text(c).lower() for c in row]
        has_codes = sum(1 for c in row if isinstance(c, str) and COMPETENCE_RE.fullmatch(c.strip()))
        if "дисциплина" in cells and has_codes:
            return i
    return None


def extract_matrix(source: Path | BinaryIO, file: str | None = None) -> CompetenceMatrix:
    """Матрица из xlsx (путь или файловый объект); читается первый лист."""
    name = file or (source.name if isinstance(source, Path) else "")
    wb = load_workbook(source, data_only=True, read_only=True)
    ws = wb.worksheets[0]
    matrix = CompetenceMatrix(file=name, sheet=ws.title)
    try:
        rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
    finally:
        wb.close()
    header_idx = _find_header(rows)
    if header_idx is None:
        matrix.warnings.append("не найдена шапка матрицы (столбец «Дисциплина» и коды компетенций)")
        return matrix
    header = rows[header_idx]
    types = rows[header_idx - 1] if header_idx > 0 else ()
    meta: dict[str, int] = {}
    for col, cell in enumerate(header):
        if (key := _META_HEADERS.get(_text(cell).lower())) is not None:
            meta[key] = col
        elif isinstance(cell, str) and COMPETENCE_RE.fullmatch(cell.strip()):
            kind = _text(types[col]) if col < len(types) else ""
            matrix.columns.append(
                MatrixColumn(
                    code=cell.strip(),
                    type=kind or None,
                    column=col + 1,
                    source=SourceRef(file=name, section=ws.title, line=header_idx + 1, quote=cell),
                )
            )
    if "discipline" not in meta:
        matrix.warnings.append("не найден столбец «Дисциплина»")
        return matrix

    for offset, row in enumerate(rows[header_idx + 1 :], start=header_idx + 2):
        first = _text(row[0]) if row else ""
        if first.lower().startswith("итого"):
            label = first.split("(", 1)[1].rstrip(") ") if "(" in first else first
            matrix.totals[label] = {
                c.code: int(_number(row[c.column - 1]) or 0) for c in matrix.columns
            }
            matrix.total_sources[label] = SourceRef(
                file=name, section=ws.title, line=offset, quote=" | ".join(map(_text, row))
            )
            continue
        discipline = _text(row[meta["discipline"]]) if meta["discipline"] < len(row) else ""
        if not discipline:
            continue
        marks = []
        for col in matrix.columns:
            value = row[col.column - 1] if col.column - 1 < len(row) else None
            if value in (1, 1.0):
                marks.append(col.code)
            elif value not in (0, 0.0, None, ""):
                matrix.warnings.append(
                    f"строка {offset}, {col.code}: значение {value!r} вместо 0/1"
                )
        number = _number(row[meta["number"]]) if "number" in meta else None
        semester = _number(row[meta["semester"]]) if "semester" in meta else None
        matrix.rows.append(
            MatrixRow(
                number=int(number) if number is not None else len(matrix.rows) + 1,
                semester=int(semester) if semester is not None else None,
                profile=_text(row[meta["profile"]]).lower() or None if "profile" in meta else None,
                part=_text(row[meta["part"]]).lower() or None if "part" in meta else None,
                credits=_number(row[meta["credits"]]) if "credits" in meta else None,
                discipline=discipline,
                competences=marks,
                source=SourceRef(
                    file=name, section=ws.title, line=offset, quote=" | ".join(map(_text, row))
                ),
            )
        )
    check_matrix(matrix)
    return matrix


def check_matrix(matrix: CompetenceMatrix) -> None:
    """Дубли столбцов, типы над кодами и согласованность итоговых строк с данными."""
    seen: set[str] = set()
    for col in matrix.columns:
        if (key := code_key(col.code)) in seen:
            matrix.warnings.append(f"столбец {col.code}: повтор кода")
        seen.add(key)
        if col.type and not col.code.upper().startswith(col.type.upper()):
            matrix.warnings.append(
                f"столбец {col.code}: тип над кодом «{col.type}» не соответствует коду"
            )
    for label, totals in matrix.totals.items():
        low = label.strip().lower()
        if low in _SUM_LABELS:
            rows = matrix.rows
        else:
            rows = [r for r in matrix.rows if r.profile in (low, _SHARED_PROFILE)]
        for col in matrix.columns:
            expected = sum(col.code in r.competences for r in rows)
            if totals.get(col.code) != expected:
                matrix.warnings.append(
                    f"итого ({label}), {col.code}: в таблице {totals.get(col.code)}, "
                    f"по строкам {expected}"
                )
