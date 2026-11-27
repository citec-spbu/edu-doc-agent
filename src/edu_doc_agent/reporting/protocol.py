"""Протокол проверки файлом DOCX: все замечания по документам, ошибки раньше предупреждений.

В чат идёт краткий итог (`orchestration/report.py`), полный протокол — этим файлом. Граф
сохраняет его в файловое хранилище сессии и возвращает в выходе запуска.
"""

import io
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from docx import Document

from edu_doc_agent.validation.findings import Finding

PROGRAM = "Программа"
SEVERITY = {"error": "ошибка", "warning": "предупреждение"}


def order(findings: Iterable[Finding]) -> list[Finding]:
    """Ошибки раньше предупреждений; внутри — по документу, программа первой."""
    return sorted(
        findings,
        key=lambda f: (f.severity != "error", f.course is not None, f.course or "", f.rule_title),
    )


def by_document(findings: Iterable[Finding]) -> dict[str, list[Finding]]:
    """Нарушения по документу: сначала документы с большим числом ошибок."""
    groups: dict[str, list[Finding]] = {}
    for f in order(findings):
        groups.setdefault(f.course or PROGRAM, []).append(f)
    return dict(
        sorted(
            groups.items(),
            key=lambda kv: (-sum(f.severity == "error" for f in kv[1]), -len(kv[1]), kv[0]),
        )
    )


def counts(findings: list[Finding]) -> str:
    errors = sum(f.severity == "error" for f in findings)
    warnings = len(findings) - errors
    parts = [f"ошибок: {errors}"] if errors else []
    return ", ".join(parts + ([f"предупреждений: {warnings}"] if warnings else []))


@dataclass(frozen=True)
class ProtocolHeader:
    program: str  # название программы и направление
    verdict: str
    totals: str  # «Без замечаний: …, нарушений: …, не проверено: …»
    catalog_version: str
    editions: list[str]
    created: datetime


def protocol_docx(header: ProtocolHeader, findings: list[Finding]) -> bytes:
    violations = [f for f in findings if f.status == "violation"]
    unchecked = [f for f in findings if f.status == "not_checked"]
    doc = Document()
    doc.add_heading("Протокол проверки документов образовательной программы", 0)
    doc.add_paragraph(header.program)
    doc.add_paragraph(f"Дата: {header.created:%d.%m.%Y %H:%M}")
    doc.add_paragraph(f"Итог: {header.verdict}. {header.totals}")
    doc.add_paragraph(f"Каталог правил: версия {header.catalog_version}.")
    if header.editions:
        doc.add_paragraph("Редакции нормативов: " + "; ".join(header.editions) + ".")

    groups = by_document(violations)
    if groups:
        doc.add_heading("Сводка по документам", 1)
        table = doc.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        header_cells = table.rows[0].cells
        for cell, title in zip(header_cells, ("Документ", "Ошибки", "Предупреждения"), strict=True):
            cell.text = title
        for name, items in groups.items():
            row = table.add_row().cells
            errors = sum(f.severity == "error" for f in items)
            row[0].text, row[1].text, row[2].text = name, str(errors), str(len(items) - errors)

        doc.add_heading("Нарушения", 1)
        for name, items in groups.items():
            doc.add_heading(f"{name} ({counts(items)})", 2)
            for f in items:
                _finding(doc, f)

    if unchecked:
        doc.add_heading("Не проверено", 1)
        doc.add_paragraph("Правила, которые проверить не удалось: проверка неполная.")
        for f in order(unchecked):
            doc.add_paragraph(f"{f.course or PROGRAM}. {f.rule_title}: {f.message}", "List Bullet")

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def _finding(doc, f: Finding) -> None:
    mark = " — требует проверки человеком" if f.needs_human else ""
    title = doc.add_paragraph()
    title.add_run(f"{f.rule_title} ({SEVERITY[f.severity]}){mark}").bold = True
    for line in f.message.splitlines():
        doc.add_paragraph(line.removeprefix("- "), "List Bullet")
    for e in f.evidence:
        doc.add_paragraph(f"Цитата ({e.document}): «{e.quote}»")
    for opinion in f.opinions:
        doc.add_paragraph(opinion)
    if f.requirement:
        r = f.requirement
        doc.add_paragraph(f"Основание: {r.clause_id} ({r.edition}): «{r.quote}»")
