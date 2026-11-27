"""Итог проверки и перечень изменений для чата (код) и короткий ответ агента-репортера
(ADR-0003).

Итог формирует код: счётчики, вердикт, нарушения по документам и самые важные нарушения;
полный протокол — файлом (`reporting/protocol.py`). Агент-репортер не вызывает инструменты: он
получает только итог и пишет короткий ответ в чат, итог идёт следом. Запрос пользователя
репортеру не передаётся: на вопросы отвечает Q&A-агент, репортер лишь пересказывает, что сделано.
"""

import logging
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from edu_doc_agent.dao.storage import FileStorageError
from edu_doc_agent.orchestration.deps import GraphDeps
from edu_doc_agent.orchestration.edits import edit_text
from edu_doc_agent.orchestration.program import load_program
from edu_doc_agent.orchestration.state import CheckSummary, OutputFile
from edu_doc_agent.orchestration.stores import Change
from edu_doc_agent.orchestration.structured import invoke_structured
from edu_doc_agent.reporting.protocol import (
    PROGRAM,
    SEVERITY,
    ProtocolHeader,
    by_document,
    counts,
    order,
    protocol_docx,
)
from edu_doc_agent.validation.findings import Finding

logger = logging.getLogger(__name__)

MAX_QUOTES = 3  # цитат замечания в чате; все — в полном протоколе
MAX_CHAT_FINDINGS = 10  # нарушений в чате, остальные — в полном протоколе
MAX_CHAT_UNCHECKED = 5
MAX_CHAT_DOCUMENTS = 10  # строк «По документам» в чате
MAX_MESSAGE_LINES = 5  # строк сообщения одного нарушения в чате
_CHANGE_STATUS = {
    "draft": "ожидает подтверждения",
    "applied": "внесено в данные",
    "rejected": "отклонено пользователем",
    "superseded": "заменено доработанным вариантом",
    "failed": "не удалось подготовить",
}

REPORTER_PROMPT = """\
Ты агент-репортер системы проверки документов образовательной программы СПбГУ. По протоколу \
запуска напиши короткий итог в чат: 2–4 предложения по-русски. После проверки — сколько \
нарушений и какие самые важные; после изменения — что изменено или почему не удалось. \
Протокол будет показан под твоим итогом, не повторяй его целиком.

Пересказывай только протокол: не делай выводов о том, чего в нём нет, и не пиши, что что-то \
«соответствует» или «совпадает», если это не следует из перечисленных правил."""


class ReporterAnswer(BaseModel):
    message: str


def _finding_lines(i: int, f: Finding) -> list[str]:
    mark = " — требует проверки человеком" if f.needs_human else ""
    lines = [f"{i}. {f.course or PROGRAM}. {f.rule_title} ({SEVERITY[f.severity]}){mark}"]
    message = f.message.splitlines()
    lines += [f"   {line}" for line in message[:MAX_MESSAGE_LINES]]
    if len(message) > MAX_MESSAGE_LINES:
        lines.append(f"   …и ещё строк: {len(message) - MAX_MESSAGE_LINES} (в полном протоколе)")
    for e in f.evidence[:MAX_QUOTES]:
        lines.append(f"   Цитата ({e.document}): «{e.quote}»")
    if len(f.evidence) > MAX_QUOTES:
        lines.append(f"   …и ещё цитат: {len(f.evidence) - MAX_QUOTES}")
    for opinion in f.opinions:
        lines.append(f"   {opinion}")
    if f.requirement:
        r = f.requirement
        lines.append(f"   Основание: {r.clause_id} ({r.edition}): «{r.quote}»")
    return lines


def verdict(summary: CheckSummary, violations: list[Finding], unchecked: list[Finding]) -> str:
    """Итоговый вердикт проверки (ADR-0003): нарушения важнее неполноты."""
    if violations:
        return "есть нарушения"
    if unchecked or summary["missing"] or summary["unknown_courses"] or not summary["rules"]:
        return "проверка неполная"
    return "нарушений нет"


def _unchecked_lines(unchecked: list[Finding]) -> list[str]:
    """Непроверенные правила по причине: «нет общей характеристики» — одной строкой со
    списком правил, а не строкой на каждое правило."""
    by_reason: dict[tuple[str, str], list[str]] = {}
    for f in unchecked:
        by_reason.setdefault((f.course or PROGRAM, f.message), []).append(f.rule_title)
    lines = []
    for (where, reason), titles in by_reason.items():
        if len(titles) == 1:
            lines.append(f"- {where}. {titles[0]}: {reason}")
        else:
            lines.append(f"- {where}: {reason} — правил: {len(titles)} ({'; '.join(titles)})")
    return lines


def totals(summary: CheckSummary, violations: list[Finding], unchecked: list[Finding]) -> str:
    return (
        f"Без замечаний: {summary['passed']}, нарушений: {len(violations)}, "
        f"не проверено: {len(unchecked)}."
    )


def check_report(
    summary: CheckSummary, findings: list[Finding], protocol_file: str | None = None
) -> str:
    """Итог проверки для чата; полный протокол — файл `protocol_file`."""
    violations = order(f for f in findings if f.status == "violation")
    unchecked = order(f for f in findings if f.status == "not_checked")
    lines = [
        f"Проверка завершена. {totals(summary, violations, unchecked)}",
        f"Итог: {verdict(summary, violations, unchecked)}.",
    ]
    if not summary["rules"]:
        sets = ", ".join(summary["rule_sets"]) or "все"
        lines.append(f"Нет правил проверки для выбранных наборов ({sets}).")
    if "direction_code" in summary["missing"]:
        lines.append("Правила ФГОС не применялись: не указан код направления.")
    if "fgos_declined" in summary["missing"]:
        lines.append("Правила ФГОС не применялись: проверка по ФГОС отменена пользователем.")
    if "fgos_rules" in summary["missing"]:
        code = summary.get("direction_code")
        where = f"направления {code}" if code else "этого направления"
        lines.append(f"Правила ФГОС не применялись: для {where} их нет в каталоге.")
    if summary["unknown_courses"]:
        lines.append("Не найдены дисциплины: " + ", ".join(summary["unknown_courses"]) + ".")
    elif summary["rules"] and not findings and not summary["passed"]:
        lines.append("Нет загруженных документов для проверки.")
    lines += _document_lines(violations) + _violation_lines(violations)
    if unchecked:
        grouped = _unchecked_lines(unchecked)
        lines += ["", "Не проверено (проверка неполная):", *grouped[:MAX_CHAT_UNCHECKED]]
        if len(grouped) > MAX_CHAT_UNCHECKED:
            lines.append(f"…и ещё строк: {len(grouped) - MAX_CHAT_UNCHECKED}.")
    lines += ["", f"Каталог правил: версия {summary['catalog_version']}."]
    if summary["editions"]:
        lines.append("Редакции нормативов: " + "; ".join(summary["editions"]) + ".")
    if protocol_file:
        lines.append(f"Полный протокол со всеми нарушениями и цитатами: «{protocol_file}».")
    return "\n".join(lines)


def _document_lines(violations: list[Finding]) -> list[str]:
    groups = by_document(violations)
    if len(groups) < 2:
        return []
    lines = ["", "По документам:"]
    lines += [
        f"- {name}: {counts(items)}" for name, items in list(groups.items())[:MAX_CHAT_DOCUMENTS]
    ]
    if len(groups) > MAX_CHAT_DOCUMENTS:
        lines.append(f"…и ещё документов: {len(groups) - MAX_CHAT_DOCUMENTS}.")
    return lines


def _violation_lines(violations: list[Finding]) -> list[str]:
    if not violations:
        return []
    title = "Нарушения" if len(violations) <= MAX_CHAT_FINDINGS else "Самые важные нарушения"
    lines = ["", f"{title}:"]
    for i, f in enumerate(violations[:MAX_CHAT_FINDINGS], start=1):
        lines += _finding_lines(i, f)
    if len(violations) > MAX_CHAT_FINDINGS:
        lines.append(f"…и ещё нарушений: {len(violations) - MAX_CHAT_FINDINGS}.")
    return lines


async def save_protocol(
    deps: GraphDeps, session_id: str, run_id: str, summary: CheckSummary, findings: list[Finding]
) -> OutputFile | None:
    """Полный протокол проверки файлом DOCX в хранилище сессии; None — без хранилища или если
    файл не записался (итог в чате остаётся)."""
    if deps.storage is None:
        return None
    program, _ = await load_program(deps, session_id)
    plan, characteristic = program.plan, program.characteristic
    title = (plan and plan.program) or (characteristic and characteristic.program_name) or ""
    direction = (plan and plan.direction) or (characteristic and characteristic.direction) or ""
    violations = [f for f in findings if f.status == "violation"]
    unchecked = [f for f in findings if f.status == "not_checked"]
    created = datetime.now()
    header = ProtocolHeader(
        program=", ".join(x for x in (title, direction) if x) or "Программа",
        verdict=verdict(summary, violations, unchecked),
        totals=totals(summary, violations, unchecked),
        catalog_version=summary["catalog_version"],
        editions=summary["editions"],
        created=created,
    )
    filename = f"Протокол проверки {created:%Y-%m-%d %H-%M}.docx"
    try:
        key = await deps.storage.save(
            session_id=session_id,
            doc_id=f"protocol-{run_id}",
            filename=filename,
            content=protocol_docx(header, findings),
        )
    except FileStorageError:
        logger.exception("протокол проверки не сохранён")
        return None
    return {
        "storage_key": key,
        "filename": filename,
        "kind": "check_protocol",
        "derived_from": None,
    }


def change_report(change: Change, titles: dict[str, str]) -> str:
    lines = [f"Изменение: {_CHANGE_STATUS[change.status]}."]
    for i, e in enumerate(change.edits, start=1):
        lines.append(f"{i}. {edit_text(e, titles)}")
    lines += [f"- {c}" for c in change.consequences]
    lines += [f"- Документ: {note}" for note in change.document_notes]
    if change.problems:
        lines.append("Проблемы: " + "; ".join(change.problems))
    return "\n".join(lines)


async def reporter_message(model: BaseChatModel, protocol: str) -> str | None:
    """Короткий итог агента-репортера по протоколу; None, если LLM недоступна."""
    try:
        answer = await invoke_structured(
            model,
            ReporterAnswer,
            [
                SystemMessage(REPORTER_PROMPT),
                HumanMessage(protocol[:20_000]),
            ],
        )
    except Exception:
        return None
    return answer.message
