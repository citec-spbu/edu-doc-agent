"""Агент-чекер и агент-ревьюер для правил `method: llm` (ADR-0003, «Проверка», шаги 3–5).

1. Чекер получает правило, цитату норматива, поля документов из `fields` и контекст программы
   и решает: выполнено или нарушено, с дословными цитатами.
2. Код проверяет цитаты нарушения и возвращает ошибки чекеру, не более 4 раз; затем
   `not_checked`. Без полей из `fields` чекер не вызывается.
3. Ревьюер проверяет вывод по тем же данным; несогласие уходит чекеру с объяснением. Без
   согласия за 3 раунда замечание помечается «требует проверки человеком».
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from edu_doc_agent.extraction.codes import canonical_code
from edu_doc_agent.extraction.llm_extract import quote_in_text
from edu_doc_agent.extraction.md_utils import split_sections
from edu_doc_agent.extraction.models import RPDExtraction
from edu_doc_agent.orchestration.structured import StructuredOutputError, invoke_structured
from edu_doc_agent.orchestration.tools import plain_data
from edu_doc_agent.validation.catalog import Rule
from edu_doc_agent.validation.checks import CheckOutcome, not_checked, passed, violation
from edu_doc_agent.validation.findings import Evidence
from edu_doc_agent.validation.units import (
    CourseDocuments,
    ProgramDocuments,
    Unit,
    plan_entries_for,
)

logger = logging.getLogger(__name__)

MAX_FIXES = 4
MAX_PROGRAM_CONTEXT = 8000  # символов контекста программы в запросе
MAX_REVIEW_ROUNDS = 3

CHECKER_PROMPT = """\
Ты агент-чекер системы проверки документов образовательной программы СПбГУ. Проверь одно \
правило по данным документов и реши, выполнено оно или нарушено.

- Решай только по переданным данным и цитате норматива, не додумывай.
- При нарушении приведи дословные цитаты из документов (поле quotes): документ — имя файла \
из списка, quote — фрагмент текста документа без изменений. Если значения в документах \
расходятся, нужна цитата из каждого документа. Если чего-то не хватает, приведи одну цитату \
и назови документ, где этого нет.
- Цитата — один непрерывный фрагмент текста документа. Не добавляй к ней пометки («знает:», \
«умеет:»), кавычки и коды компетенций, не склеивай фрагменты из разных мест.
- explanation — коротко, по-русски, что именно нарушено или почему правило выполнено."""

REVIEWER_PROMPT = """\
Ты агент-ревьюер. Агент-чекер проверил правило по документам образовательной программы и \
считает, что оно нарушено. Проверь его вывод по тем же данным документов и контексту \
программы, решай по критерию правила.

- Вывод верен — confirm = true.
- Вывод неверен — confirm = false; в reason объясни почему, а если это видно из документа, \
приведи дословные цитаты в quotes.
- reason — коротко, по-русски."""


class Quote(BaseModel):
    document: str = Field(description="имя файла документа из списка")
    quote: str = Field(description="дословный фрагмент текста документа")


class CheckerAnswer(BaseModel):
    satisfied: bool = Field(description="true — правило выполнено, false — нарушено")
    explanation: str
    quotes: list[Quote] = []


class ReviewerVerdict(BaseModel):
    confirm: bool = Field(description="true — вывод чекера верен, нарушение есть")
    reason: str
    quotes: list[Quote] = Field(default=[], description="при confirm = false: цитаты из документов")


class CheckerReply(BaseModel):
    keep: bool = Field(description="true — нарушение остаётся, false — снято")
    explanation: str


@dataclass
class CheckerVerdict:
    outcome: CheckOutcome
    evidence: list[Evidence]
    needs_human: bool = False
    opinions: list[str] | None = None


def program_context(unit: Unit, program: ProgramDocuments | None) -> str:
    """Место дисциплины в программе: строки учебного плана, закреплённые за ней компетенции с
    формулировками и результатами из ЦКМВ. Без него документ судится в отрыве от программы:
    результаты из ЦКМВ, общие для нескольких дисциплин, выглядят скопированными."""
    data = unit.data.course if isinstance(unit.data, CourseDocuments) else unit.data
    if program is None or program.plan is None or not isinstance(data, RPDExtraction):
        return ""
    plan = program.plan
    entries = plan_entries_for(data, plan)
    lines = [f"Программа: {plan.program or ''}, направление {plan.direction or ''}"]
    if not entries:
        return "\n".join([*lines, "Дисциплины нет в учебном плане (по коду и названию)."])
    for e in entries:
        lines.append(
            f"Учебный план: {e.name}, семестр {e.semester}, {e.credits:g} з.е., "
            f"профиль {e.profile or 'общий'}"
        )
    names = {canonical_code(c.code): c.name for c in plan.competences}
    ckmv = {canonical_code(c.code): c for c in program.ckmv.competences} if program.ckmv else {}
    lines.append("Компетенции, закреплённые за дисциплиной в учебном плане:")
    for code in dict.fromkeys(c for e in entries for c in e.competences):
        key = canonical_code(code)
        lines.append(f"- {code}: {names.get(key, '')}")
        for indicator in ckmv[key].indicators if key in ckmv else []:
            outcomes = "; ".join(o.text for o in indicator.outcomes)
            lines.append(f"  - индикатор ЦКМВ {indicator.code} {indicator.name}: {outcomes}")
    return "\n".join(lines)[:MAX_PROGRAM_CONTEXT]


def _rule_text(rule: Rule) -> str:
    lines = [f"Правило: {rule.title}"]
    if rule.criterion:
        lines.append(f"Критерий: {rule.criterion}")
    if rule.requirement:
        r = rule.requirement
        lines.append(f"Норматив ({r.clause_id}, ред. {r.edition}): «{r.quote}»")
    return "\n".join(lines)


def _fields(data: object, fields: list[str]) -> dict:
    return {f: plain_data(getattr(data, f)) for f in fields if hasattr(data, f)}


def unit_fields(rule: Rule, unit: Unit) -> dict[str, dict]:
    """Поля из `rule.fields` по документам единицы проверки: имя файла → поля."""
    data = unit.data
    if isinstance(data, CourseDocuments):
        out = {data.course.file: _fields(data.course, rule.fields)}
        if data.plan is not None:
            out[data.plan.file] = {"строки учебного плана": plain_data(list(data.plan_entries))}
        return out
    if isinstance(data, ProgramDocuments):
        docs = [data.plan, data.characteristic, data.ckmv, data.matrix]
        out = {d.file: _fields(d, rule.fields) for d in docs if d is not None}
        for c in data.courses:
            out[c.filename] = _fields(c.data, rule.fields)
        return {k: v for k, v in out.items() if v}
    return {data.file: _fields(data, rule.fields)}


# вид результата обучения хранится отдельным полем `kind`, модель приписывает его к цитате
_KIND_PREFIX_RE = re.compile(r"^[\s«\"]*(знает|умеет|владеет)[\s»\":—-]*", re.IGNORECASE)


def clean_quotes(answer: CheckerAnswer) -> CheckerAnswer:
    """Убирает из начала цитат пометки «знает», «умеет», «владеет»."""
    quotes = [
        q.model_copy(update={"quote": _KIND_PREFIX_RE.sub("", q.quote).strip(' «»"')})
        for q in answer.quotes
    ]
    return answer.model_copy(update={"quotes": quotes})


def verify_answer(answer: CheckerAnswer, unit: Unit, texts: dict[str, str]) -> list[str]:
    """Замечания кода к ответу чекера; пустой список — ответ принят."""
    if answer.satisfied:
        return []
    problems = []
    if not answer.quotes:
        problems.append("у нарушения нет ни одной цитаты из документов")
    for q in answer.quotes:
        if q.document not in unit.files:
            problems.append(f"документа «{q.document}» нет, есть: {', '.join(unit.files)}")
        elif not quote_in_text(q.quote, texts.get(q.document, "")):
            problem = f"цитаты «{q.quote[:120]}» нет в документе «{q.document}» дословно"
            if similar := closest_fragment(q.quote, texts.get(q.document, "")):
                problem += f"; похожий фрагмент документа: «{similar}»"
            problems.append(problem)
    return problems


def closest_fragment(quote: str, text: str) -> str | None:
    """Самый похожий на цитату фрагмент текста той же длины в словах: модель часто ошибается
    в одном слове и повторяет ошибку, если не показать ей верный текст."""
    words = re.sub(r"<br>|\|", " ", text).split()
    n = len(quote.split())
    windows = {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
    found = difflib.get_close_matches(quote, windows, n=1, cutoff=0.8)
    return found[0] if found else None


_FIELD_TITLES = {
    "goals": "цели дисциплины",
    "learning_outcomes": "результаты обучения",
    "practice": "сведения о практике",
}


def missing_fields(rule: Rule, unit: Unit) -> list[str]:
    """Поля из `rule.fields`, которых нет в документе дисциплины (не извлечены)."""
    data = unit.data.course if isinstance(unit.data, CourseDocuments) else unit.data
    if isinstance(data, ProgramDocuments):
        return []
    return [
        _FIELD_TITLES.get(f, f) for f in rule.fields if getattr(data, f, None) in (None, [], "")
    ]


def _sections_with(quotes: list[Quote], texts: dict[str, str]) -> str:
    """Разделы документов, из которых взяты цитаты, целиком."""
    parts = []
    for q in quotes:
        for s in split_sections(texts.get(q.document, "")):
            body = s.full_text or s.text
            if quote_in_text(q.quote, f"{s.heading}\n{body}"):
                part = f"[{q.document}] {s.heading}\n{body[:6000]}"
                if part not in parts:
                    parts.append(part)
                break
    return "\n\n".join(parts) or "(разделы не найдены)"


def _texts_for(unit: Unit, texts: dict[str, str], fields: dict[str, dict]) -> dict[str, str]:
    """Текст для сверки цитат: Markdown документа и значения переданных полей."""
    return {
        f: "\n".join([texts.get(f, ""), json.dumps(fields.get(f, {}), ensure_ascii=False)])
        for f in unit.files
    }


@dataclass
class _Case:
    """Пара «правило, единица проверки» с данными, которые видят оба агента."""

    rule: Rule
    unit: Unit
    texts: dict[str, str]  # Markdown документов
    quote_texts: dict[str, str]  # текст для сверки цитат: Markdown и переданные поля
    context: str  # правило, контекст программы и данные документов

    @property
    def where(self) -> str:
        return f"{self.rule.id} / {self.unit.course}"


async def check_rule(
    checker: BaseChatModel,
    reviewer: BaseChatModel,
    rule: Rule,
    unit: Unit,
    texts: dict[str, str],
    program: ProgramDocuments | None = None,
) -> CheckerVerdict:
    if isinstance(unit.data, CourseDocuments) and unit.data.plan is None:
        return CheckerVerdict(not_checked("нет учебного плана"), [])  # сверять не с чем
    if missing := missing_fields(rule, unit):
        return CheckerVerdict(not_checked("в документе не найдены: " + ", ".join(missing)), [])
    fields = unit_fields(rule, unit)
    quote_texts = _texts_for(unit, texts, fields)
    in_program = program_context(unit, program)
    context = (
        f"{_rule_text(rule)}\n\nДокументы: {', '.join(unit.files)}\n\n"
        + (f"Контекст программы:\n{in_program}\n\n" if in_program else "")
        + f"Данные документов (JSON):\n{json.dumps(fields, ensure_ascii=False, indent=1)}"
    )
    case = _Case(rule, unit, texts, quote_texts, context)
    history: list[BaseMessage] = [SystemMessage(CHECKER_PROMPT), HumanMessage(context)]
    answer = None
    for _ in range(MAX_FIXES + 1):
        try:
            answer = clean_quotes(await invoke_structured(checker, CheckerAnswer, history))
        except StructuredOutputError as error:
            return CheckerVerdict(not_checked(f"ответ агента-чекера не по схеме: {error}"), [])
        problems = verify_answer(answer, unit, quote_texts)
        if not problems:
            break
        history += [
            AIMessage(answer.model_dump_json()),
            HumanMessage(
                "Ответ не прошёл проверку кодом:\n- " + "\n- ".join(problems) + "\nИсправь ответ."
            ),
        ]
    else:
        return CheckerVerdict(
            not_checked("ответ агента-чекера не прошёл проверку цитат за 4 исправления"), []
        )
    assert answer is not None
    if answer.satisfied:
        return CheckerVerdict(passed(), [])
    return await _review(checker, reviewer, case, answer, history)


async def _review(
    checker: BaseChatModel,
    reviewer: BaseChatModel,
    case: _Case,
    answer: CheckerAnswer,
    checker_history: list[BaseMessage],
) -> CheckerVerdict:
    """Спор ревьюера с чекером: до `MAX_REVIEW_ROUNDS` раундов, затем решает человек."""
    evidence = [Evidence(document=q.document, quote=q.quote) for q in answer.quotes]
    sections = _sections_with(answer.quotes, case.texts)
    quotes = "\n".join(f"- [{q.document}] «{q.quote}»" for q in answer.quotes)
    review: list[BaseMessage] = [
        SystemMessage(REVIEWER_PROMPT),
        HumanMessage(
            f"{case.context}\n\nВывод чекера: {answer.explanation}\nЦитаты чекера:\n{quotes}"
            f"\n\nРазделы документов с цитатами:\n{sections}"
        ),
    ]
    explanation, reason = answer.explanation, ""
    history = [*checker_history, AIMessage(answer.model_dump_json())]
    where = case.where
    for _ in range(MAX_REVIEW_ROUNDS):
        try:
            verdict = await invoke_structured(reviewer, ReviewerVerdict, review)
        except StructuredOutputError:
            break  # ревьюер не ответил: нарушение остаётся по решению чекера
        if verdict.confirm:
            logger.info("агент-ревьюер %s: подтвердил", where)
            return CheckerVerdict(violation(explanation, []), evidence)
        reason = verdict.reason
        logger.info("агент-ревьюер %s: не согласен: %s", where, reason)
        history.append(
            HumanMessage(
                f"Агент-ревьюер не согласен: {reason}{_reviewer_quotes(verdict, case)}"
                "\nСнимаешь нарушение или оставляешь? Если оставляешь, объясни почему."
            )
        )
        try:
            reply = await invoke_structured(checker, CheckerReply, history)
        except StructuredOutputError:
            break
        if not reply.keep:
            logger.info("агент-чекер %s: снял нарушение: %s", where, reply.explanation)
            return CheckerVerdict(passed(), [])
        explanation = reply.explanation
        logger.info("агент-чекер %s: оставил нарушение: %s", where, explanation)
        history.append(AIMessage(reply.model_dump_json()))
        review += [
            AIMessage(verdict.model_dump_json()),
            HumanMessage(f"Чекер оставляет нарушение: {explanation}\nВерен ли его вывод?"),
        ]
    else:
        logger.info("%s: согласия нет, требует проверки человеком", where)
        return CheckerVerdict(
            violation(explanation, []),
            evidence,
            needs_human=True,
            opinions=[f"Агент-чекер: {explanation}", f"Агент-ревьюер: {reason}"],
        )
    return CheckerVerdict(violation(explanation, []), evidence)


def _reviewer_quotes(verdict: ReviewerVerdict, case: _Case) -> str:
    """Цитаты ревьюера для чекера; цитату, которой нет в документе, код помечает."""
    lines = []
    for q in clean_quotes(
        CheckerAnswer(satisfied=False, explanation="", quotes=verdict.quotes)
    ).quotes:
        text = case.quote_texts.get(q.document, "")
        found = q.document in case.unit.files and quote_in_text(q.quote, text)
        lines.append(f"- [{q.document}] «{q.quote}»" + ("" if found else " (нет в документе)"))
    return "\nЦитаты ревьюера:\n" + "\n".join(lines) if lines else ""
