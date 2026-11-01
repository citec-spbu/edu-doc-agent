"""LLM-извлечение смысловых полей РПД (ADR-0002, часть 2) через шлюз `llm/`.

Для каждой задачи берётся текст нужного раздела, модель отвечает JSON по схеме, ответ
проверяется: невалидный JSON повторно запрашивается (до `MAX_RETRIES` раз), элементы с цитатой,
которой нет в тексте раздела, отбрасываются с предупреждением. Все вызовы идут через `LLMClient`,
конкретный провайдер сюда не протекает.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from edu_doc_agent.extraction import llm_prompts as prompts
from edu_doc_agent.extraction.llm_schemas import (
    ActiveFormItem,
    ActiveFormsOutput,
    GradingRulesOutput,
    LLMActiveForm,
    LLMExtraction,
    LLMGradingRule,
    LLMPrerequisite,
    PrerequisitesOutput,
)
from edu_doc_agent.extraction.md_utils import MdSection, clean_text, split_sections
from edu_doc_agent.extraction.source import SourceRef
from edu_doc_agent.llm import LLMClient

MAX_RETRIES = 2  # повторов после первого ответа, как в ADR-0002


@dataclass(frozen=True)
class LLMTask:
    name: str
    prompt: str
    output: type[BaseModel]
    select: Callable[[list[MdSection]], list[MdSection]]  # разделы документа для одного вызова
    with_tables: bool = False  # подавать таблицы разделов модели как текст


_SECTION_END_RE = re.compile(r"^(раздел|приложение)\b")
_GRADING_SUB_RE = re.compile(r"методик|критери")


def select_prerequisites(sections: list[MdSection]) -> list[MdSection]:
    """РПД: «1.2. Требования к подготовленности…»; у практик тот же раздел лежит под номером 1.5,
    поэтому ищем по заголовку, а номер 1.2 — запасной вариант."""
    hits = [s for s in sections if "подготовленност" in s.heading.lower()]
    return hits or [s for s in sections if s.number == "1.2"]


def select_active_forms(sections: list[MdSection]) -> list[MdSection]:
    """«1.4. Перечень активных и интерактивных форм…». У практик такого раздела нет."""
    return [
        s
        for s in sections
        if "активн" in s.heading.lower() and (s.number is None or s.number.startswith("1."))
    ]


def select_grading(sections: list[MdSection]) -> list[MdSection]:
    """«3.1.3. Методика … и критерии оценивания» вместе с вложенными разделами.

    В РПД критерии часто идут безномерными заголовками того же уровня сразу после 3.1.3
    («Критерии выставления оценок за экзамен», «Таблица. Критерии оценивания…»), поэтому
    раздел продолжается до следующего пронумерованного заголовка. У практик 3.1.3 только
    объединяет подразделы: берём «Методика…» и «Критерии…», но не форму отчёта и ФОС.
    """
    picked: list[MdSection] = []
    inside = False
    for s in sections:
        if s.number is None:
            if _SECTION_END_RE.match(s.heading.lower()):
                inside = False
        elif s.number == "3.1.3":
            inside = True
        elif s.number.startswith("3.1.3."):
            inside = bool(_GRADING_SUB_RE.search(s.heading.lower()))
        else:
            inside = False
        if inside:
            picked.append(s)
    return picked


TASKS = [
    LLMTask("prerequisites", prompts.PREREQUISITES, PrerequisitesOutput, select_prerequisites),
    LLMTask("active_forms", prompts.ACTIVE_FORMS, ActiveFormsOutput, select_active_forms),
    LLMTask(
        "grading_rules",
        prompts.GRADING_RULES,
        GradingRulesOutput,
        select_grading,
        with_tables=True,
    ),
]
TASKS_BY_NAME = {t.name: t for t in TASKS}

_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})


def normalize(text: str) -> str:
    """Для сравнения цитат и названий: без регистра, ё/е, тире и пробелов."""
    text = clean_text(text).translate(_DASHES).casefold().replace("ё", "е")
    return " ".join(text.split())


_ELLIPSIS_RE = re.compile(r"\.{3}|…")


def quote_in_text(quote: str, text: str) -> bool:
    """Цитата должна быть подстрокой текста. Многоточие внутри разрешено: модель так сокращает
    длинные места, поэтому каждый кусок между многоточиями проверяется отдельно."""
    parts = [normalize(p).strip(" .,;:") for p in _ELLIPSIS_RE.split(quote)]
    parts = [p for p in parts if p]
    haystack = normalize(text)
    return bool(parts) and all(p in haystack for p in parts)


def run_task(task: LLMTask, text: str, client: LLMClient) -> tuple[BaseModel | None, str | None]:
    """Один запрос с повторами. Возвращает (ответ, None) или (None, причина отказа)."""
    user = task.prompt.replace("{text}", text)
    schema = task.output.model_json_schema()
    error = ""
    for _ in range(MAX_RETRIES + 1):
        raw = client.complete_json(
            system=prompts.SYSTEM,
            user=user + prompts.REPAIR.format(error=error) if error else user,
            schema=schema,
        )
        try:
            return task.output.model_validate_json(raw), None
        except ValidationError as exc:
            error = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    return None, f"{task.name}: ответ не прошёл проверку после {MAX_RETRIES + 1} попыток ({error})"


def _verified(items: list, text: str, task: str, warnings: list[str]) -> list:
    kept = []
    for item in items:
        if quote_in_text(item.quote, text):
            kept.append(item)
        else:
            warnings.append(
                f"{task}: цитата не найдена в тексте раздела, элемент отброшен: {item.quote[:60]!r}"
            )
    return kept


def postprocess(output: BaseModel) -> BaseModel:
    """Детерминированная чистка ответа модели: то, что промпт гарантировать не может."""
    if isinstance(output, ActiveFormsOutput):
        return output.model_copy(update={"items": _merge_forms(output.items)})
    if isinstance(output, PrerequisitesOutput):
        seen: set[str] = set()
        items = []
        for item in output.items:
            if (key := normalize(item.name)) not in seen:
                seen.add(key)
                items.append(item)
        return output.model_copy(update={"items": items})
    return output


def _merge_forms(items: list[ActiveFormItem]) -> list[ActiveFormItem]:
    """Вид занятий в нижнем регистре; одна цитата с теми же часами — одна запись.

    «Практические и лабораторные занятия (24 часа)» модель иногда делит на два элемента по
    24 часа, и сумма часов получается вдвое больше. Сливаем, только если число часов в цитате
    встречается один раз: «семинары — 16 часов, практические занятия — 16 часов» — две формы.
    """
    merged: list[ActiveFormItem] = []
    for item in items:
        item = item.model_copy(update={"kind": item.kind.strip().lower()})
        twin = next(
            (
                m
                for m in merged
                if item.hours is not None
                and m.hours == item.hours
                and normalize(m.quote) == normalize(item.quote)
                and _count_number(item.quote, item.hours) == 1
            ),
            None,
        )
        if twin is None:
            merged.append(item)
        elif item.kind not in twin.kind:
            twin.kind = f"{twin.kind} и {item.kind}"
    return merged


def _count_number(text: str, number: float) -> int:
    """Сколько раз число стоит в тексте отдельным числом (запятая как десятичный разделитель)."""
    return len(re.findall(rf"(?<![\d.]){number:g}(?![\d.])", text.replace(",", ".")))


def section_text(sections: list[MdSection], with_tables: bool) -> str:
    """Текст для одного вызова. Заголовки безномерных подразделов остаются: по ним видно, к чему
    относится следующий кусок (например, «Критерии оценивания заданий контрольной работы»)."""
    parts = []
    for i, s in enumerate(sections):
        lines = (s.full_text if with_tables else s.text).split("\n")
        body = [ln for ln in lines if ln.strip()]
        if body and i > 0 and s.number is None:
            body.insert(0, s.heading)
        parts += body
    return "\n".join(parts)


def _locate(quote: str, sections: list[MdSection], file: str, with_tables: bool) -> SourceRef:
    """Раздел, в котором нашлась цитата (по умолчанию — первый раздел задачи)."""
    for s in sections:
        if quote_in_text(quote, s.full_text if with_tables else s.text):
            return SourceRef(file=file, section=s.heading, line=s.line)
    first = sections[0]
    return SourceRef(file=file, section=first.heading, line=first.line)


def extract_with_llm(
    md: str, file: str, client: LLMClient, tasks: list[str] | None = None
) -> LLMExtraction:
    """Прогоняет выбранные задачи (по умолчанию все): один вызов на задачу и документ."""
    result = LLMExtraction(file=file)
    sections = split_sections(md)
    for task in (TASKS_BY_NAME[n] for n in tasks) if tasks else TASKS:
        picked = task.select(sections)
        text = section_text(picked, task.with_tables) if picked else ""
        if not text:
            continue
        output, problem = run_task(task, text, client)
        if output is None:
            result.warnings.append(f"{problem} [{picked[0].heading[:40]}]")
            continue
        _collect(task, postprocess(output), text, picked, file, result)
    return result


def _collect(
    task: LLMTask,
    output: BaseModel,
    text: str,
    sections: list[MdSection],
    file: str,
    result: LLMExtraction,
) -> None:
    name = task.name

    def source(item: BaseModel) -> SourceRef:
        return _locate(item.quote, sections, file, task.with_tables)

    if isinstance(output, PrerequisitesOutput):
        for item in _verified(output.items, text, name, result.warnings):
            result.prerequisites.append(LLMPrerequisite(**item.model_dump(), source=source(item)))
    elif isinstance(output, ActiveFormsOutput):
        if output.total_hours is not None and re.search(
            rf"\b{output.total_hours:g}\b", text.replace(",", ".")
        ):
            result.active_total_hours = output.total_hours
        elif output.total_hours is not None:
            result.warnings.append(
                f"{name}: общий объём {output.total_hours:g} ч не найден в тексте"
            )
        for item in _verified(output.items, text, name, result.warnings):
            result.active_forms.append(LLMActiveForm(**item.model_dump(), source=source(item)))
    elif isinstance(output, GradingRulesOutput):
        for item in _verified(output.rules, text, name, result.warnings):
            result.grading_rules.append(LLMGradingRule(**item.model_dump(), source=source(item)))
