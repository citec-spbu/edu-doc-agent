"""Проверка черновика изменения: правила по данным после правки и опора правок.

Черновик проверяют правила `method: code`, затем `method: llm`, которые используют изменённые
поля. Опора правки — данные или ответ пользователя: текст целей, миссии, аннотации и
результатов обучения, новый код компетенции и удаление компетенции генератор не придумывает.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import get_args

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from edu_doc_agent.extraction.codes import canonical_code
from edu_doc_agent.extraction.curriculum import CurriculumPlan
from edu_doc_agent.extraction.models import RPDExtraction
from edu_doc_agent.orchestration.agent import user_answer
from edu_doc_agent.orchestration.checker import check_rule
from edu_doc_agent.orchestration.deps import GraphDeps
from edu_doc_agent.orchestration.edits import (
    apply_edits,
    get_value,
    path_parts,
    rules_using,
    top_field,
)
from edu_doc_agent.orchestration.program import program_documents
from edu_doc_agent.orchestration.stores import ChangeEdit, StoredDocument
from edu_doc_agent.validation.catalog import Rule
from edu_doc_agent.validation.checks.competences import same_wording
from edu_doc_agent.validation.findings import Finding
from edu_doc_agent.validation.runner import CheckResults, check_with_code
from edu_doc_agent.validation.units import plan_entries_for, units

# поля, текст которых пишет автор документа
_AUTHORED = re.compile(r"^(goals|mission|annotation|learning_outcomes(/\d+(/text)?)?)$")
_CODE_PATH = re.compile(r"^(competences/\d+/code|learning_outcomes/\d+/competence_codes/\d+)$")
_COMPETENCE = re.compile(r"^competences/(\d+)$")


@dataclass
class DraftReview:
    consequences: list[str]
    problems: list[str]
    remaining: list[str]  # замечания цели, которые черновик не исправляет


def _violations(results: CheckResults) -> dict[str, Finding]:
    return {f.id: f for f in results.findings if f.status == "violation"}


async def review_draft(
    deps: GraphDeps,
    session_id: str,
    run_id: str,
    rules: list[Rule],
    documents: list[StoredDocument],
    edits: list[ChangeEdit],
    targets: list[str],
    *,
    optional: Sequence[str] = (),
    user_paths: Sequence[str] = (),
) -> DraftReview:
    """Последствия и проблемы черновика. Цель из `optional` может остаться (пользователь
    просит её не исправлять). Нарушение правила `method: llm` по тексту из `user_paths`
    (его дал пользователь) — предупреждение, а не проблема: решает пользователь."""
    draft, errors = apply_edits(documents, edits)
    if errors:
        return DraftReview([], errors, [])
    before, _ = program_documents(documents)
    after, texts = program_documents(draft)
    affected = {e.document_id for e in edits}
    old, new = (
        _violations(
            check_with_code(rules, p, session_id=session_id, run_id=run_id, document_ids=affected)
        )
        for p in (before, after)
    )
    using = rules_using(rules, {top_field(e.path) for e in edits})
    consequences = _hours(draft, edits)
    if using:
        consequences.append(
            "изменённые данные используют правила: " + "; ".join(r.title for r in using)
        )
    consequences += [
        f"будет снято замечание «{f.rule_title}» ({f.course or 'программа'})"
        for fid, f in old.items()
        if fid not in new
    ]
    problems = [
        f"появится нарушение «{f.rule_title}» ({f.course}): {f.message}"
        for fid, f in new.items()
        if fid not in old
    ]
    remaining = [fid for fid in targets if fid in new]
    problems += [
        f"замечание {fid} осталось: {new[fid].message}" for fid in remaining if fid not in optional
    ]
    if problems:
        return DraftReview(consequences, problems, remaining)

    for rule in (r for r in using if r.method == "llm"):
        for unit in units(rule.applies_to, rule.doc_kinds, after):
            if unit.document_id not in affected:
                continue
            verdict = await check_rule(
                deps.models("checker"), deps.models("reviewer"), rule, unit, texts, after
            )
            if verdict.outcome.status != "violation":
                continue
            text = f"«{rule.title}» ({unit.course}): {verdict.outcome.message}"
            paths = [e.path for e in edits if top_field(e.path) in rule.fields]
            if paths and set(paths) <= set(user_paths):
                consequences.append(f"после правки будет нарушено {text} (текст дал пользователь)")
            else:
                problems.append(f"после правки нарушено {text}")
    return DraftReview(consequences, problems, remaining)


def _hours(draft: list[StoredDocument], edits: list[ChangeEdit]) -> list[str]:
    """Пересчитанные часы строк таблицы трудоёмкости, которые меняет правка."""
    out = []
    for e in edits:
        parts = path_parts(e.path)
        if parts[0] != "workload_table" or len(parts) < 2 or e.op != "set":
            continue
        data = next(d.data for d in draft if d.document_id == e.document_id)
        row = get_value(data.model_dump(mode="json"), "/".join(map(str, parts[:2])))
        hours = sum(row.get("contact_hours", {}).values()) + sum(
            row.get("self_study_hours", {}).values()
        )
        out.append(f"всего часов в строке таблицы трудоёмкости: {hours:g}")
    return out


def fgos_wordings(rules: list[Rule]) -> dict[str, str]:
    """Формулировки компетенций ФГОС из правил каталога: канонический код → текст."""
    return {
        canonical_code(code): text
        for r in rules
        if r.check == "competence_wording_matches"
        for code, text in r.params.get("wordings", {}).items()
    }


def unsupported_edits(
    edits: list[ChangeEdit],
    answer_ids: list[str | None],
    documents: dict[str, StoredDocument],
    rules: list[Rule],
    history: list[BaseMessage],
) -> list[str]:
    """Правки без опоры в данных или в ответе пользователя (`answer_ids` — answer_id правок,
    ответ ищется по вызову ask_user в работе агента)."""
    plan = next((d.data for d in documents.values() if isinstance(d.data, CurriculumPlan)), None)
    wordings = fgos_wordings(rules)
    problems = []
    for e, answer_id in zip(edits, answer_ids, strict=True):
        answer = user_answer(history, answer_id)
        given = answer["answer"].strip() if answer else None
        if e.op != "remove" and _AUTHORED.match(e.path):
            problems += _authored_problem(e, given, answer_id)
        if (code := _new_code(e)) is not None and not _code_grounded(
            e, code, documents[e.document_id], plan, wordings, given
        ):
            where = "учебного плана нет" if plan is None else "в учебном плане у дисциплины его нет"
            problems.append(
                f"{e.path}: код {code} не подтверждён ({where}, формулировка компетенции в "
                "документе не совпадает с формулировкой этого кода во ФГОС) — спроси код у "
                "пользователя (ask_user, варианты в options) и укажи answer_id ответа"
            )
        if isinstance(documents[e.document_id].data, CurriculumPlan) and answer is None:
            problems.append(
                f"{e.path}: учебный план — источник истины, его не меняй: исправь документ, "
                "который с ним расходится; менять план можно, только если об этом попросил "
                "пользователь (укажи answer_id ответа)"
            )
        if e.op == "remove" and _COMPETENCE.match(e.path) and answer is None:
            problems.append(
                f"{e.path}: компетенцию не удаляй без согласия пользователя — спроси его "
                "(ask_user) и укажи answer_id ответа; если код ошибочный, исправь код"
            )
    return problems


def user_paths(
    edits: list[ChangeEdit], answer_ids: list[str | None], history: list[BaseMessage]
) -> list[str]:
    """Пути правок, значение которых дал или одобрил пользователь."""
    return [
        e.path
        for e, answer_id in zip(edits, answer_ids, strict=True)
        if user_answer(history, answer_id) is not None
    ]


def _authored_problem(e: ChangeEdit, given: str | None, answer_id: str | None) -> list[str]:
    new = e.new.get("text") if isinstance(e.new, dict) else e.new
    if not isinstance(new, str) or not new.strip():
        return []
    if given is None:
        return [
            f"{e.path}: новый текст не придумывай — возьми его из ответа пользователя или "
            "предложи свой вариант (ask_user, текст целиком в draft) и укажи answer_id ответа"
        ]
    if given != new.strip():
        return [
            f"{e.path}: текст не совпадает с ответом {answer_id}. Записывай ответ пользователя "
            "как есть; если ты его доработал, предложи итоговый текст (ask_user, draft) и укажи "
            "answer_id того ответа"
        ]
    return []


def _code_grounded(
    e: ChangeEdit,
    code: str,
    course: StoredDocument,
    plan: CurriculumPlan | None,
    wordings: dict[str, str],
    given: str | None,
) -> bool:
    """Код из строк учебного плана дисциплины, по формулировке ФГОС или от пользователя."""
    key = canonical_code(code)
    rows = plan_entries_for(course.data, plan) if isinstance(course.data, RPDExtraction) else ()
    if key in {canonical_code(c) for row in rows for c in row.competences}:
        return True
    wording, reference = _wording(e, course), wordings.get(key)
    if wording and reference and same_wording(wording, reference):
        return True
    return given is not None and canonical_code(given) == key


def _new_code(e: ChangeEdit) -> str | None:
    """Новый код компетенции, если правка его вводит."""
    if e.op == "remove":
        return None
    if _CODE_PATH.match(e.path) and isinstance(e.new, str):
        return e.new
    if isinstance(e.new, dict) and (e.path == "competences" or _COMPETENCE.match(e.path)):
        old = e.old.get("code") if isinstance(e.old, dict) else None
        code = e.new.get("code")
        return code if isinstance(code, str) and code != old else None
    return None


def _wording(e: ChangeEdit, document: StoredDocument) -> str | None:
    """Формулировка компетенции, код которой меняет правка."""
    if isinstance(e.new, dict) and isinstance(e.new.get("name"), str):
        return e.new["name"]
    m = re.match(r"^competences/(\d+)", e.path)
    data = document.data
    if m and isinstance(data, RPDExtraction) and int(m.group(1)) < len(data.competences):
        return data.competences[int(m.group(1))].name
    return None


def _model_fields(model: type[BaseModel], seen: set[type]) -> set[str]:
    """Имена полей модели данных и вложенных моделей."""
    if model in seen:
        return set()
    seen.add(model)
    names = set(model.model_fields)
    for field in model.model_fields.values():
        stack = [field.annotation]
        while stack:
            tp = stack.pop()
            if isinstance(tp, type) and issubclass(tp, BaseModel):
                names |= _model_fields(tp, seen)
            stack += list(get_args(tp))
    return names


def internal_names(
    texts: list[str], rules: list[Rule], documents: list[StoredDocument]
) -> list[str]:
    """Служебные имена в текстах для пользователя: ID правил и документов, поля данных."""
    seen: set[type] = set()
    names = {r.id for r in rules} | {d.document_id for d in documents}
    for d in documents:
        names |= _model_fields(type(d.data), seen)
    text = "\n".join(texts)
    return sorted(n for n in names if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text))
