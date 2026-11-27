"""Подграф «Проверка» (ADR-0003, «Подграфы»).

    select ─┬─> code_rules (по набору правил) ─┬─> assemble
            └─> checker (правило × единица)  ──┘

1. `select` выбирает правила по `Scope` и коду направления и раскладывает работу.
2. Правила `method: code` каждого набора (ФГОС, нормы СПбГУ, сверка документов) проверяет
   код в отдельной ветке. Ревьюер эти замечания не перепроверяет.
3. Правила `method: llm` проверяет агент-чекер, отдельная ветка на пару «правило, единица
   проверки»; ответ проверяет код, нарушение — агент-ревьюер (`checker.py`).
4. `assemble` записывает замечания и формирует итог: число нарушений, `not_checked`, версию
   каталога правил и редакции нормативов.

Ветки выполняются параллельно; результаты успешных веток сохраняет Checkpointer, после сбоя
повторяется только упавшая ветка.
"""

import asyncio
import logging
import operator
import re
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.orchestration.checker import check_rule
from edu_doc_agent.orchestration.deps import GraphDeps
from edu_doc_agent.orchestration.program import load_program, session_id_of
from edu_doc_agent.orchestration.state import CheckSummary, Scope, empty_scope
from edu_doc_agent.orchestration.structured import user_error
from edu_doc_agent.validation.catalog import Rule, load_rules
from edu_doc_agent.validation.checks import not_checked
from edu_doc_agent.validation.findings import Finding, finding_id
from edu_doc_agent.validation.runner import check_with_code, make_result
from edu_doc_agent.validation.units import ProgramDocuments, Unit, units

logger = logging.getLogger(__name__)

# предел времени на одну пару «правило, единица» для method: llm: зависший вызов провайдера
# (таймаут клиента × повторы — до 15 минут) иначе держит всю проверку
LLM_RULE_TIMEOUT = 300.0


class CheckInput(TypedDict):
    run_id: str
    scope: Scope | None
    finding_ids: list[str]
    direction_code: str | None


class CheckOutput(TypedDict):
    finding_ids: list[str]
    check_summary: CheckSummary | None


class CheckState(CheckInput, CheckOutput):
    work: list[dict[str, Any]]  # ветки: {"node": ..., аргументы}
    # результаты веток: {"id": ID замечания, "finding": dict | None}
    results: Annotated[list[dict[str, Any]], operator.add]


def catalog_version(rules_dir: Path) -> str:
    digest = sha256()
    for path in sorted(rules_dir.rglob("*.yaml")):
        digest.update(path.relative_to(rules_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _stems(text: str) -> set[str]:
    """Основы слов: до 5 букв и без последней буквы — окончание («химия» → «хими»)."""
    return {w[: min(5, len(w) - 1)] for w in re.findall(r"\w{4,}", normalize(text))}


def course_matches(title: str, query: str) -> bool:
    """Название из запроса подходит к дисциплине: вхождение или каждая основа слова названия
    начинает слово запроса («РПД по машинному обучению» → «Машинное обучение», «химию» →
    «Химия»)."""
    t, q = normalize(title), normalize(query)
    if t in q or q in t:
        return True
    stems = _stems(title)
    words = re.findall(r"\w+", q)
    return bool(stems) and all(any(w.startswith(s) for w in words) for s in stems)


def _title_matches(title: str | None, courses: list[str]) -> bool:
    if not courses:
        return True
    return title is not None and any(course_matches(title, c) for c in courses)


def fgos_codes(rules_dir: Path) -> list[str]:
    """Коды направлений, для которых в каталоге есть правила ФГОС."""
    return sorted(p.stem for p in (rules_dir / "fgos").glob("*.yaml"))


def select_rules(rules_dir: Path, direction_code: str | None, scope: Scope) -> list[Rule]:
    rules = load_rules(rules_dir, [direction_code] if direction_code else [])
    if scope["rule_sets"]:
        rules = [r for r in rules if r.rule_set in scope["rule_sets"]]
    return rules


def scoped_units(rule: Rule, program: ProgramDocuments, scope: Scope) -> list[Unit]:
    if scope["courses"] and rule.applies_to == "program":
        return []  # проверка названных дисциплин: правила программы не нужны
    return [
        u
        for u in units(rule.applies_to, rule.doc_kinds, program)
        if _title_matches(u.course, scope["courses"])
    ]


def _fgos_not_checked(session_id: str, run_id: str) -> Finding:
    return Finding(
        id=finding_id("fgos", None),
        session_id=session_id,
        run_id=run_id,
        rule_id="fgos",
        rule_title="Правила ФГОС",
        rule_set="fgos",
        severity="error",
        fixable=False,
        document_id=None,
        course=None,
        status="not_checked",
        message="не указан код направления, правила ФГОС не проверялись",
    )


def build_check_subgraph(deps: GraphDeps) -> CompiledStateGraph:
    async def select(state: CheckState, config: RunnableConfig) -> dict:
        session_id = session_id_of(config)
        scope = state["scope"] or empty_scope()
        program, _ = await load_program(deps, session_id)
        if not program.texts and not program.courses:
            return {"work": [], "results": []}  # документов нет: проверять нечего
        rules = select_rules(deps.rules_dir, state["direction_code"], scope)
        work: list[dict[str, Any]] = []
        for rule_set in dict.fromkeys(r.rule_set for r in rules if r.method == "code"):
            work.append({"node": "code_rules", "rule_set": rule_set})
        for rule in rules:
            if rule.method == "llm":
                for unit in scoped_units(rule, program, scope):
                    work.append(
                        {"node": "checker", "rule_id": rule.id, "unit_id": unit.document_id}
                    )
        results = []
        if _fgos_wanted(scope) and not state["direction_code"]:
            f = _fgos_not_checked(session_id, state["run_id"])
            results.append({"id": f.id, "finding": f.model_dump(mode="json")})
        return {"work": work, "results": results}

    def fan_out(state: CheckState) -> list[Send] | str:
        if not state["work"]:
            return "assemble"
        common = {
            "run_id": state["run_id"],
            "scope": state["scope"],
            "direction_code": state["direction_code"],
        }
        return [Send(w["node"], common | w) for w in state["work"]]

    async def code_rules(branch: dict, config: RunnableConfig) -> dict:
        session_id = session_id_of(config)
        scope = branch["scope"] or empty_scope()
        program, _ = await load_program(deps, session_id)
        rules = [
            r
            for r in select_rules(deps.rules_dir, branch["direction_code"], scope)
            if r.rule_set == branch["rule_set"]
        ]
        document_ids = None
        if scope["courses"]:
            document_ids = {
                c.document_id
                for c in program.courses
                if _title_matches(c.data.title, scope["courses"])
            }
            rules = [r for r in rules if r.applies_to != "program"]
        checked = check_with_code(
            rules,
            program,
            session_id=session_id,
            run_id=branch["run_id"],
            document_ids=document_ids,
        )
        return {"results": [_result(r.unit_id, r.finding) for r in checked.results]}

    async def checker(branch: dict, config: RunnableConfig) -> dict:
        session_id = session_id_of(config)
        scope = branch["scope"] or empty_scope()
        program, texts = await load_program(deps, session_id)
        rule = next(
            r
            for r in select_rules(deps.rules_dir, branch["direction_code"], scope)
            if r.id == branch["rule_id"]
        )
        unit = next(
            u
            for u in units(rule.applies_to, rule.doc_kinds, program)
            if u.document_id == branch["unit_id"]
        )
        try:
            verdict = await asyncio.wait_for(
                check_rule(
                    deps.models("checker"), deps.models("reviewer"), rule, unit, texts, program
                ),
                LLM_RULE_TIMEOUT,
            )
            outcome, evidence = verdict.outcome, verdict.evidence
            extra = {"needs_human": verdict.needs_human, "opinions": verdict.opinions}
        except TimeoutError:
            minutes = f"{LLM_RULE_TIMEOUT / 60:g}"
            outcome, evidence, extra = not_checked(f"LLM не ответила за {minutes} мин"), [], {}
        except Exception as error:  # временные ошибки LLM исчерпаны: правило не проверено
            outcome, evidence, extra = not_checked(user_error(error)), [], {}
        logger.info(
            "агент-чекер %s / %s: %s %s", rule.id, unit.course, outcome.status, outcome.message
        )
        result = make_result(
            rule,
            unit,
            outcome,
            session_id=session_id,
            run_id=branch["run_id"],
            evidence=evidence,
            **extra,
        )
        return {"results": [_result(result.unit_id, result.finding)]}

    async def assemble(state: CheckState, config: RunnableConfig) -> dict:
        session_id = session_id_of(config)
        program, _ = await load_program(deps, session_id)
        # одинаковые замечания (то же правило и та же единица) объединяются
        by_id: dict[str, dict | None] = {}
        for r in state["results"]:
            by_id[r["id"]] = r["finding"]
        findings = [Finding.model_validate(f) for f in by_id.values() if f is not None]
        passed_ids = [fid for fid, f in by_id.items() if f is None]
        await deps.stores.delete_findings(session_id, passed_ids)
        await deps.stores.save_findings(session_id, findings)

        rules = select_rules(
            deps.rules_dir, state["direction_code"], state["scope"] or empty_scope()
        )
        editions = sorted(
            {
                f"{r.requirement.clause_id.split(':')[0]}: {r.requirement.edition}"
                for r in rules
                if r.requirement
            }
        )
        new_ids = [f.id for f in findings]
        scope = state["scope"] or empty_scope()
        summary: CheckSummary = {
            "rules": len(rules),
            "rule_sets": scope["rule_sets"],
            "passed": len(passed_ids),
            "finding_ids": new_ids,
            "missing": _missing(scope, state["direction_code"], fgos_codes(deps.rules_dir)),
            "direction_code": state["direction_code"],
            "unknown_courses": [
                c
                for c in scope["courses"]
                if not any(course_matches(d.data.title or "", c) for d in program.courses)
            ],
            "catalog_version": catalog_version(deps.rules_dir),
            "editions": editions,
        }
        evaluated = set(by_id)
        return {
            "finding_ids": [i for i in state["finding_ids"] if i not in evaluated] + new_ids,
            "check_summary": summary,
        }

    graph = StateGraph(CheckState, input_schema=CheckInput, output_schema=CheckOutput)
    graph.add_node("select", select)
    graph.add_node("code_rules", code_rules)
    graph.add_node("checker", checker)
    graph.add_node("assemble", assemble)
    graph.add_edge(START, "select")
    graph.add_conditional_edges("select", fan_out, ["code_rules", "checker", "assemble"])
    graph.add_edge("code_rules", "assemble")
    graph.add_edge("checker", "assemble")
    graph.add_edge("assemble", END)
    return graph.compile()


def _fgos_wanted(scope: Scope) -> bool:
    return not scope["rule_sets"] or "fgos" in scope["rule_sets"]


def _missing(scope: Scope, direction_code: str | None, available: list[str]) -> list[str]:
    """Почему правила ФГОС не применялись: пользователь отказался от них, для кода нет правил
    в каталоге или код не указан. Об отказе и коде без правил протокол сообщает и при наборах
    правил без ФГОС: иначе не видно, что ФГОС не проверен."""
    if direction_code == "":
        return ["fgos_declined"]
    if direction_code and direction_code not in available:
        return ["fgos_rules"]
    return ["direction_code"] if _fgos_wanted(scope) and not direction_code else []


def _result(fid: str, finding: Finding | None) -> dict[str, Any]:
    return {"id": fid, "finding": finding.model_dump(mode="json") if finding else None}
