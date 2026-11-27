"""Оркестратор: вид запроса и объём работы (ADR-0003).

Оркестратор — агент: он читает диалог сессии, состояние сессии получает в системном сообщении,
замечания — инструментом `get_findings`. Итог — `OrchestratorDecision`. Если без ответа
пользователя нельзя выбрать вид запроса, дисциплины или код направления для правил ФГОС, итог
— `unclear` с вопросом: ответ приходит в диалог, и оркестратор решает заново. Инструмента
`ask_user` у него нет: с ним модель начинает отвечать пользователю сама. Код проверяет итог по
данным: дисциплины — среди загруженных документов, код направления — в документах или в
репликах пользователя и в каталоге правил. Если интерфейс задал `action` и код направления не
нужен, LLM не вызывается.
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from edu_doc_agent.orchestration.agent import run_agent, to_json
from edu_doc_agent.orchestration.check import course_matches
from edu_doc_agent.orchestration.history import dialogue
from edu_doc_agent.orchestration.state import (
    Action,
    GraphState,
    RequestKind,
    Scope,
    empty_scope,
    question,
)
from edu_doc_agent.orchestration.tools import SessionTools

SKIP_FGOS = "Проверить без ФГОС"

ACTION_KINDS: dict[Action, RequestKind] = {
    "check": "check",
    "generate": "change",
    "qa": "question",
}

SYSTEM_PROMPT = """\
Ты оркестратор системы проверки документов образовательной программы СПбГУ. Ты не отвечаешь \
пользователю: ты определяешь вид запроса и объём работы, а работу выполнят другие агенты.

Виды запроса:
- question: вопрос о программе, замечаниях или нормативных документах;
- check: проверить документы на соответствие ФГОС, нормам СПбГУ и друг другу;
- change: изменить или дополнить документы, в том числе исправить найденные замечания (их ID \
— в finding_ids, найди их через get_findings);
- check_and_fix: проверить документы и исправить найденные нарушения — только если \
пользователь просит и проверить, и исправить.

Запрос — последнее сообщение пользователя в диалоге. Если оно продолжает прежний разговор \
(уточняет, отвечает на вопрос системы, соглашается с её предложением), отнеси его к тому, что \
обсуждалось. Последнее изменение сессии — last_change в состоянии сессии.

Объём работы указывай, только если пользователь его ограничил:
- courses — дисциплины из загруженных документов, точными названиями из состояния сессии; \
если пользователь исключает дисциплины, перечисли остальные; дисциплину, которой нет среди \
документов, укажи в missing_courses; название программы — не дисциплина;
- rule_sets — наборы правил (fgos, spbu, cross_check); finding_ids — ID замечаний; \
change_goal — что изменить, своими словами.

Код направления (только для check и check_and_fix): что о нём известно, сказано в состоянии \
сессии. Если код в документах один, direction_code не нужен. Иначе возьми код, который \
пользователь назвал в диалоге, или спроси его (kind=unclear, about=direction_code): код \
пользователь напишет в чате, кнопку «Проверить без ФГОС» добавит код, options не нужны. \
Если пользователь назвал код, для которого правил ФГОС нет, не предлагай \
вместо него другое направление: объясни, что по ФГОС это направление проверить нельзя, \
остальные правила будут проверены, и спроси, проверять ли без ФГОС или код указан с ошибкой. \
Если пользователь не хочет проверять по ФГОС или для его кода правил нет, skip_fgos=true, а \
direction_code — код, который он назвал (если назвал); rule_sets из-за этого не меняй. Если \
ответ не дал кода, объясни в вопросе, что известно из документов, и спроси снова.

Если без ответа пользователя нельзя выбрать вид запроса, дисциплины или код направления, \
kind=unclear, а в question — один короткий уточняющий вопрос (варианты — в options). \
Остальное не спрашивай: на вопрос ответит Q&A-агент, новые значения, тексты и подробности \
изменения спросит агент, который его готовит. Загружены ли документы, на вид запроса не \
влияет. Итог — вызов OrchestratorDecision."""


class ScopeDecision(BaseModel):
    courses: list[str] = []
    missing_courses: list[str] = Field(
        default_factory=list, description="дисциплины из запроса, которых нет среди документов"
    )
    rule_sets: list[str] = []
    finding_ids: list[str] = []
    change_goal: str | None = None


class OrchestratorDecision(BaseModel):
    """Вид запроса и объём работы или (`unclear`) уточняющий вопрос пользователю."""

    kind: RequestKind | Literal["unclear"] = Field(description="вид запроса")
    question: str | None = Field(None, description="при unclear: вопрос пользователю")
    options: list[str] = Field(default_factory=list, description="при unclear: варианты ответа")
    about: Literal["kind", "courses", "direction_code"] = Field(
        "kind", description="при unclear: о чём вопрос — вид запроса, дисциплины, код направления"
    )
    scope: ScopeDecision = ScopeDecision()
    direction_code: str | None = Field(
        None, description="код направления для правил ФГОС, если в документах он не один"
    )
    skip_fgos: bool = Field(False, description="пользователь не хочет проверять по ФГОС")


async def _system(
    state: GraphState, tools: SessionTools, codes: list[str], available: list[str]
) -> str:
    """Промпт с состоянием сессии: это не реплики диалога. Состояние — в промпте, а не
    результатом инструмента: увидев «документов нет» в результате вызова, модель берётся
    отвечать пользователю вместо выбора вида запроса."""
    known = f"в документах: {', '.join(codes)}" if codes else "в документах не найден"
    lines = [
        SYSTEM_PROMPT,
        "",
        f"Состояние сессии: {to_json(await tools.get_current_state())}",
        f"Код направления {known}; правила ФГОС есть для: {', '.join(available) or 'нет'}.",
    ]
    if state["action"] is not None:
        lines.append(f"Вид запроса задан интерфейсом: {ACTION_KINDS[state['action']]}.")
    if state["ingest_errors"]:
        lines.append("Не удалось обработать файлы: " + "; ".join(state["ingest_errors"]))
    return "\n".join(lines)


def unknown_courses(courses: list[str], titles: list[str]) -> list[str]:
    """Дисциплины из итога оркестратора, которых нет среди загруженных документов."""
    return [c for c in courses if not any(course_matches(t, c) for t in titles)]


def _needs_code(kind: str, rule_sets: list[str], state: GraphState, codes: list[str]) -> bool:
    """Нужен ли код направления от пользователя: проверка загруженных документов с правилами
    ФГОС, а в документах кода нет или их несколько."""
    if kind not in ("check", "check_and_fix") or (rule_sets and "fgos" not in rule_sets):
        return False
    return bool(state["document_ids"]) and len(codes) != 1 and state["direction_code"] is None


def _from_user(code: str, history: list[BaseMessage]) -> bool:
    """Код есть в репликах пользователя."""
    return any(code in str(m.content) for m in history if isinstance(m, HumanMessage))


def _code_problems(
    decision: OrchestratorDecision,
    state: GraphState,
    codes: list[str],
    available: list[str],
    history: list[BaseMessage],
) -> list[str]:
    allowed = codes or available
    code = decision.direction_code
    if code in codes or (decision.skip_fgos and (code is None or code not in available)):
        return []  # отказ от ФГОС: без кода или с кодом, для которого правил нет
    if code is None and not _needs_code(decision.kind, decision.scope.rule_sets, state, codes):
        return []
    if code in allowed and _from_user(code, history):
        return []
    if code in allowed:  # код из каталога, но пользователь его не называл: угадан
        return [
            f"кода {code} нет ни в документах, ни в репликах пользователя — не угадывай: "
            "спроси пользователя (kind=unclear, about=direction_code); если он не хочет "
            "проверять по ФГОС, skip_fgos=true"
        ]
    if code and not codes and _from_user(code, history):  # код пользователя без правил
        return [
            f"для направления {code} правил ФГОС в каталоге нет. Не предлагай вместо него "
            "другое направление: объясни это и спроси (kind=unclear, about=direction_code), "
            "проверять без ФГОС или "
            "код указан с ошибкой; если без ФГОС — skip_fgos=true и direction_code="
            f"{code}"
        ]
    where = "в документах указаны" if codes else "правила ФГОС в каталоге есть для"
    given = f"для кода {code} проверка невозможна" if code else "нужен код направления"
    return [
        f"{given}: {where} {', '.join(allowed)}. Возьми код из диалога или спроси пользователя "
        "(kind=unclear, about=direction_code); если он не хочет проверять по ФГОС, skip_fgos=true"
    ]


async def decide(
    model: BaseChatModel,
    state: GraphState,
    tools: SessionTools,
    titles: list[str],
    codes: list[str],
    available: list[str],
) -> dict:
    """Обновление состояния: вид и объём работы либо вопрос пользователю. `titles` —
    названия загруженных дисциплин, `codes` — коды направления из документов, `available` —
    коды, для которых есть правила ФГОС."""
    fixed = ACTION_KINDS[state["action"]] if state["action"] is not None else None
    if fixed is not None and not _needs_code(fixed, [], state, codes):
        update: dict = {"kind": fixed, "scope": empty_scope()}
        return update | ({"direction_code": codes[0]} if len(codes) == 1 else {})

    async def validate(name: str, result: BaseModel, history: list[BaseMessage]) -> list[str]:
        assert isinstance(result, OrchestratorDecision)
        if fixed is not None and result.kind not in (fixed, "unclear"):
            return [f"вид запроса задан интерфейсом: kind={fixed}"]
        if result.kind == "unclear":
            return [] if result.question else ["при kind=unclear нужен question"]
        problems = _code_problems(result, state, codes, available, history)
        if unknown := unknown_courses(result.scope.courses, titles):
            loaded = ", ".join(f"«{t}»" for t in titles) or "нет ни одной"
            problems.append(
                f"среди документов нет дисциплин: {', '.join(unknown)}. Загружены: {loaded}. "
                "Укажи названия как в документах; дисциплину, которой нет, — в "
                "missing_courses; если неясно, о какой дисциплине речь, спроси (kind=unclear)"
            )
        return problems

    decision = await run_agent(
        model,
        system=await _system(state, tools, codes, available),
        messages=dialogue(state["messages"], state["summary"]),
        tools=tools.as_tools(["get_findings"]),
        final=OrchestratorDecision,
        validate=validate,
    )
    if decision.kind == "unclear":  # ответ придёт в диалог, оркестратор решит заново
        assert decision.question is not None
        # код направления пользователь пишет сам; кнопка — только отказ от ФГОС
        options = [SKIP_FGOS] if decision.about == "direction_code" else decision.options
        pending = question("clarification", decision.question, "orchestrator", options=options)
        return {"pending": pending}
    scope: Scope = {
        # недостающие дисциплины — в объёме: проверка сообщит, что их документов нет
        "courses": [*decision.scope.courses, *decision.scope.missing_courses],
        "rule_sets": decision.scope.rule_sets,
        "finding_ids": decision.scope.finding_ids,
        "change_goal": decision.scope.change_goal,
    }
    update = {"kind": decision.kind, "scope": scope}
    if len(codes) == 1:
        update["direction_code"] = codes[0]
    elif decision.skip_fgos:  # код без правил сохраняется: отчёт назовёт его
        code = decision.direction_code
        update["direction_code"] = code if code and code not in available else ""
    elif decision.direction_code:
        update["direction_code"] = decision.direction_code
    return update
