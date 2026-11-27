"""Подграф «Генерация» (ADR-0003, «Подграфы»): изменение данных документов.

1. Код определяет цели: замечания из `Scope`, замечания с `fixable: true` после проверки или
   цель изменения из запроса.
2. Агент-генератор находит изменяемые значения инструментами и предлагает точечные правки
   (`propose_change`). Чего не хватает, он спрашивает у пользователя (`ask_user`).
3. Код проверяет черновик (`draft.py`): правила по данным после правки и опору правок. Если
   исправляемое замечание осталось или появились новые, правка возвращается агенту.
4. Граф показывает черновик с кнопками «Применить» и «Отклонить» (`graph.py`). Свой текст
   пользователя получает генератор: поясняет черновик, убирает часть правок, готовит новый
   черновик или отклоняет его.

Правки вносятся в данные извлечения после подтверждения (`corrected.py`: данные, перепроверка,
исправленный DOCX).
"""

from hashlib import sha256
from typing import Annotated, Any, Literal, TypedDict

import openai
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from edu_doc_agent.orchestration.agent import (
    AgentLimitError,
    AgentTurn,
    AskUser,
    run_agent_turn,
    to_json,
)
from edu_doc_agent.orchestration.check import select_rules
from edu_doc_agent.orchestration.deps import GraphDeps
from edu_doc_agent.orchestration.draft import (
    internal_names,
    review_draft,
    unsupported_edits,
    user_paths,
)
from edu_doc_agent.orchestration.edits import get_value
from edu_doc_agent.orchestration.history import dialogue
from edu_doc_agent.orchestration.program import session_id_of
from edu_doc_agent.orchestration.state import (
    CheckSummary,
    PendingQuestion,
    RequestKind,
    Scope,
    empty_scope,
    question,
)
from edu_doc_agent.orchestration.stores import Change, ChangeEdit, StoredDocument
from edu_doc_agent.orchestration.structured import user_error
from edu_doc_agent.orchestration.tools import SessionTools
from edu_doc_agent.reporting.protocol import PROGRAM, by_document
from edu_doc_agent.validation.catalog import Rule
from edu_doc_agent.validation.checks import CHECK_FIELDS
from edu_doc_agent.validation.findings import Finding

GENERATOR_TOOLS = [
    "get_findings",
    "get_rule",
    "get_course",
    "find_course",
    "get_plan_entries",
    "get_plan_summary",
    "get_matrix_rows",
    "get_field_values",
    "search_document",
    "get_document_outline",
    "read_section",
]

GENERATOR_PROMPT = """\
Ты агент-генератор системы проверки документов образовательной программы СПбГУ. Ты готовишь \
точечные правки данных документов: какое значение заменить и на что.

- Найди изменяемые значения инструментами: find_course — документ по названию, get_course — \
данные документа (лучше с fields), get_findings и get_rule — замечание и правило с цитатой \
норматива.
- Путь к значению (path) записывается через «/» по данным get_course: «credits», \
«workload_table/2/credits», «practice/type», «goals». Номера элементов списков — с нуля.
- Вид правки (op): set — заменить значение; remove — удалить элемент списка (path — путь к \
элементу, «competences/5», value не нужен); append — добавить элемент в конец списка (path — \
путь к списку). Не переписывай список целиком, чтобы удалить или добавить один элемент.
- Чтобы изменить поле элемента списка, укажи путь до поля: «competences/0/code», а не весь \
элемент «competences/0» — иначе пропадут остальные поля и источник значения. Замену \
значения не делай удалением и добавлением: измени поле (set).
- Данные документов по замечаниям уже приведены в сообщении: не запрашивай их повторно.
- Строки учебного плана — get_plan_entries (фильтр по названию, блоку, профилю), матрицы \
компетенций — get_matrix_rows; не листай план целиком через get_course.
- Если документы расходятся, верно значение учебного плана: исправляй документ, который с \
ним расходится, а план не меняй. Если учебного плана нет, а верное значение из документов не \
следует, спроси пользователя (ask_user, варианты — значения из документов).
- Меняй только то, что нужно для замечаний цели; другие замечания и документы не трогай. \
Значения — того же типа, что и прежние.
- Правку предлагай вызовом propose_change. Код проверит черновик правилами и вернёт \
проблемы, если исправляемое замечание осталось или появились новые; тогда исправь правку.
- Если цель неоднозначна или в документах не хватает данных для правки (например, неизвестно \
новое значение), спроси пользователя: ask_user с вопросом и вариантами ответа. Ответ придёт \
результатом вызова с его answer_id. Если вместо ответа пользователь спросил о чём-то, ответь \
ему в comment следующего ask_user и спроси снова. Если код не принял правку по ответу \
пользователя, объясни в comment, почему.
- Новый текст целей, миссии, аннотации и результатов обучения не придумывай: он берётся из \
ответа пользователя. Если пользователь поручает составить текст тебе, подготовь его для этой \
дисциплины и предложи пользователю: ask_user, текст целиком — в draft (кнопку «Принять \
текст» добавит код). В правке укажи answer_id ответа, в \
котором пользователь дал или принял этот текст; текст правки — ровно этот ответ.
- Новый код компетенции — из строк учебного плана этой дисциплины или код, формулировка \
которого во ФГОС совпадает с формулировкой компетенции в документе. Иначе спроси код у \
пользователя (ask_user) и укажи answer_id ответа. Компетенцию не удаляй, если пользователь \
не согласился на это в ответе ask_user (укажи его answer_id).
- Пользователь видит черновик с нумерованными правками и кнопками «Применить» и \
«Отклонить». Если вместо кнопки он написал свой текст, реши по смыслу:
  - вопрос о черновике — explain_draft (коротко, по данным черновика и замечаний); если \
пользователь снова спрашивает о том, что ты уже объяснил, не повторяй объяснение: дай \
подробности из данных документов, которых ещё не было в ответе;
  - просьба не вносить часть правок — drop_edits с номерами этих правок;
  - просьба изменить правку — новый propose_change; замечания цели, которые пользователь \
просит не исправлять, — в skipped_finding_ids;
  - отказ от всего черновика — reject_draft.
- В текстах для пользователя (question, comment, explain_draft) не называй служебные ID \
(course_id, id замечаний и правил, answer_id) и поля данных (path): документы — по \
названию, замечания и правила — по сути, значения — по разделам документа."""


class Edit(BaseModel):
    course_id: str = Field(description="ID документа")
    path: str = Field(description="путь к значению: «credits», «workload_table/2/credits»")
    op: Literal["set", "remove", "append"] = Field("set", description="вид правки")
    value: Any = Field(None, description="новое значение (для set и append)")
    answer_id: str | None = Field(
        None,
        description="answer_id ответа пользователя (ask_user), в котором он дал или одобрил "
        "это значение: для текста целей и результатов, нового кода, удаления компетенции",
    )


class ProposeChange(BaseModel):
    """Подготовка правок: значение в данных и новое значение. Код проверяет черновик и
    возвращает перечень проблем, если правка не годится."""

    edits: list[Edit] = Field(min_length=1)
    rationale: str = Field(description="почему так, коротко, по-русски")
    finding_ids: list[str] = Field(default_factory=list, description="исправляемые замечания")
    skipped_finding_ids: list[str] = Field(
        default_factory=list,
        description="замечания цели, которые пользователь просит не исправлять",
    )


class DropEdits(BaseModel):
    """Убрать из черновика правки, которые пользователь просит не вносить; остальные правки
    остаются. Замечания, которые после этого не исправляются, остаются открытыми."""

    numbers: list[int] = Field(min_length=1, description="номера правок в черновике, с 1")
    reason: str = Field(description="почему, коротко, по-русски")


class ExplainDraft(BaseModel):
    """Ответ на вопрос пользователя о черновике: черновик остаётся, пользователь снова
    выбирает «Применить» или «Отклонить»."""

    answer: str = Field(description="ответ по-русски, 1–3 предложения, по данным черновика")


class RejectDraft(BaseModel):
    """Пользователь отказался от всего черновика: изменение не вносится."""

    reason: str = Field(description="почему, коротко")


FINALS: dict[str, type[BaseModel]] = {
    "propose_change": ProposeChange,
    "drop_edits": DropEdits,
    "explain_draft": ExplainDraft,
    "reject_draft": RejectDraft,
}


class GenInput(TypedDict):
    run_id: str
    message: str
    kind: RequestKind | None
    scope: Scope | None
    finding_ids: list[str]
    change_ids: list[str]
    direction_code: str | None
    check_summary: CheckSummary | None
    messages: Annotated[list[AnyMessage], add_messages]  # диалог сессии
    summary: str | None
    work: list[AnyMessage]
    work_agent: str | None
    change_id: str | None
    run_change_ids: list[str]
    fix_queue: list[list[str]] | None
    fix_targets: list[str]


class GenOutput(TypedDict):
    change_ids: list[str]
    change_id: str | None
    pending: PendingQuestion | None
    work: list[AnyMessage]
    work_agent: str | None
    change_explanation: str | None
    run_change_ids: list[str]
    fix_queue: list[list[str]] | None
    fix_targets: list[str]


class GenState(GenInput, GenOutput):
    goal: str | None
    target_ids: list[str]


def _goal(state: GenState, targets: list[Finding]) -> str:
    if targets:
        return "Исправить замечания:\n" + "\n".join(
            f"- {f.id}: {f.rule_title} ({f.course or 'программа'}): {f.message}" for f in targets
        )
    return (state["scope"] or empty_scope())["change_goal"] or state["message"]


def fix_groups(findings: list[Finding]) -> list[list[Finding]]:
    """Исправимые замечания по документам, документы с большим числом ошибок первыми;
    замечания уровня программы — в конце: их исправление чаще требует решений пользователя."""
    groups = by_document(findings)
    program = groups.pop(PROGRAM, None)
    return [*groups.values(), *([program] if program else [])]


def build_generation_subgraph(deps: GraphDeps) -> CompiledStateGraph:
    async def targets(state: GenState, config: RunnableConfig) -> dict:
        session_id = session_id_of(config)
        ids = list((state["scope"] or empty_scope())["finding_ids"])
        if not ids and state["kind"] == "check_and_fix" and state["check_summary"]:
            ids = state["check_summary"]["finding_ids"]
        findings = await deps.stores.get_findings(session_id, ids)
        fixable = [f for f in findings if f.status == "violation" and f.fixable]
        if state["kind"] != "check_and_fix":
            return {"goal": _goal(state, fixable), "target_ids": [f.id for f in fixable]}
        # проверка с исправлением: одно изменение на документ, каждое со своим подтверждением;
        # документ остаётся текущим, пока граф не решит его изменение (очищает fix_targets)
        queue = state["fix_queue"]
        if queue is None:
            queue = [[f.id for f in group] for group in fix_groups(fixable)]
        if state["fix_targets"]:
            current = state["fix_targets"]
        elif queue:
            current, queue = queue[0], queue[1:]
        else:
            return {"goal": None, "target_ids": [], "fix_queue": [], "fix_targets": []}
        return {
            "goal": _goal(state, await deps.stores.get_findings(session_id, current)),
            "target_ids": current,
            "fix_queue": queue,
            "fix_targets": current,
        }

    async def generator(state: GenState, config: RunnableConfig) -> dict:
        if state["goal"] is None:
            return {**_unchanged(state), "work": [], "work_agent": None}
        session_id = session_id_of(config)
        documents = await deps.stores.list_documents(session_id)
        rules = select_rules(deps.rules_dir, state["direction_code"], empty_scope())
        # генератор видит только замечания своей цели: другие он не исправляет
        tools = SessionTools(
            deps.stores,
            session_id,
            rules,
            state["target_ids"],
            state["change_ids"],
            state["direction_code"],
        )
        draft = await deps.stores.get_change(session_id, state["change_id"] or "")
        draft = draft if draft is not None and draft.status == "draft" else None
        if state["work_agent"] == "generator" and state["work"]:
            work = list(state["work"])
        else:
            work = await _first_work(deps, session_id, state, tools, rules, documents)
        # у каждой доработки свой ID: работа растёт с каждым ответом пользователя
        key = f"{state['run_id']}\0{state['goal']}\0{len(work)}"
        change_id = sha256(key.encode()).hexdigest()[:12]
        prepared: dict[str, Any] = {}

        async def validate(name: str, result: BaseModel, history: list[BaseMessage]) -> list[str]:
            problems, ready = await _prepare(
                deps, session_id, state, rules, documents, draft, result, history
            )
            prepared.update(ready)
            return problems

        try:
            turn = await run_agent_turn(
                deps.models("generator"),
                system=GENERATOR_PROMPT,
                messages=work,
                tools=tools.as_tools(_tools_for(state["target_ids"])),
                finals=FINALS,
                validate=validate,
                max_steps=16,
                ask=AskUser,
            )
        except (AgentLimitError, openai.APIError) as error:  # LLM недоступна после повторов
            return await _failed(deps, session_id, state, change_id, draft, error)
        return await _outcome(deps, session_id, state, turn, draft, change_id, prepared)

    graph = StateGraph(GenState, input_schema=GenInput, output_schema=GenOutput)
    graph.add_node("targets", targets)
    graph.add_node("generator", generator)
    graph.add_edge(START, "targets")
    graph.add_edge("targets", "generator")
    graph.add_edge("generator", END)
    return graph.compile()


def _user_texts(result: BaseModel) -> list[str]:
    """Тексты итога генератора, которые увидит пользователь (кроме текста для документа)."""
    if isinstance(result, AskUser):
        return [result.question, result.comment or "", *result.options]
    if isinstance(result, ExplainDraft):
        return [result.answer]
    if isinstance(result, ProposeChange):
        return [result.rationale]
    if isinstance(result, DropEdits | RejectDraft):
        return [result.reason]
    return []


def _tools_for(target_ids: list[str]) -> list[str]:
    """Инструменты генератора; без замечаний цели (изменение по запросу) get_findings и
    get_rule всегда пусты, и модель перебирает их фильтры вместо работы."""
    if target_ids:
        return GENERATOR_TOOLS
    return [t for t in GENERATOR_TOOLS if t not in ("get_findings", "get_rule")]


async def _first_work(
    deps: GraphDeps,
    session_id: str,
    state: GenState,
    tools: SessionTools,
    rules: list[Rule],
    documents: list[StoredDocument],
) -> list[BaseMessage]:
    """Начало работы генератора: диалог сессии, цель и данные документов замечаний."""
    targets = await deps.stores.get_findings(session_id, state["target_ids"])
    request = [f"(Система) Цель: {state['goal']}"]
    if not targets:
        request.append("Замечаний у цели нет: изменение по запросу пользователя.")
    request += await _target_data(tools, rules, targets, documents)
    declined = await _declined(deps, session_id, state["change_ids"], targets)
    if declined:
        request.append(
            "Пользователь раньше просил не исправлять замечания: "
            + "; ".join(f"{f.id} ({f.rule_title})" for f in declined)
            + ". Если в последнем сообщении он не просит исправить их снова, не исправляй их и "
            "укажи в skipped_finding_ids."
        )
    return [*dialogue(state["messages"], state["summary"]), HumanMessage("\n".join(request))]


async def _declined(
    deps: GraphDeps, session_id: str, change_ids: list[str], targets: list[Finding]
) -> list[Finding]:
    """Замечания цели, которые пользователь уже просил не исправлять в этой сессии."""
    skipped: set[str] = set()
    for change_id in change_ids:
        change = await deps.stores.get_change(session_id, change_id)
        if change is not None and change.status != "superseded":
            skipped.update(change.skipped_finding_ids)
    return [f for f in targets if f.id in skipped]


async def _prepare(
    deps: GraphDeps,
    session_id: str,
    state: GenState,
    rules: list[Rule],
    documents: list[StoredDocument],
    draft: Change | None,
    result: BaseModel,
    history: list[BaseMessage],
) -> tuple[list[str], dict[str, Any]]:
    """Проверка итога генератора кодом: проблемы для агента или поля нового черновика."""
    if leaked := internal_names(_user_texts(result), rules, documents):
        return [
            "в тексте для пользователя служебные имена: " + ", ".join(leaked) + ". Пиши без "
            "них: правило — по сути, значение — по разделу документа и названию по-русски"
        ], {}
    if isinstance(result, AskUser):
        return [], {}
    if not isinstance(result, ProposeChange) and draft is None:
        return [
            "черновика ещё нет: на реплику пользователя ответь в comment вызова ask_user или "
            "предложи правку (propose_change)"
        ], {}
    if isinstance(result, ExplainDraft | RejectDraft):
        return [], {}
    if isinstance(result, DropEdits):
        assert draft is not None
        return await _drop(deps, session_id, state["run_id"], rules, documents, draft, result)
    assert isinstance(result, ProposeChange)
    by_id = {d.document_id: d for d in documents}
    edits = _change_edits(result.edits, by_id)
    if isinstance(edits, str):
        return [edits], {}
    answer_ids = [e.answer_id for e in result.edits]
    if problems := unsupported_edits(edits, answer_ids, by_id, rules, history):
        return problems, {}
    paths = user_paths(edits, answer_ids, history)
    targets = list(dict.fromkeys([*state["target_ids"], *result.finding_ids]))
    skipped = result.skipped_finding_ids
    review = await review_draft(
        deps,
        session_id,
        state["run_id"],
        rules,
        documents,
        edits,
        targets,
        optional=skipped,
        user_paths=paths,
    )
    if review.problems:
        return review.problems, {}
    return [], {
        "edits": edits,
        "rationale": result.rationale,
        "finding_ids": [t for t in targets if t not in skipped],
        "skipped_finding_ids": skipped,
        "consequences": review.consequences,
        "user_paths": paths,
    }


async def _drop(
    deps: GraphDeps,
    session_id: str,
    run_id: str,
    rules: list[Rule],
    documents: list[StoredDocument],
    draft: Change,
    result: DropEdits,
) -> tuple[list[str], dict[str, Any]]:
    """Черновик без правок, которые пользователь просит не вносить."""
    numbers = set(result.numbers)
    if not numbers <= set(range(1, len(draft.edits) + 1)):
        return [f"в черновике правки с номерами 1–{len(draft.edits)}"], {}
    edits = [e for i, e in enumerate(draft.edits, start=1) if i not in numbers]
    if not edits:
        return ["убраны все правки — это отказ от черновика: reject_draft"], {}
    kept = {e.path for e in edits}
    paths = [p for p in draft.user_paths if p in kept]
    review = await review_draft(
        deps,
        session_id,
        run_id,
        rules,
        documents,
        edits,
        draft.finding_ids,
        optional=draft.finding_ids,
        user_paths=paths,
    )
    if review.problems:
        return review.problems, {}
    left = await deps.stores.get_findings(session_id, review.remaining)
    dropped = ", ".join(map(str, sorted(numbers)))
    return [], {
        "edits": edits,
        "rationale": f"{draft.rationale.rstrip('. ')}. Без правок {dropped}: {result.reason}",
        "finding_ids": [i for i in draft.finding_ids if i not in review.remaining],
        "skipped_finding_ids": [*draft.skipped_finding_ids, *review.remaining],
        "consequences": review.consequences
        + [f"останется замечание «{f.rule_title}» ({f.course or 'программа'})" for f in left],
        "user_paths": paths,
    }


def _change_edits(raw: list[Edit], documents: dict[str, StoredDocument]) -> list[ChangeEdit] | str:
    """Правки агента с прежними значениями из данных или ошибка пути."""
    edits = []
    for e in raw:
        document = documents.get(e.course_id)
        if document is None:
            return f"нет документа {e.course_id}"
        try:
            old = get_value(document.data.model_dump(mode="json"), e.path)
        except (KeyError, IndexError, TypeError):
            return f"{document.filename}: нет значения по пути «{e.path}»"
        edits.append(
            ChangeEdit(
                document_id=e.course_id,
                path=e.path,
                op=e.op,
                old=None if e.op == "append" else old,
                new=None if e.op == "remove" else e.value,
            )
        )
    return edits


async def _outcome(
    deps: GraphDeps,
    session_id: str,
    state: GenState,
    turn: AgentTurn,
    draft: Change | None,
    change_id: str,
    prepared: dict[str, Any],
) -> dict:
    """Выход подграфа по итогу генератора: вопрос, пояснение, отказ или новый черновик.
    Работа генератора остаётся, пока черновик или вопрос ждут ответа пользователя."""
    if turn.ask is not None:
        pending = question(
            "missing_data",
            turn.ask.text(),
            "generate",
            options=turn.ask.choices(),
            call_id=turn.call_id,
            draft=turn.ask.draft if isinstance(turn.ask, AskUser) else None,
        )
        return {**_unchanged(state), "pending": pending, "work": turn.messages}
    if isinstance(turn.result, ExplainDraft):
        return {
            **_unchanged(state),
            "work": turn.messages,
            "change_explanation": turn.result.answer,
        }
    if isinstance(turn.result, RejectDraft):
        assert draft is not None
        await deps.stores.save_change(draft.model_copy(update={"status": "rejected"}))
        return {**_unchanged(state), "work": [], "work_agent": None, "fix_targets": []}
    await _supersede(deps, draft)  # новый черновик заменяет прежний
    change = Change(
        id=change_id,
        session_id=session_id,
        run_id=state["run_id"],
        goal=state["goal"] or "",
        **prepared,
    )
    await deps.stores.save_change(change)
    return {**_with_change(state, change_id), "work": turn.messages}


async def _failed(
    deps: GraphDeps,
    session_id: str,
    state: GenState,
    change_id: str,
    draft: Change | None,
    error: Exception,
) -> dict:
    """Генератор не дал итог: изменение со статусом failed, документ решён."""
    change = Change(
        id=change_id,
        session_id=session_id,
        run_id=state["run_id"],
        goal=state["goal"] or "",
        edits=[],
        status="failed",
        problems=[f"{user_error(error)}, повторите запрос"],
    )
    await deps.stores.save_change(change)
    await _supersede(deps, draft)  # прежний черновик заменён неудачей: не показывать его
    return {**_with_change(state, change_id), "fix_targets": [], "work": [], "work_agent": None}


async def _supersede(deps: GraphDeps, draft: Change | None) -> None:
    if draft is not None:
        await deps.stores.save_change(draft.model_copy(update={"status": "superseded"}))


def _unchanged(state: GenState) -> dict:
    """Выход подграфа без нового изменения."""
    return {
        "change_id": state["change_id"],
        "change_ids": state["change_ids"],
        "run_change_ids": state["run_change_ids"],
        "pending": None,
        "work": state["work"],
        "work_agent": "generator",
        "change_explanation": None,
        "fix_queue": state["fix_queue"],
        "fix_targets": state["fix_targets"],
    }


def _with_change(state: GenState, change_id: str) -> dict:
    def added(ids: list[str]) -> list[str]:
        return ids if change_id in ids else [*ids, change_id]

    return {
        **_unchanged(state),
        "change_id": change_id,
        "change_ids": added(state["change_ids"]),
        "run_change_ids": added(state["run_change_ids"]),
    }


async def _target_data(
    tools: SessionTools,
    rules: list[Rule],
    targets: list[Finding],
    documents: list[StoredDocument],
) -> list[str]:
    """Поля документов, которые читают правила замечаний: генератор правит по ним сразу."""
    by_id = {r.id: r for r in rules}
    fields: dict[str, set[str]] = {}
    for f in targets:
        rule = by_id.get(f.rule_id)
        if f.document_id is None or rule is None:
            continue
        used = rule.fields if rule.method == "llm" else CHECK_FIELDS.get(rule.check or "", ())
        fields.setdefault(f.document_id, set()).update(used)
    lines = []
    if any(f.document_id is None for f in targets):
        lines.append(_checked_documents(documents))
    for doc_id, names in fields.items():
        if not names:
            continue
        try:
            data = await tools.get_course(doc_id, sorted(names))
        except ValueError:  # поле правила не из модели этого документа
            continue
        lines.append(f"Данные документа {doc_id} (get_course, поля правил): {to_json(data)}")
    return lines


def _checked_documents(documents: list[StoredDocument]) -> str:
    """Документы программы, к которым относятся замечания уровня программы: при нескольких
    одного вида проверялся последний загруженный."""
    last = {d.doc_type: d for d in documents if d.doc_type != "rpd"}
    listed = ", ".join(f"{d.filename} (course_id {d.document_id})" for d in last.values())
    return f"Замечания уровня программы относятся к документам: {listed}."
