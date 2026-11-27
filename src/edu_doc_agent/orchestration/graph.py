"""Граф оркестрации (ADR-0003): точка входа для воркера.

    start -> [ingest] -> orchestrator --> qa -> END
                             |
                             +-> check -> [generate] -> reporter -> END
                             +-> generate -> confirm -> ask_user -> apply_change -> reporter
                                    ^                                    |
                                    +------------ доработка -------------+

Вопросы пользователю задают агенты — оркестратор (в том числе код направления) и генератор —
вызовом `ask_user`. Узел `ask_user` ставит граф на паузу (`interrupt()`); ответ возвращается
агенту результатом его вызова, и агент продолжает с прежней работой (`work`). LLM-вызовы
вынесены из `ask_user`, поэтому при возобновлении они не повторяются. Вопрос и ответ попадают
в диалог сессии (`messages`).

Подтверждение изменения — кнопки «Применить» и «Отклонить»: нажатую кнопку обрабатывает код,
свой текст пользователя получает генератор.
"""

import openai
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from edu_doc_agent.orchestration.agent import answer_message
from edu_doc_agent.orchestration.check import build_check_subgraph, fgos_codes, select_rules
from edu_doc_agent.orchestration.corrected import apply_draft
from edu_doc_agent.orchestration.deps import GraphDeps, default_deps
from edu_doc_agent.orchestration.edits import change_text
from edu_doc_agent.orchestration.generation import build_generation_subgraph
from edu_doc_agent.orchestration.history import question_turn, remember
from edu_doc_agent.orchestration.ingest import UnknownDocumentError, original_filename
from edu_doc_agent.orchestration.orchestrator import decide
from edu_doc_agent.orchestration.program import direction_codes, invalidate, session_id_of
from edu_doc_agent.orchestration.qa import build_qa_subgraph
from edu_doc_agent.orchestration.report import (
    change_report,
    check_report,
    reporter_message,
    save_protocol,
)
from edu_doc_agent.orchestration.state import (
    GraphInput,
    GraphOutput,
    GraphState,
    OutputFile,
    empty_scope,
    question,
)
from edu_doc_agent.orchestration.tools import SessionTools, document_title

_APPLY = "Применить"
_REJECT = "Отклонить"
_STOP = "Завершить исправления"  # проверка с исправлением: остальные документы не исправлять
_ASK_TARGETS = ("orchestrator", "generate", "apply_change")
# кто задал вопрос (узел, который получает ответ) — имя автора реплики в диалоге
_AUTHORS = {
    "orchestrator": "orchestrator",
    "generate": "generator",
    "apply_change": "generator",
}


async def _ask_user(state: GraphState) -> Command:
    """Пауза с вопросом агента или подтверждением; ответ — в диалог и агенту, который спросил."""
    pending = state["pending"]
    assert pending is not None
    answer = str(
        interrupt(
            {
                "type": pending["type"],
                "question": pending["question"],
                "options": pending["options"],
            }
        )
    )
    update: dict = {
        "messages": question_turn(
            pending["question"], pending["options"], answer, _AUTHORS[pending["resume_to"]]
        ),
        "answer": answer,
        "pending": None,
    }
    if pending["call_id"]:
        reply = answer_message(pending["call_id"], answer, pending["options"], pending.get("draft"))
        update["work"] = [*state["work"], reply]
    return Command(goto=pending["resume_to"], update=update)


def _next_document(state: GraphState, *, stop: bool = False, update: dict | None = None) -> Command:
    """После решения по изменению: следующий документ проверки с исправлением или отчёт."""
    update = {
        **(update or {}),
        "fix_targets": [],
        "change_id": None,
        "change_explanation": None,
        "work": [],
        "work_agent": None,
    }
    if stop:
        update["fix_queue"] = []
    more = state["kind"] == "check_and_fix" and state["fix_queue"] and not stop
    return Command(goto="generate" if more else "reporter", update=update)


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None, deps: GraphDeps | None = None
) -> CompiledStateGraph:
    deps = deps or default_deps()

    async def start(state: GraphState, config: RunnableConfig) -> dict:
        """Сброс полей запуска; данные сессии остаются."""
        invalidate(deps, session_id_of(config))
        return {
            "document_ids": state.get("document_ids", []),
            "ingested_paths": state.get("ingested_paths", []),
            "finding_ids": state.get("finding_ids", []),
            "change_ids": state.get("change_ids", []),
            "direction_code": state.get("direction_code"),
            "messages": [HumanMessage(state["message"])],  # запрос — в диалог сессии
            "summary": state.get("summary"),
            "kind": None,
            "scope": None,
            "pending": None,
            "answer": None,
            "work": [],
            "work_agent": None,
            "ingest_errors": [],
            "check_summary": None,
            "change_id": None,
            "change_explanation": None,
            "reply": None,
            "files": [],
            "run_change_ids": [],
            "fix_queue": None,
            "fix_targets": [],
        }

    async def ingest(state: GraphState, config: RunnableConfig) -> dict:
        session_id = session_id_of(config)
        document_ids = list(state["document_ids"])
        ingested = list(state["ingested_paths"])
        errors = []
        for path in state["file_paths"]:
            if path in ingested:
                continue
            try:
                document = await deps.ingestor.ingest(session_id, path)
            except (UnknownDocumentError, ValueError, OSError) as error:
                errors.append(str(error) or original_filename(path))
                continue
            await deps.stores.save_document(document)
            ingested.append(path)
            if document.document_id not in document_ids:
                document_ids.append(document.document_id)
        invalidate(deps, session_id)
        return {"document_ids": document_ids, "ingested_paths": ingested, "ingest_errors": errors}

    def after_start(state: GraphState) -> str:
        new = [p for p in state["file_paths"] if p not in state["ingested_paths"]]
        return "ingest" if new else "orchestrator"

    async def session_tools(state: GraphState, config: RunnableConfig) -> SessionTools:
        rules = select_rules(deps.rules_dir, state["direction_code"], empty_scope())
        return SessionTools(
            deps.stores,
            session_id_of(config),
            rules,
            state["finding_ids"],
            state["change_ids"],
            state["direction_code"],
        )

    async def orchestrator(state: GraphState, config: RunnableConfig) -> dict:
        tools = await session_tools(state, config)
        documents = await deps.stores.list_documents(session_id_of(config))
        titles = [
            d.data.title
            for d in documents
            if d.doc_type == "rpd" and getattr(d.data, "title", None)
        ]
        return await decide(
            deps.models("orchestrator"),
            state,
            tools,
            titles,
            direction_codes(documents),
            fgos_codes(deps.rules_dir),
        )

    async def confirm(state: GraphState, config: RunnableConfig) -> dict:
        session_id = session_id_of(config)
        change = await deps.stores.get_change(session_id, state["change_id"])
        assert change is not None
        documents = await deps.stores.list_documents(session_id)
        titles = {d.document_id: document_title(d) for d in documents}
        options = [_APPLY, _REJECT]
        explanation = state["change_explanation"]  # ответ генератора на вопрос пользователя
        progress = ""
        if state["kind"] == "check_and_fix" and state["fix_queue"]:
            options.append(_STOP)
            progress = f"\nПосле этого документа к исправлению ещё: {len(state['fix_queue'])}."
        text = (
            (f"{explanation}\n\n" if explanation else "")
            + change_text(change, titles)
            + progress
            + "\n\nПрименить изменение? Выберите вариант или напишите свой ответ: вопрос "
            "или что доработать."
        )
        return {
            "pending": question(
                "confirmation", text, "apply_change", options=options, change_id=change.id
            ),
            "change_explanation": None,
        }

    async def apply(state: GraphState, config: RunnableConfig) -> Command:
        session_id = session_id_of(config)
        change = await deps.stores.get_change(session_id, state["change_id"])
        assert change is not None
        answer = state["answer"]
        assert answer is not None
        if answer in (_REJECT, _STOP):
            await deps.stores.save_change(change.model_copy(update={"status": "rejected"}))
            return _next_document(state, stop=answer == _STOP)
        if answer != _APPLY:  # свой текст: решает генератор, который подготовил черновик
            note = HumanMessage(f"Ответ пользователя на черновик (не кнопка): {answer}")
            return Command(goto="generate", update={"work": [*state["work"], note]})
        finding_ids, files = await apply_draft(
            deps, session_id, state["run_id"], change, state["direction_code"], state["finding_ids"]
        )
        return _next_document(
            state, update={"finding_ids": finding_ids, "files": [*state["files"], *files]}
        )

    async def reporter(state: GraphState, config: RunnableConfig) -> dict:
        session_id = session_id_of(config)
        parts = []
        if state["ingest_errors"]:
            parts.append(
                "Не удалось обработать файлы:\n"
                + "\n".join(f"- {e}" for e in state["ingest_errors"])
            )
        files: list[OutputFile] = list(state["files"])  # исправленные документы
        if (summary := state["check_summary"]) is not None:
            findings = await deps.stores.get_findings(session_id, summary["finding_ids"])
            saved = await save_protocol(deps, session_id, state["run_id"], summary, findings)
            files += [saved] if saved else []
            parts.append(check_report(summary, findings, saved["filename"] if saved else None))
        if state["kind"] in ("change", "check_and_fix"):
            changes = [
                c
                for i in state["run_change_ids"]
                if (c := await deps.stores.get_change(session_id, i)) and c.status != "superseded"
            ]
            if not changes:
                parts.append("Изменений нет: нет исправимых нарушений или цели изменения.")
            else:
                documents = await deps.stores.list_documents(session_id)
                titles = {d.document_id: document_title(d) for d in documents}
                parts += [change_report(c, titles) for c in changes]
        protocol = "\n\n".join(parts)
        message = await reporter_message(deps.models("reporter"), protocol) if protocol else None
        return {"reply": "\n\n".join(p for p in (message, protocol) if p), "files": files}

    async def remember_turns(state: GraphState) -> dict:
        """Ответ запуска — в диалог сессии; ранние реплики сворачиваются в краткое содержание."""
        author = "qa_agent" if state["kind"] == "question" else "reporter"
        return await remember(
            deps.models("summarizer"), state["messages"], state["summary"], state["reply"], author
        )

    def after_orchestrator(state: GraphState) -> str:
        if state["pending"] is not None:
            return "ask_user"
        return {"question": "qa", "change": "generate"}.get(state["kind"], "check")

    def after_check(state: GraphState) -> str:
        return "generate" if state["kind"] == "check_and_fix" else "reporter"

    async def after_generate(state: GraphState, config: RunnableConfig) -> str:
        if state["pending"] is not None:
            return "ask_user"
        if state["change_id"]:
            change = await deps.stores.get_change(session_id_of(config), state["change_id"])
            if change is not None and change.status == "draft":
                return "confirm"
        # изменение документа не удалось: проверка с исправлением переходит к следующему
        if state["kind"] == "check_and_fix" and state["fix_queue"]:
            return "generate"
        return "reporter"

    graph = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphOutput)
    graph.add_node("start", start)
    graph.add_node("ingest", ingest)
    # временная ошибка LLM, не снятая повторами клиента (ADR-0003): узел повторяется целиком
    llm_retry = RetryPolicy(max_attempts=3, retry_on=openai.APIError)
    graph.add_node("orchestrator", orchestrator, retry_policy=llm_retry)
    graph.add_node("ask_user", _ask_user, destinations=_ASK_TARGETS)
    graph.add_node("check", build_check_subgraph(deps))
    graph.add_node("generate", build_generation_subgraph(deps))
    graph.add_node("confirm", confirm)
    graph.add_node("apply_change", apply, destinations=("generate", "reporter"))
    graph.add_node("qa", build_qa_subgraph(deps))
    graph.add_node("reporter", reporter, retry_policy=llm_retry)

    graph.add_edge(START, "start")
    graph.add_conditional_edges("start", after_start, ["ingest", "orchestrator"])
    graph.add_edge("ingest", "orchestrator")
    graph.add_conditional_edges(
        "orchestrator", after_orchestrator, ["ask_user", "qa", "check", "generate"]
    )
    graph.add_conditional_edges("check", after_check, ["generate", "reporter"])
    graph.add_conditional_edges(
        "generate", after_generate, ["ask_user", "confirm", "generate", "reporter"]
    )
    graph.add_edge("confirm", "ask_user")
    graph.add_node("remember", remember_turns)
    graph.add_edge("qa", "remember")
    graph.add_edge("reporter", "remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)
