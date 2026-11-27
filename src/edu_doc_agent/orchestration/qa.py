"""Подграф «Вопросы» (ADR-0003): Q&A-агент отвечает на вопросы о программе, замечаниях и
нормативных документах. Данные программы и замечания он получает инструментами сессии,
поиск по нормативам — инструментами RAG-компонента (`retrieval/`, ADR-0005). Агент только
читает данные.
"""

from typing import Annotated, TypedDict

import openai
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from edu_doc_agent.orchestration.agent import AgentLimitError, run_agent
from edu_doc_agent.orchestration.check import select_rules
from edu_doc_agent.orchestration.deps import GraphDeps
from edu_doc_agent.orchestration.history import dialogue
from edu_doc_agent.orchestration.program import direction_codes, session_id_of
from edu_doc_agent.orchestration.state import empty_scope
from edu_doc_agent.orchestration.stores import DOC_TYPE_TITLES, StoredDocument
from edu_doc_agent.orchestration.structured import user_error
from edu_doc_agent.orchestration.tools import (
    SessionTools,
    data_fields,
    document_title,
    function_tool,
)

SESSION_TOOLS = [
    "get_findings",
    "get_rule",
    "get_course",
    "find_course",
    "get_plan_entries",
    "get_plan_summary",
    "get_matrix_rows",
    "search_document",
    "get_document_outline",
    "read_section",
]
REGULATION_TOOLS = ["search_all_regulations", "get_fgos_clause", "get_spbu_clause"]

QA_PROMPT = """\
Ты Q&A-агент системы проверки документов образовательной программы СПбГУ. Отвечай на вопрос \
пользователя по-русски, коротко и по существу, только на основе данных из инструментов.

- Документы сессии с типом и полями извлечённых данных перечислены в конце этого \
сообщения. Вопрос о программе — по данным её документов; нормативы — для требований ФГОС \
и норм СПбГУ.
- Данные документа — get_course(course_id, fields): возьми поле, где ответ. У списков в \
скобках — подполя элементов: если нужны не все, запроси только их \
(fields=["список.подполе"]). Длинные списки фильтруй match (текст, который есть в нужных \
элементах) и листай offset, пока next_offset не станет null: не делай вывод по неполному \
списку. Строки учебного плана — get_plan_entries и \
get_plan_summary, матрицы компетенций — get_matrix_rows.
- Отвечай по тому документу, о котором вопрос, не подменяй его другим.
- Если ответ есть и в нормативах, и в документах программы и они расходятся, приведи оба и \
укажи расхождение.
- Текст документа (get_document_outline, read_section, search_document) — только если в \
извлечённых данных нужного нет.
- Замечания проверки: get_findings; правило, по которому найдено замечание, — get_rule по \
rule_id из замечания. Правила проверки — это то, что проверяет система, а не текст \
норматива: на вопросы о требованиях ФГОС и норм СПбГУ отвечай по нормативам.
- Нормативы (ФГОС, нормы СПбГУ): search_all_regulations, полный текст пункта — \
get_fgos_clause и get_spbu_clause по id и edition из результата поиска.
- Ищи в нормативах несколькими запросами разной формулировки (своими словами и терминами \
норматива) и прочитай найденные пункты целиком, включая таблицы, прежде чем ответить, что \
требования нет.
- Ссылайся на источник: документ и раздел или пункт норматива (id).
- Если в вопросе есть код (УК-1, 06.001), а поиск не нашёл его точно (method "exact"), \
похожие пункты не подтверждают, что требование к этому коду существует: скажи об этом.
- Если данных для ответа нет, так и скажи своими словами; не додумывай. Если документов \
программы нет, а вопрос о программе, ответь, что её данных нет, и предложи загрузить учебный \
план, РПД и общую характеристику.
- Вопрос пользователя — его последнее сообщение в диалоге; прежние реплики — контекст: \
короткое сообщение может уточнять прежний вопрос.
- В ответе не называй инструменты и служебные ID (course_id, id замечаний): документы — по \
названию и имени файла, пункты нормативов — номером пункта.
Итог — вызов QAAnswer."""


class QAAnswer(BaseModel):
    """Ответ пользователю."""

    answer: str = Field(description="ответ по-русски со ссылками на источники")


class QAInput(TypedDict):
    message: str
    finding_ids: list[str]
    change_ids: list[str]
    direction_code: str | None
    messages: Annotated[list[AnyMessage], add_messages]  # диалог сессии
    summary: str | None


class QAOutput(TypedDict):
    reply: str | None


class QAState(QAInput, QAOutput):
    pass


def _session_documents(documents: list[StoredDocument]) -> str:
    if not documents:
        return (
            "Документы сессии: не загружено ни одного документа. Данных программы "
            "(дисциплин, практик, компетенций) нет."
        )
    lines = ["Документы сессии (course_id — тип — название: поля):"]
    lines += [
        f"- {d.document_id} — {DOC_TYPE_TITLES.get(d.doc_type, d.doc_type)} — "
        f"{document_title(d)}: {', '.join(data_fields(d))}"
        for d in documents
    ]
    return "\n".join(lines)


def build_qa_subgraph(deps: GraphDeps) -> CompiledStateGraph:
    async def answer(state: QAState, config: RunnableConfig) -> dict:
        session_id = session_id_of(config)
        documents = await deps.stores.list_documents(session_id)
        code = state["direction_code"]
        if code is None:  # проверки ещё не было: код направления из документов
            codes = direction_codes(documents)
            code = codes[0] if len(codes) == 1 else None
        rules = select_rules(deps.rules_dir, code, empty_scope())
        session = SessionTools(
            deps.stores,
            session_id,
            rules,
            state["finding_ids"],
            state["change_ids"],
            code,
        )
        # без документов инструменты сессии бесполезны: только нормативы
        tools = session.as_tools(SESSION_TOOLS) if documents else []
        if deps.regulations is not None:
            functions = await deps.regulations(code or None)
            tools += [function_tool(functions[n], n) for n in REGULATION_TOOLS if n in functions]
        try:
            result = await run_agent(
                deps.models("qa"),
                system=f"{QA_PROMPT}\n\n{_session_documents(documents)}",
                messages=dialogue(state["messages"], state["summary"]),
                tools=tools,
                final=QAAnswer,
                max_steps=12,
            )
        except (AgentLimitError, openai.APIError) as error:  # LLM недоступна после повторов
            return {
                "reply": f"Не удалось подготовить ответ: {user_error(error)}. Повторите запрос."
            }
        return {"reply": result.answer}

    graph = StateGraph(QAState, input_schema=QAInput, output_schema=QAOutput)
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile()
