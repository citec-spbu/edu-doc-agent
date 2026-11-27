"""Запуск графа воркером (ADR-0003, «Интерфейс с воркером»).

    async with postgres_graph() as graph:
        turn = await run_turn(graph, session_id, run_id, message, file_paths=[...])

Состояние графа хранит `AsyncPostgresSaver`, данные сессии (документы, замечания,
изменения) — `AsyncPostgresStore`, обе в базе из настроек `dao`. `run_turn` сам выбирает вход:
ответ на вопрос графа (`Command(resume=...)`), повтор после сбоя (`None`) или новый запрос.
Граф для воркера — чёрный ящик: в `worker/tasks.py` достаточно вызвать `run_turn` вместо
заглушки бизнес-логики.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.types import Command
from sqlalchemy.engine import make_url

from edu_doc_agent.config import Settings, get_settings
from edu_doc_agent.orchestration.deps import default_deps
from edu_doc_agent.orchestration.graph import build_graph
from edu_doc_agent.orchestration.state import Action, OutputFile, make_input

MAX_CONCURRENCY = 8  # параллельных веток проверки (вызовов LLM) в одном запуске


def connection_string(settings: Settings | None = None) -> str:
    """Строка подключения psycopg из настроек `dao` (без драйвера SQLAlchemy)."""
    url = make_url((settings or get_settings()).database_connection_url)
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


@asynccontextmanager
async def postgres_graph(conn_string: str | None = None) -> AsyncIterator[CompiledStateGraph]:
    url = conn_string or connection_string()
    async with (
        AsyncPostgresSaver.from_conn_string(url) as checkpointer,
        AsyncPostgresStore.from_conn_string(url) as store,
    ):
        await checkpointer.setup()
        await store.setup()
        yield build_graph(checkpointer, default_deps(store))


@dataclass
class Turn:
    reply: str | None  # ответ пользователю, если запуск завершён
    question: dict[str, Any] | None  # вопрос графа: run переходит в waiting_for_input
    files: list[OutputFile] = field(default_factory=list)  # файлы ответа: протокол проверки


async def run_turn(
    graph: CompiledStateGraph,
    session_id: str,
    run_id: str,
    message: str,
    *,
    file_paths: list[str] | None = None,
    action: Action | None = None,
    retry: bool = False,
) -> Turn:
    """Один шаг диалога: новый запрос, ответ на вопрос графа или (`retry`) повтор того же
    запуска после сбоя воркера."""
    config = {"configurable": {"thread_id": session_id}, "max_concurrency": MAX_CONCURRENCY}
    state = await graph.aget_state(config)
    if state.interrupts:
        value: Any = Command(resume=message)
    elif retry and state.next:
        value = None  # запуск упал: продолжить с последнего завершённого шага
    else:
        value = make_input(run_id, message, action=action, file_paths=file_paths)
    result = await graph.ainvoke(value, config)
    interrupts = result.get("__interrupt__")
    if interrupts:
        return Turn(reply=None, question=interrupts[0].value)
    return Turn(reply=result.get("reply"), question=None, files=result.get("files") or [])
