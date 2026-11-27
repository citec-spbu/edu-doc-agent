"""Общее для тестов графа: подмена моделей по ролям агентов, документы сессии, запуск."""

import asyncio
import itertools
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from edu_doc_agent.extraction.curriculum import CurriculumPlan, FinalAttestation, PlanEntry
from edu_doc_agent.extraction.source import SourceRef
from edu_doc_agent.llm import FakeChatModel
from edu_doc_agent.orchestration import GraphDeps, build_graph
from edu_doc_agent.orchestration.stores import StoredDocument

sys.path.insert(0, str(Path(__file__).parents[1] / "validation"))
from rpd_samples import row, rpd  # noqa: E402

RULES_DIR = Path(__file__).parents[3] / "rules"
SRC = SourceRef(file="plan.md", section="2", line=5)
_ids = itertools.count()

__all__ = ["row", "rpd"]


def call(name: str, **args: Any) -> AIMessage:
    """Ответ модели: вызов инструмента или схемы структурированного ответа."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{next(_ids)}"}])


def decision(**args: Any) -> AIMessage:
    return call("OrchestratorDecision", **args)


def ask(question: str, options: list[str] | None = None, **kw: Any) -> AIMessage:
    """Агент спрашивает пользователя (`ask_user`); у оркестратора — с `about`."""
    return call("ask_user", question=question, options=options or [], **kw)


def answer_id(message: AIMessage) -> str:
    return message.tool_calls[0]["id"]


class Models:
    """Модели по ролям агентов; у роли без ответов любой вызов падает."""

    def __init__(self, **replies: list[AIMessage]) -> None:
        self.by_role = {role: FakeChatModel(messages=iter(r)) for role, r in replies.items()}

    def __call__(self, role: str) -> FakeChatModel:
        return self.by_role.setdefault(role, FakeChatModel(messages=iter([])))

    def calls(self, role: str) -> list:
        return self(role).calls


class Ingestor:
    """Отдаёт заранее заданные данные документов по пути файла."""

    def __init__(self, documents: dict[str, tuple[str, BaseModel]], fail_times: int = 0) -> None:
        self.documents = documents
        self.paths: list[str] = []
        self.fail_times = fail_times

    async def ingest(self, session_id: str, path: str) -> StoredDocument:
        self.paths.append(path)
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("сбой извлечения")
        doc_type, data = self.documents[path]
        text = getattr(data, "goals", None) or ""
        name = Path(path).name  # ID по имени файла, как у ExtractionIngestor
        return StoredDocument(f"doc:{name}", session_id, doc_type, data, name, text, path)


def plan(*entries: PlanEntry, gia: float = 9) -> CurriculumPlan:
    return CurriculumPlan(
        file="plan.md",
        direction="02.03.02 Фундаментальная информатика и информационные технологии",
        entries=list(entries),
        final_attestation=[
            FinalAttestation(block="Блок.3.гиа", credits=gia, name="ВКР", source=SRC)
        ],
    )


def entry(name: str, code: str, credits: float, competences: list[str], **kw: Any) -> PlanEntry:
    return PlanEntry(
        block=kw.pop("block", "Блок.1.дисц"),
        code=code,
        name=name,
        credits=credits,
        competences=competences,
        semester=kw.pop("semester", 1),
        source=SRC,
        **kw,
    )


def make_graph(models: Models, documents: dict[str, tuple[str, BaseModel]] | None = None, **kw):
    deps = GraphDeps(
        models=models,
        store=InMemoryStore(),
        ingestor=kw.pop("ingestor", None) or Ingestor(documents or {}),
        rules_dir=kw.pop("rules_dir", RULES_DIR),
        **kw,
    )
    return build_graph(InMemorySaver(), deps), deps


def config(session: str = "s1") -> dict:
    return {"configurable": {"thread_id": session}}


def run(graph, value, session: str = "s1"):
    """Состояние после запуска и порядок пройденных узлов графа верхнего уровня."""

    async def scenario():
        nodes: list[str] = []
        async for update in graph.astream(value, config(session), stream_mode="updates"):
            nodes += [name for name in update if name != "__interrupt__"]
        return await graph.aget_state(config(session)), nodes

    return asyncio.run(scenario())


def findings(deps: GraphDeps, ids: list[str], session: str = "s1"):
    return asyncio.run(deps.stores.get_findings(session, ids))
