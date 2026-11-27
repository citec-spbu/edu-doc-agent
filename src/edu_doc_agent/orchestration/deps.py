"""Модели и службы, которые использует граф. В тестах подменяются."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from edu_doc_agent.dao.storage import FileStorage
from edu_doc_agent.orchestration.ingest import DocumentIngestor, ExtractionIngestor
from edu_doc_agent.orchestration.stores import SessionStores

RULES_DIR = Path("rules")

# роль агента → чат-модель: orchestrator, checker, reviewer, generator, qa, reporter
ModelFactory = Callable[[str], BaseChatModel]
# код направления (или None) → инструменты поиска по нормативам по именам ADR-0003
RegulationTools = Callable[[str | None], Awaitable[dict[str, Callable[..., Awaitable[Any]]]]]


@dataclass
class GraphDeps:
    models: ModelFactory
    store: BaseStore = field(default_factory=InMemoryStore)
    ingestor: DocumentIngestor = field(default_factory=ExtractionIngestor)
    rules_dir: Path = RULES_DIR
    regulations: RegulationTools | None = None
    # файлы ответа (протокол проверки); None — файлы не сохраняются, итог только в чате
    storage: FileStorage | None = None
    # документы программы по сессиям: кэш процесса, его сбрасывает запись документов
    cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def stores(self) -> SessionStores:
        return SessionStores(self.store)


def default_deps(store: BaseStore | None = None) -> GraphDeps:
    """Модели из настроек `llm/`, поиск по нормативам из `retrieval/`, если он доступен."""
    from edu_doc_agent.config import get_settings
    from edu_doc_agent.dao.storage import LocalFileStorage
    from edu_doc_agent.llm import get_chat_model
    from edu_doc_agent.orchestration.regulations import retrieval_tools

    return GraphDeps(
        models=get_chat_model,
        store=store or InMemoryStore(),
        regulations=retrieval_tools(),
        storage=LocalFileStorage(get_settings().storage_dir),
    )
