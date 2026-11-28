"""RAG-подсистема: корпус нормативов, поиск по нему и инструменты для агентов (ADR-0005).

Для графа: `create_regulation_search()` и `RegulationTools` (retrieval/tools.py),
для тестов графа без базы и модели — `InMemoryRegulationSearch` (retrieval/fake.py).
Импорт ленивый: сборке корпуса (retrieval.corpus) не нужны база и клиент LLM.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "RegulationHit": "edu_doc_agent.retrieval.search",
    "RegulationSearch": "edu_doc_agent.retrieval.search",
    "RegulationTools": "edu_doc_agent.retrieval.tools",
    "SearchBackend": "edu_doc_agent.retrieval.tools",
    "create_regulation_search": "edu_doc_agent.retrieval.tools",
    "InMemoryRegulationSearch": "edu_doc_agent.retrieval.fake",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_EXPORTS[name]), name)
