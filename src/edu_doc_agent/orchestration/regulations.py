"""Поиск по нормативам для Q&A-агента: инструменты RAG-компонента (`retrieval/`, ADR-0005).

RAG отдаёт инструменты обычными async-функциями (`RegulationTools.as_functions()`), список
документов программы задаётся кодом по коду направления, а не LLM.
"""

import functools
import logging
from collections.abc import Awaitable, Callable
from importlib.util import find_spec
from typing import Any

from edu_doc_agent.orchestration.deps import RegulationTools

logger = logging.getLogger(__name__)


def _logged(name: str, function: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Пишет в лог запрос и ID найденных пунктов: по логу видно, что нашёл поиск."""

    @functools.wraps(function)
    async def call(*args: Any, **kwargs: Any) -> Any:
        result = await function(*args, **kwargs)
        hits = result if isinstance(result, list) else [result] if result else []
        found = [getattr(h, "id", h) for h in hits] or "ничего"
        logger.info("RAG %s %s: %s", name, kwargs or args, found)
        return result

    return call


def retrieval_tools() -> RegulationTools | None:
    """Инструменты RAG на общих настройках или None, если RAG-компонента ещё нет в сборке."""
    if find_spec("edu_doc_agent.retrieval.tools") is None:
        return None
    from edu_doc_agent.retrieval.tools import RegulationTools as Tools
    from edu_doc_agent.retrieval.tools import create_regulation_search

    search = None

    async def tools(direction_code: str | None) -> dict[str, Callable[..., Awaitable[Any]]]:
        nonlocal search
        search = search or create_regulation_search()
        if direction_code:
            functions = (await Tools.for_program(search, direction_code)).as_functions()
        else:
            functions = Tools(search).as_functions()
        return {name: _logged(name, f) for name, f in functions.items()}

    return tools
