"""Инструменты RAG для агентов (ADR-0003, «Инструменты»; ADR-0005) — без привязки к фреймворку.

Это контракт между RAG и графом: обычные async-функции с типизированными аргументами,
docstring — описание инструмента для LLM, результат — Pydantic-модели (сериализуются в JSON).
Граф оборачивает их в инструменты LangChain без изменений, например
`StructuredTool.from_function(coroutine=tools.search_fgos)`.

Как пользоваться из графа:

    search = create_regulation_search()                          # один раз на процесс
    tools = await RegulationTools.for_program(search, "02.03.02")  # на запуск: из ProgramInfo
    hits = await tools.search_all_regulations("какой объем программы?")

Список документов задаёт код по данным программы (`for_program`), а не LLM: тексты разных
ФГОС почти совпадают. Поэтому у инструментов поиска нет аргумента `doc_ids`.

Для тестов графа без базы и модели — `InMemoryRegulationSearch` (retrieval/fake.py) с тем
же интерфейсом.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from edu_doc_agent.retrieval.search import RegulationHit


class SearchBackend(Protocol):
    """Что нужно инструментам от поиска: реализуют RegulationSearch и InMemoryRegulationSearch."""

    async def search(
        self,
        query: str,
        *,
        doc_ids: Sequence[str] | None = None,
        doc_types: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
        top_k: int = 5,
    ) -> list[RegulationHit]: ...

    async def get_clause(self, id: str, edition: str) -> RegulationHit | None: ...

    async def documents_for_direction(self, direction_code: str) -> list[str]: ...


class RegulationTools:
    """Инструменты одного запуска графа: поиск ограничен документами программы."""

    def __init__(
        self, search: SearchBackend, doc_ids: Sequence[str] | None = None, top_k: int = 5
    ) -> None:
        self._search = search
        self.doc_ids = list(doc_ids) if doc_ids is not None else None
        self.top_k = top_k

    @classmethod
    async def for_program(
        cls, search: SearchBackend, direction_code: str, top_k: int = 5
    ) -> "RegulationTools":
        """Инструменты для программы направления: её ФГОС и нормы СПбГУ."""
        return cls(search, await search.documents_for_direction(direction_code), top_k)

    async def search_fgos(self, query: str, top_k: int | None = None) -> list[RegulationHit]:
        """Поиск по ФГОС программы. Возвращает пункты дословно со ссылкой для цитирования.

        query: вопрос или требование своими словами; код компетенции («УК-1») или номер
            пункта («п. 2.3») ищется точно.
        top_k: сколько пунктов вернуть (по умолчанию 5).

        У каждого результата: id и edition — ссылка для get_fgos_clause; path — где пункт в
        документе; text — дословный текст; context — вводные строки родительских пунктов;
        method — как найден. Если в запросе был код, а method не "exact", найденное лишь
        похоже на запрос и не подтверждает, что требование к этому коду существует.
        """
        return await self._find(query, top_k, doc_types=["fgos"])

    async def search_spbu_regulations(
        self, query: str, top_k: int | None = None
    ) -> list[RegulationHit]:
        """Поиск по нормам СПбГУ (Образовательный стандарт СПбГУ). Аргументы и результат —
        как у search_fgos; ссылка для чтения пункта — get_spbu_clause."""
        return await self._find(query, top_k, doc_types=["spbu_standard"])

    async def search_prof_standards(
        self, query: str, top_k: int | None = None
    ) -> list[RegulationHit]:
        """Поиск по профстандартам, соответствующим направлению программы: строки перечня
        из приложения ФГОС (код, название, приказ об утверждении). Тексты самих
        профстандартов в корпус пока не загружены. Аргументы и результат — как у search_fgos."""
        return await self._find(query, top_k, kinds=["prof_standard_ref"])

    async def search_all_regulations(
        self, query: str, top_k: int | None = None
    ) -> list[RegulationHit]:
        """Поиск по всем нормативам программы: ФГОС, нормы СПбГУ, профстандарты. У каждого
        результата doc_id и doc_type — из какого он документа. Аргументы — как у search_fgos."""
        return await self._find(query, top_k)

    async def get_fgos_clause(self, id: str, edition: str) -> RegulationHit | None:
        """Полный текст пункта ФГОС по ссылке (id и edition из результата поиска или из
        правила каталога). None — такого пункта в этой редакции нет."""
        return await self._search.get_clause(id, edition)

    async def get_spbu_clause(self, id: str, edition: str) -> RegulationHit | None:
        """Полный текст пункта норм СПбГУ по ссылке. None — такого пункта в этой редакции нет."""
        return await self._search.get_clause(id, edition)

    def as_functions(self) -> dict[str, Callable[..., Awaitable]]:
        """Инструменты по именам из ADR-0003 — для регистрации в графе."""
        return {
            name: getattr(self, name)
            for name in (
                "search_fgos",
                "search_spbu_regulations",
                "search_prof_standards",
                "search_all_regulations",
                "get_fgos_clause",
                "get_spbu_clause",
            )
        }

    async def _find(
        self,
        query: str,
        top_k: int | None,
        *,
        doc_types: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[RegulationHit]:
        return await self._search.search(
            query,
            doc_ids=self.doc_ids,
            doc_types=doc_types,
            kinds=kinds,
            top_k=top_k or self.top_k,
        )


def create_regulation_search() -> SearchBackend:
    """Поиск по нормативам на общих настройках: сессии БД из dao, эмбеддинги из llm."""
    from edu_doc_agent.dao.database import async_session_maker
    from edu_doc_agent.dao.regulations import RegulationDAO
    from edu_doc_agent.llm.embeddings import OllamaEmbedder
    from edu_doc_agent.retrieval.search import RegulationSearch

    return RegulationSearch(RegulationDAO(async_session_maker), OllamaEmbedder.from_settings())
