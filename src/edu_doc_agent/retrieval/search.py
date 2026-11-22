"""Поиск по нормативам для агентов (ADR-0005, решение 3; инструменты — ADR-0003).

Порядок:

1. В запросе есть код компетенции / профстандарта или «п. 2.3» — точный поиск
   (`method="exact"`). Если он что-то нашёл, это и есть ответ.
2. Иначе гибрид: полнотекстовый поиск и векторный (BGE-M3) сливаются через RRF
   (`method="hybrid"`). Если модель эмбеддингов недоступна — только полнотекстовый
   (`method="fts"`). Такие результаты не подтверждают, что требование к коду существует.

Разделённый на части пункт возвращается целиком; редакция результата — та, что в базе.
Список документов (`doc_ids`) задаёт вызывающий код по данным программы, а не LLM.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from edu_doc_agent.dao.regulations import ClauseKey, RegulationDAO
from edu_doc_agent.llm.embeddings import Embedder
from edu_doc_agent.llm.openai_compat import LLMError
from edu_doc_agent.retrieval.ranking import query_keys, rrf

logger = logging.getLogger(__name__)

Method = Literal["exact", "hybrid", "fts", "vector"]
# auto — как описано выше; fts / vector — один способ после точного (для оценки качества)
Mode = Literal["auto", "fts", "vector"]
CANDIDATES = 50  # сколько кандидатов берёт каждый способ поиска до слияния


class RegulationHit(BaseModel):
    """Результат поиска: пункт норматива дословно, со ссылкой для цитирования."""

    id: str  # «fgos-02.03.02:2.3» — вместе с edition ссылка для get_*_clause
    edition: str
    doc_id: str
    doc_type: str
    kind: str  # clause / competency / prof_standard_ref
    path: str
    text: str  # дословно; у пункта из частей — склейка всех частей
    context: str = ""  # вводные строки родительских пунктов: без них пункт перечня непонятен
    category: str | None = None
    rank: int  # место в выдаче, с 1
    method: Method


class RegulationSearch:
    """Поиск и чтение пунктов нормативов. Сам ничего не пишет."""

    def __init__(self, dao: RegulationDAO, embedder: Embedder | None = None) -> None:
        self._dao = dao
        self._embedder = embedder

    async def search(
        self,
        query: str,
        *,
        doc_ids: Sequence[str] | None = None,
        doc_types: Sequence[str] | None = None,
        top_k: int = 5,
        mode: Mode = "auto",
    ) -> list[RegulationHit]:
        filters = {"doc_ids": doc_ids, "doc_types": doc_types, "limit": CANDIDATES}
        keys = query_keys(query)
        if keys:
            exact = [
                *(await self._dao.find_by_codes(keys.codes, **filters) if keys.codes else []),
                *(await self._dao.find_by_numbers(keys.numbers, **filters) if keys.numbers else []),
            ]
            if exact:
                return await self._hits(list(dict.fromkeys(exact))[:top_k], "exact")

        if mode == "vector":
            vector = await self._vector(query, filters)
            return await self._hits(list(dict.fromkeys(vector or []))[:top_k], "vector")
        full_text = await self._dao.full_text(query, **filters)
        vector = await self._vector(query, filters) if mode == "auto" else None
        if vector is None:
            return await self._hits(list(dict.fromkeys(full_text))[:top_k], "fts")
        return await self._hits(rrf([full_text, vector])[:top_k], "hybrid")

    async def get_clause(self, id: str, edition: str) -> RegulationHit | None:
        """Пункт целиком по ссылке из результата поиска или из правила каталога."""
        hits = await self._hits([(id, edition)], "exact")
        return hits[0] if hits else None

    async def documents_for_direction(self, direction_code: str) -> list[str]:
        """Документы, применимые к программе направления: её ФГОС и нормы СПбГУ."""
        documents = await self._dao.list_documents()
        return [
            d["doc_id"]
            for d in documents
            if d["direction_code"] == direction_code or d["doc_type"] == "spbu_standard"
        ]

    # обёртки под инструменты агентов (ADR-0003)

    async def search_fgos(
        self, query: str, *, doc_ids: Sequence[str] | None = None, top_k: int = 5
    ) -> list[RegulationHit]:
        return await self.search(query, doc_ids=doc_ids, doc_types=["fgos"], top_k=top_k)

    async def search_spbu_regulations(
        self, query: str, *, doc_ids: Sequence[str] | None = None, top_k: int = 5
    ) -> list[RegulationHit]:
        return await self.search(query, doc_ids=doc_ids, doc_types=["spbu_standard"], top_k=top_k)

    async def search_prof_standards(
        self, query: str, *, doc_ids: Sequence[str] | None = None, top_k: int = 5
    ) -> list[RegulationHit]:
        return await self.search(query, doc_ids=doc_ids, doc_types=["prof_standard"], top_k=top_k)

    async def search_all_regulations(
        self, query: str, *, doc_ids: Sequence[str] | None = None, top_k: int = 5
    ) -> list[RegulationHit]:
        return await self.search(query, doc_ids=doc_ids, top_k=top_k)

    async def get_fgos_clause(self, id: str, edition: str) -> RegulationHit | None:
        return await self.get_clause(id, edition)

    async def get_spbu_clause(self, id: str, edition: str) -> RegulationHit | None:
        return await self.get_clause(id, edition)

    # --- внутреннее

    async def _vector(self, query: str, filters: dict) -> list[ClauseKey] | None:
        if self._embedder is None:
            return None
        try:
            (embedding,) = await asyncio.to_thread(self._embedder.embed, [query])
        except LLMError as error:
            logger.warning("эмбеддинги недоступны, только полнотекстовый поиск: %s", error)
            return None
        return await self._dao.nearest(embedding, **filters)

    async def _hits(self, keys: Sequence[ClauseKey], method: Method) -> list[RegulationHit]:
        parts = await self._dao.get_parts(keys)
        by_key: dict[ClauseKey, list[dict]] = {}
        for row in parts:
            by_key.setdefault((row["id"], row["edition"]), []).append(row)
        hits = []
        for key in keys:
            rows = by_key.get(key)
            if not rows:
                continue
            first = rows[0]
            hits.append(
                RegulationHit(
                    id=first["id"],
                    edition=first["edition"],
                    doc_id=first["doc_id"],
                    doc_type=first["doc_type"],
                    kind=first["kind"],
                    path=first["path"],
                    text="".join(r["text"] for r in rows),
                    context=first["context"],
                    category=first["category"],
                    rank=len(hits) + 1,
                    method=method,
                )
            )
        return hits


async def _cli(args) -> int:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from edu_doc_agent.config import get_settings
    from edu_doc_agent.llm.embeddings import OllamaEmbedder

    settings = get_settings()
    engine = create_async_engine(settings.database_connection_url, poolclass=NullPool)
    try:
        search = RegulationSearch(
            RegulationDAO(async_sessionmaker(engine, expire_on_commit=False)),
            None if args.no_embeddings else OllamaEmbedder.from_settings(settings),
        )
        doc_ids = args.doc_id or None
        if args.direction:
            doc_ids = [*(doc_ids or []), *await search.documents_for_direction(args.direction)]
        hits = await search.search(args.query, doc_ids=doc_ids, top_k=args.top_k)
    finally:
        await engine.dispose()
    for hit in hits:
        text = hit.text if args.full else hit.text[:300] + ("…" if len(hit.text) > 300 else "")
        print(f"{hit.rank}. [{hit.method}] {hit.id} ({hit.edition})\n   {hit.path}\n   {text}\n")
    if not hits:
        print("Ничего не найдено")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Поиск из терминала — проверить выдачу на загруженном корпусе:

    uv run python -m edu_doc_agent.retrieval.search "объем программы" --direction 02.03.02
    """
    import argparse

    parser = argparse.ArgumentParser(description="Поиск по нормативам в базе RAG")
    parser.add_argument("query")
    parser.add_argument("--direction", help="код направления: его ФГОС и нормы СПбГУ")
    parser.add_argument("--doc-id", action="append", help="документ для поиска (можно повторять)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="печатать текст пунктов целиком")
    parser.add_argument("--no-embeddings", action="store_true", help="только полнотекстовый")
    return asyncio.run(_cli(parser.parse_args(argv)))


if __name__ == "__main__":
    import sys

    sys.exit(main())
