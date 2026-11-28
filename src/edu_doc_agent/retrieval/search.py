"""Поиск по нормативам для агентов (ADR-0005, решение 3; инструменты — ADR-0003).

Порядок:

1. В запросе есть код компетенции / профстандарта или «п. 2.3» — точный поиск; найденное
   идёт первым (`method="exact"`). Остальные места добирает гибрид по всему запросу: на
   вопрос «Дисциплины, формирующие ОПК-2, должны быть в обязательной части?» ответ в
   пункте 2.9, а не в тексте ОПК-2. Индикатор («УК-1.2») не код: во ФГОС индикаторов нет.
2. Гибрид: векторный поиск (BGE-M3) и полнотекстовый со всеми словами запроса (И)
   сливаются взвешенным RRF (`method="hybrid"`). Полнотекстовый с И находит мало, но
   точно, и только поднимает пункты, где есть все слова запроса; порядок задаёт вектор.
   Если модель эмбеддингов недоступна — только полнотекстовый, со словами через ИЛИ
   (`method="fts"`): с И он почти ничего не находит.
   Такие результаты не подтверждают, что требование к коду существует.

У каждого результата `similarity` — косинусное сходство запроса с пунктом: по нему агент
отличает слабое совпадение от ответа (на вопрос, ответа на который в нормативах нет, поиск
всё равно вернёт top_k пунктов). Без модели эмбеддингов — None.

Разделённый на части пункт возвращается целиком. Перечень («2.2.4. рабочие программы…,
содержащие:» и подпункты 2.2.4.1–2.2.4.3) — одним результатом: найденный подпункт
заменяется вводным пунктом со всеми подпунктами (`covers` — их id). Редакция результата —
та, что в базе.
Список документов (`doc_ids`) задаёт вызывающий код по данным программы, а не LLM.
Инструменты для агентов поверх этого поиска — retrieval/tools.py.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from edu_doc_agent.dao.regulations import ClauseKey, RegulationDAO, TextMatch
from edu_doc_agent.llm.embeddings import Embedder
from edu_doc_agent.llm.openai_compat import LLMError
from edu_doc_agent.retrieval.ranking import collapse_lists, number_order, query_keys, rrf

logger = logging.getLogger(__name__)

Method = Literal["exact", "hybrid", "fts", "vector"]
# auto — как описано выше; fts / vector — один способ после точного (для оценки качества)
Mode = Literal["auto", "fts", "vector"]
CANDIDATES = 50  # сколько кандидатов берёт каждый способ поиска до слияния
# Как полнотекстовый поиск в гибриде сочетает слова запроса и вес его списка в RRF (у
# векторного — 1). На золотом наборе (34 вопроса, 8 направлений) правильный пункт первым /
# в первых 5: вектор 88% / 100%; гибрид с И, вес 0.25 — 91% / 100%, вес 1 — 88% / 100%;
# гибрид с ИЛИ, вес 0.25 — 79% / 97%, вес 1 — 62% / 94%; текст с ИЛИ — 35% / 59%.
# С ИЛИ под запрос подходит большая часть документа, и пункты из обоих списков обгоняют
# сильный результат одного вектора.
FTS_WEIGHT = 0.25
FTS_MATCH: TextMatch = "all"


class RegulationHit(BaseModel):
    """Результат поиска: пункт норматива дословно, со ссылкой для цитирования."""

    id: str  # «fgos-02.03.02:2.3» — вместе с edition ссылка для get_*_clause
    edition: str
    doc_id: str
    doc_type: str
    kind: str  # clause / competency / prof_standard_ref
    path: str
    text: str  # дословно; у пункта из частей — склейка частей, у перечня — пунктов по строкам
    context: str = ""  # вводные строки родительских пунктов: без них пункт перечня непонятен
    category: str | None = None
    rank: int  # место в выдаче, с 1
    method: Method
    covers: list[str] = []  # у перечня: id всех вошедших пунктов (вводный и подпункты)
    # косинусное сходство запроса с пунктом (у перечня — с лучшим из пунктов); None — без
    # модели эмбеддингов
    similarity: float | None = None


class RegulationSearch:
    """Поиск и чтение пунктов нормативов. Сам ничего не пишет."""

    def __init__(
        self,
        dao: RegulationDAO,
        embedder: Embedder | None = None,
        *,
        fts_weight: float = FTS_WEIGHT,
        fts_match: TextMatch = FTS_MATCH,
    ) -> None:
        self._dao = dao
        self._embedder = embedder
        self._fts_weight = fts_weight
        self._fts_match = fts_match

    async def search(
        self,
        query: str,
        *,
        doc_ids: Sequence[str] | None = None,
        doc_types: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
        top_k: int = 5,
        mode: Mode = "auto",
        fts_weight: float | None = None,
        fts_match: TextMatch | None = None,
    ) -> list[RegulationHit]:
        filters = {"doc_ids": doc_ids, "doc_types": doc_types, "kinds": kinds, "limit": CANDIDATES}
        keys = query_keys(query)
        exact: list[ClauseKey] = []
        if keys.codes:
            exact += await self._dao.find_by_codes(keys.codes, **filters)
        if keys.numbers:
            exact += await self._dao.find_by_numbers(keys.numbers, **filters)

        embedding = await self._embed(query) if mode != "fts" else None
        method: Method
        if mode == "vector":
            found = await self._dao.nearest(embedding, **filters) if embedding else []
            method = "vector"
        else:
            match = fts_match or (self._fts_match if embedding is not None else "any")
            full_text = await self._dao.full_text(query, match=match, **filters)
            if embedding is None:
                found, method = full_text, "fts"
            else:
                vector = await self._dao.nearest(embedding, **filters)
                weight = self._fts_weight if fts_weight is None else fts_weight
                found, method = rrf([full_text, vector], [weight, 1.0]), "hybrid"
        return await self._ranked(exact, found, method, top_k, embedding)

    async def get_clause(self, id: str, edition: str) -> RegulationHit | None:
        """Пункт по ссылке из результата поиска или из правила каталога (без подпунктов)."""
        hits = await self._hits([((id, edition), False, "exact")])
        return hits[0] if hits else None

    async def documents_for_direction(self, direction_code: str) -> list[str]:
        """Документы, применимые к программе направления: её ФГОС и нормы СПбГУ."""
        documents = await self._dao.list_documents()
        return [
            d["doc_id"]
            for d in documents
            if d["direction_code"] == direction_code or d["doc_type"] == "spbu_standard"
        ]

    # --- внутреннее

    async def _embed(self, query: str) -> list[float] | None:
        if self._embedder is None:
            return None
        try:
            (embedding,) = await asyncio.to_thread(self._embedder.embed, [query])
        except LLMError as error:
            logger.warning("эмбеддинги недоступны, только полнотекстовый поиск: %s", error)
            return None
        return list(embedding)

    async def _ranked(
        self,
        exact: Sequence[ClauseKey],
        found: Sequence[ClauseKey],
        method: Method,
        top_k: int,
        embedding: Sequence[float] | None,
    ) -> list[RegulationHit]:
        """Точные — первыми, затем найденные остальными способами; свернуть перечни,
        отрезать top_k, собрать тексты и сходство с запросом."""
        keys = list(dict.fromkeys([*exact, *found]))
        doc_ids = sorted({fragment_id.split(":", 1)[0] for fragment_id, _ in keys})
        heads = await self._dao.list_heads(doc_ids) if doc_ids else set()
        exact_items = {key for key, _ in collapse_lists(list(dict.fromkeys(exact)), heads)}
        items = [
            (key, expand, "exact" if key in exact_items else method)
            for key, expand in collapse_lists(keys, heads)[:top_k]
        ]
        hits = await self._hits(items)
        if embedding is not None and hits:
            clauses = {h.rank: [(i, h.edition) for i in h.covers or [h.id]] for h in hits}
            wanted = [key for ids in clauses.values() for key in ids]
            similarity = await self._dao.similarities(embedding, wanted)
            for hit in hits:
                values = [similarity[key] for key in clauses[hit.rank] if key in similarity]
                hit.similarity = round(max(values), 3) if values else None
        return hits

    async def _hits(self, items: Sequence[tuple[ClauseKey, bool, Method]]) -> list[RegulationHit]:
        plain = [key for key, expand, _ in items if not expand]
        by_key: dict[ClauseKey, list[dict]] = {}
        for row in await self._dao.get_parts(plain):
            by_key.setdefault((row["id"], row["edition"]), []).append(row)
        hits = []
        for key, expand, method in items:
            covers: list[str] = []
            if expand:
                doc_id, _, number = key[0].partition(":")
                rows = await self._dao.get_subtree(doc_id, key[1], number)
                clauses: dict[str, list[dict]] = {}
                for row in rows:
                    clauses.setdefault(row["number"], []).append(row)
                order = sorted(clauses, key=number_order)
                texts = ["".join(r["text"] for r in clauses[n]) for n in order]
                covers = [clauses[n][0]["id"] for n in order]
                first = clauses[number][0] if number in clauses else None
                text = "\n".join(texts)
            else:
                rows = by_key.get(key, [])
                first = rows[0] if rows else None
                text = "".join(r["text"] for r in rows)
            if first is None:
                continue
            hits.append(
                RegulationHit(
                    id=first["id"],
                    edition=first["edition"],
                    doc_id=first["doc_id"],
                    doc_type=first["doc_type"],
                    kind=first["kind"],
                    path=first["path"],
                    text=text,
                    context=first["context"],
                    category=first["category"],
                    rank=len(hits) + 1,
                    method=method,
                    covers=covers,
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
        similarity = "" if hit.similarity is None else f", сходство {hit.similarity:.2f}"
        print(f"{hit.rank}. [{hit.method}{similarity}] {hit.id} ({hit.edition})")
        print(f"   {hit.path}\n   {text}\n")
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
