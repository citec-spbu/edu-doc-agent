"""Поиск по нормативам в памяти — заглушка для тестов графа без базы и модели эмбеддингов.

Интерфейс тот же, что у RegulationSearch (см. retrieval/tools.py, SearchBackend):

    corpus = [chunk_document(...)]          # или CorpusDocument из YAML-снимка data/corpus
    tools = await RegulationTools.for_program(InMemoryRegulationSearch(corpus), "02.03.02")

Точный поиск по коду и номеру пункта — как в настоящем (найденное первым, остальное добирается
поиском по словам); вместо гибридного — число общих слов запроса и фрагмента
(method="fts", similarity=None). Качество поиска на заглушке не проверяют.
"""

import re
from collections.abc import Sequence

from edu_doc_agent.retrieval.corpus.models import CorpusDocument, Fragment
from edu_doc_agent.retrieval.ranking import query_keys
from edu_doc_agent.retrieval.search import RegulationHit

_WORD = re.compile(r"\w{3,}")


class InMemoryRegulationSearch:
    def __init__(self, documents: Sequence[CorpusDocument]) -> None:
        self._documents = [d.document for d in documents]
        self._fragments = [f for d in documents for f in d.fragments]

    async def search(
        self,
        query: str,
        *,
        doc_ids: Sequence[str] | None = None,
        doc_types: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
        top_k: int = 5,
    ) -> list[RegulationHit]:
        pool = [
            f
            for f in self._fragments
            if (doc_ids is None or f.doc_id in doc_ids)
            and (doc_types is None or f.doc_type in doc_types)
            and (kinds is None or f.kind in kinds)
        ]
        keys = query_keys(query)
        exact = sorted(  # как в настоящем: сначала сами компетенции, потом где упомянуты
            (
                f
                for f in pool
                if set(keys.codes) & set(f.codes)
                or (f.kind == "clause" and f.number in keys.numbers)
            ),
            key=lambda f: f.kind == "clause",
        )
        words = _words(query)
        scored = [(len(words & _words(f"{f.path} {f.context} {f.text}")), f) for f in pool]
        ranked = [f for score, f in sorted(scored, key=lambda x: -x[0]) if score > 0]
        return self._hits([*((f, "exact") for f in exact), *((f, "fts") for f in ranked)], top_k)

    async def get_clause(self, id: str, edition: str) -> RegulationHit | None:
        found = [f for f in self._fragments if f.id == id and f.edition == edition]
        hits = self._hits([(f, "exact") for f in found], 1)
        return hits[0] if hits else None

    async def documents_for_direction(self, direction_code: str) -> list[str]:
        return [
            d.doc_id
            for d in self._documents
            if d.direction_code == direction_code or d.doc_type == "spbu_standard"
        ]

    def _hits(self, found: list[tuple[Fragment, str]], top_k: int) -> list[RegulationHit]:
        hits: list[RegulationHit] = []
        seen: set[tuple[str, str]] = set()
        for f, method in found:
            key = (f.id, f.edition)
            if key in seen:
                continue
            seen.add(key)
            parts = sorted(
                (p for p in self._fragments if (p.id, p.edition) == key), key=lambda p: p.part
            )
            hits.append(
                RegulationHit(
                    id=f.id,
                    edition=f.edition,
                    doc_id=f.doc_id,
                    doc_type=f.doc_type,
                    kind=f.kind,
                    path=f.path,
                    text="".join(p.text for p in parts),
                    context=f.context,
                    category=f.category,
                    rank=len(hits) + 1,
                    method=method,
                )
            )
            if len(hits) == top_k:
                break
        return hits


def _words(text: str) -> set[str]:
    return {w[:6] for w in _WORD.findall(text.lower())}  # грубая «основа» слова
