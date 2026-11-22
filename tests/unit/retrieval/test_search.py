"""Логика поиска по нормативам на поддельном DAO: порядок способов, слияние, сборка частей."""

import asyncio

from edu_doc_agent.llm import FakeEmbedder, LLMError
from edu_doc_agent.retrieval.search import RegulationSearch

FGOS, OS = "fgos-02.03.02", "spbu-os"
ED = "808"


def row(id: str, text: str, *, part: int = 0, kind: str = "clause", doc_id: str = FGOS) -> dict:
    return {
        "id": id,
        "edition": ED,
        "part": part,
        "doc_id": doc_id,
        "doc_type": "fgos" if doc_id == FGOS else "spbu_standard",
        "kind": kind,
        "path": f"путь › {id}",
        "text": text,
        "context": "",
        "category": None,
    }


ROWS = [
    row(f"{FGOS}:1.9", "1.9. Объем программы 240 з.е."),
    row(f"{FGOS}:3.2:УК-1", "УК-1. Способен к поиску", kind="competency"),
    row(f"{FGOS}:4.1", "Первая часть. ", part=0),
    row(f"{FGOS}:4.1", "Вторая часть.", part=1),
    row(f"{OS}:2.3", "2.3. Пункт ОС", doc_id=OS),
]


class FakeDAO:
    def __init__(self, *, codes=(), numbers=(), full_text=(), nearest=()) -> None:
        self.answers = {
            "codes": list(codes),
            "numbers": list(numbers),
            "full_text": list(full_text),
            "nearest": list(nearest),
        }
        self.calls: list[tuple[str, dict]] = []

    async def _answer(self, name: str, kwargs: dict) -> list:
        self.calls.append((name, kwargs))
        return [(id, ED) for id in self.answers[name]]

    async def find_by_codes(self, codes, **kwargs):
        return await self._answer("codes", {"codes": codes, **kwargs})

    async def find_by_numbers(self, numbers, **kwargs):
        return await self._answer("numbers", {"numbers": numbers, **kwargs})

    async def full_text(self, text, **kwargs):
        return await self._answer("full_text", kwargs)

    async def nearest(self, embedding, **kwargs):
        return await self._answer("nearest", kwargs)

    async def get_parts(self, keys):
        wanted = set(keys)
        return [r for r in ROWS if (r["id"], r["edition"]) in wanted]

    async def list_documents(self):
        return [
            {"doc_id": FGOS, "doc_type": "fgos", "direction_code": "02.03.02"},
            {"doc_id": "fgos-01.03.02", "doc_type": "fgos", "direction_code": "01.03.02"},
            {"doc_id": OS, "doc_type": "spbu_standard", "direction_code": None},
        ]


class BrokenEmbedder:
    dim = 1024

    def embed(self, texts):
        raise LLMError("Ollama недоступна")


def run(coro):
    return asyncio.run(coro)


def names(dao: FakeDAO) -> list[str]:
    return [name for name, _ in dao.calls]


def test_code_in_query_finds_exactly() -> None:
    dao = FakeDAO(codes=[f"{FGOS}:3.2:УК-1"], full_text=[f"{FGOS}:1.9"])
    hits = run(RegulationSearch(dao, FakeEmbedder()).search("что требует ук-1", doc_ids=[FGOS]))
    assert [(h.id, h.method, h.rank) for h in hits] == [(f"{FGOS}:3.2:УК-1", "exact", 1)]
    assert names(dao) == ["codes"]  # гибрид не нужен
    assert dao.calls[0][1]["codes"] == ("УК-1",) and dao.calls[0][1]["doc_ids"] == [FGOS]


def test_exact_miss_falls_back_to_hybrid() -> None:
    dao = FakeDAO(
        codes=[],
        full_text=[f"{FGOS}:1.9", f"{FGOS}:4.1"],
        nearest=[f"{FGOS}:4.1", f"{OS}:2.3"],
    )
    hits = run(RegulationSearch(dao, FakeEmbedder()).search("УК-99 и объем", top_k=2))
    assert names(dao) == ["codes", "full_text", "nearest"]
    # 4.1 нашли оба способа — он первый; разделённый пункт склеен из частей
    assert [h.id for h in hits] == [f"{FGOS}:4.1", f"{FGOS}:1.9"]
    assert hits[0].text == "Первая часть. Вторая часть."
    assert {h.method for h in hits} == {"hybrid"} and [h.rank for h in hits] == [1, 2]


def test_without_embeddings_only_full_text() -> None:
    dao = FakeDAO(full_text=[f"{FGOS}:1.9"], nearest=[f"{OS}:2.3"])
    for embedder in (None, BrokenEmbedder()):
        hits = run(RegulationSearch(dao, embedder).search("объем программы"))
        assert [(h.id, h.method) for h in hits] == [(f"{FGOS}:1.9", "fts")]
    assert "nearest" not in names(dao)


def test_wrappers_filter_by_document_type() -> None:
    dao = FakeDAO(full_text=[f"{OS}:2.3"])
    search = RegulationSearch(dao)
    run(search.search_spbu_regulations("структура программы", doc_ids=[OS], top_k=3))
    run(search.search_fgos("структура программы"))
    run(search.search_all_regulations("структура программы"))
    types = [kwargs["doc_types"] for name, kwargs in dao.calls if name == "full_text"]
    assert types == [["spbu_standard"], ["fgos"], None]


def test_get_clause_and_documents_for_direction() -> None:
    search = RegulationSearch(FakeDAO())
    clause = run(search.get_fgos_clause(f"{FGOS}:4.1", ED))
    assert clause is not None and clause.text == "Первая часть. Вторая часть."
    assert run(search.get_clause(f"{FGOS}:9.9", ED)) is None
    assert run(search.documents_for_direction("02.03.02")) == [FGOS, OS]
