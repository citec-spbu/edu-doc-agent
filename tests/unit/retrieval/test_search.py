"""Логика поиска по нормативам на поддельном DAO: порядок способов, слияние, сборка частей."""

import asyncio

from edu_doc_agent.llm import FakeEmbedder, LLMError
from edu_doc_agent.retrieval.search import RegulationSearch
from edu_doc_agent.retrieval.tools import RegulationTools

FGOS, OS = "fgos-02.03.02", "spbu-os"
ED = "808"


def row(id: str, text: str, *, part: int = 0, kind: str = "clause", doc_id: str = FGOS) -> dict:
    return {
        "id": id,
        "number": id.split(":")[1],
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
    row(f"{OS}:2.2.4", "2.2.4. рабочие программы, содержащие:", doc_id=OS),
    row(f"{OS}:2.2.4.2", "2.2.4.2. фонд оценочных средств;", doc_id=OS),
    row(f"{OS}:2.2.4.1", "2.2.4.1. цели и задачи;", doc_id=OS),
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

    async def similarities(self, embedding, keys):
        self.calls.append(("similarities", {"keys": list(keys)}))
        # у каждого пункта своё сходство; у частей одного пункта — одно
        return {key: round(0.5 + 0.01 * len(key[0]), 3) for key in keys}

    async def get_parts(self, keys):
        wanted = set(keys)
        return [r for r in ROWS if (r["id"], r["edition"]) in wanted]

    async def list_heads(self, doc_ids):
        return {(OS, ED, "2.2.4")}

    async def get_subtree(self, doc_id, edition, number):
        return [
            r
            for r in ROWS
            if r["doc_id"] == doc_id
            and r["kind"] == "clause"
            and (r["number"] == number or r["number"].startswith(number + "."))
        ]

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


def test_exact_match_comes_first_and_the_rest_is_filled_by_hybrid() -> None:
    dao = FakeDAO(
        codes=[f"{FGOS}:3.2:УК-1"],
        full_text=[f"{FGOS}:1.9", f"{FGOS}:3.2:УК-1"],
        nearest=[f"{FGOS}:3.2:УК-1", f"{FGOS}:4.1"],
    )
    search = RegulationSearch(dao, FakeEmbedder())
    hits = run(search.search("в обязательной ли части дисциплины для ук-1", doc_ids=[FGOS]))
    # найденное точно — первым, остальное — гибрид по всему запросу, без повторов
    assert [(h.id, h.method) for h in hits] == [
        (f"{FGOS}:3.2:УК-1", "exact"),
        (f"{FGOS}:4.1", "hybrid"),
        (f"{FGOS}:1.9", "hybrid"),
    ]
    assert names(dao) == ["codes", "full_text", "nearest", "similarities"]
    assert dao.calls[0][1]["codes"] == ("УК-1",) and dao.calls[0][1]["doc_ids"] == [FGOS]
    assert all(h.similarity is not None for h in hits)
    assert run(search.search("УК-1", top_k=1))[0].method == "exact"


def test_exact_miss_falls_back_to_hybrid() -> None:
    dao = FakeDAO(
        codes=[],
        full_text=[f"{FGOS}:1.9", f"{FGOS}:4.1"],
        nearest=[f"{FGOS}:4.1", f"{OS}:2.3"],
    )
    hits = run(RegulationSearch(dao, FakeEmbedder()).search("УК-99 и объем", top_k=2))
    assert names(dao) == ["codes", "full_text", "nearest", "similarities"]
    # 4.1 нашли оба способа — он первый; второй — из векторного: у него вес больше,
    # чем у полнотекстового; разделённый пункт склеен из частей
    assert [h.id for h in hits] == [f"{FGOS}:4.1", f"{OS}:2.3"]
    assert hits[0].text == "Первая часть. Вторая часть."
    assert {h.method for h in hits} == {"hybrid"} and [h.rank for h in hits] == [1, 2]


def test_hybrid_uses_all_words_and_fallback_any_word() -> None:
    dao = FakeDAO(full_text=[f"{FGOS}:1.9"])
    run(RegulationSearch(dao, FakeEmbedder()).search("объем программы"))  # гибрид — И
    run(RegulationSearch(dao).search("объем программы"))  # без эмбеддингов — ИЛИ
    run(RegulationSearch(dao, FakeEmbedder(), fts_match="any").search("объем программы"))
    run(RegulationSearch(dao).search("объем программы", fts_match="all"))  # явно — как задано
    matches = [kwargs["match"] for name, kwargs in dao.calls if name == "full_text"]
    assert matches == ["all", "any", "any", "all"]


def test_without_embeddings_only_full_text() -> None:
    dao = FakeDAO(full_text=[f"{FGOS}:1.9"], nearest=[f"{OS}:2.3"])
    for embedder in (None, BrokenEmbedder()):
        hits = run(RegulationSearch(dao, embedder).search("объем программы"))
        assert [(h.id, h.method, h.similarity) for h in hits] == [(f"{FGOS}:1.9", "fts", None)]
    assert "nearest" not in names(dao) and "similarities" not in names(dao)


def test_tools_pass_program_documents_and_types_to_queries() -> None:
    dao = FakeDAO(full_text=[f"{OS}:2.3"])
    tools = RegulationTools(RegulationSearch(dao), doc_ids=[FGOS, OS])
    run(tools.search_spbu_regulations("структура программы", top_k=3))
    run(tools.search_fgos("структура программы"))
    run(tools.search_prof_standards("программист"))
    run(tools.search_all_regulations("структура программы"))
    calls = [kwargs for name, kwargs in dao.calls if name == "full_text"]
    assert [(c["doc_types"], c["kinds"]) for c in calls] == [
        (["spbu_standard"], None),
        (["fgos"], None),
        (None, ["prof_standard_ref"]),
        (None, None),
    ]
    assert all(c["doc_ids"] == [FGOS, OS] for c in calls)


def test_get_clause_and_documents_for_direction() -> None:
    search = RegulationSearch(FakeDAO())
    clause = run(search.get_clause(f"{FGOS}:4.1", ED))
    assert clause is not None and clause.text == "Первая часть. Вторая часть."
    assert run(search.get_clause(f"{FGOS}:9.9", ED)) is None
    assert run(search.documents_for_direction("02.03.02")) == [FGOS, OS]


def test_list_item_returns_whole_list() -> None:
    dao = FakeDAO(full_text=[f"{OS}:2.2.4.2", f"{OS}:2.3"], nearest=[f"{OS}:2.2.4.1"])
    hits = run(RegulationSearch(dao, FakeEmbedder()).search("что содержит рабочая программа"))
    head = hits[0]
    assert head.id == f"{OS}:2.2.4"
    assert head.text == (
        "2.2.4. рабочие программы, содержащие:\n2.2.4.1. цели и задачи;\n"
        "2.2.4.2. фонд оценочных средств;"
    )
    assert head.covers == [f"{OS}:2.2.4", f"{OS}:2.2.4.1", f"{OS}:2.2.4.2"]
    assert [h.id for h in hits] == [f"{OS}:2.2.4", f"{OS}:2.3"]
    # сходство перечня — лучшее из его пунктов
    assert head.similarity == max(round(0.5 + 0.01 * len(i), 3) for i in head.covers)
    # чтение по ссылке — сам пункт, без подпунктов
    assert run(RegulationSearch(dao).get_clause(f"{OS}:2.2.4", ED)).covers == []
