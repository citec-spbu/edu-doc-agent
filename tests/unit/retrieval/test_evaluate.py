"""Метрики оценки поиска (retrieval.evaluate) без базы."""

import asyncio
from pathlib import Path

from edu_doc_agent.retrieval.evaluate import (
    Outcome,
    Question,
    Variant,
    evaluate_by_tool,
    first_relevant,
    is_relevant,
    load_golden,
    metrics,
    report,
    split_by_type,
    variants,
)
from edu_doc_agent.retrieval.search import RegulationHit

GOLDEN = Path(__file__).parents[3] / "data" / "regulations" / "golden.yaml"


def test_relevance_supports_prefix() -> None:
    assert is_relevant("fgos-01.03.02:app:06.001", ["fgos-01.03.02:app:*"])
    assert not is_relevant("fgos-01.03.02:1.9", ["fgos-01.03.02:app:*"])
    assert is_relevant("spbu-os:2.2.4", ["spbu-os:2.2.4"])
    assert not is_relevant("spbu-os:2.2.4.1", ["spbu-os:2.2.4"])


def test_metrics_and_report() -> None:
    q = Question("вопрос", "02.03.02", ["x"])
    none = Question("Сколько стоит обучение?", "02.03.02", [])
    outcomes = [
        Outcome(q, 1, ["x"], 0.71),
        Outcome(q, 3, ["a", "b", "x"], 0.62),
        Outcome(q, None, ["a"], 0.55),
        Outcome(none, None, ["a"], 0.41),  # вопрос без ответа в метрики не входит
    ]
    assert metrics(outcomes) == {"первым": 1 / 3, "в первых 5": 2 / 3}
    text = report({"вектор": outcomes, "гибрид, ИЛИ, вес текста 0.25": outcomes})
    assert "Вопросов с ответом: 3" in text
    assert "| вектор | 33% (1) | 67% (2) |" in text
    assert "Правильный пункт не первым (гибрид, ИЛИ, вес текста 0.25)" in text
    assert "- есть ответ: от 0.55 до 0.71, медиана 0.62" in text
    assert "- ответа нет: от 0.41 до 0.41" in text
    assert "Сколько стоит обучение? — 0.41, a" in text


def test_variants_compare_vector_text_and_hybrids() -> None:
    names = [v.name for v in variants([1.0, 0.25])]
    assert names == [
        "вектор",
        "текст, ИЛИ",
        "текст, И",
        "гибрид, ИЛИ, вес текста 1",
        "гибрид, ИЛИ, вес текста 0.25",
        "гибрид, И, вес текста 1",
        "гибрид, И, вес текста 0.25",
    ]


def test_list_result_counts_if_it_covers_relevant_clause() -> None:
    hit = RegulationHit(
        id="spbu-os:2.2",
        edition="1",
        doc_id="spbu-os",
        doc_type="spbu_standard",
        kind="clause",
        path="2 › 2.2",
        text="…",
        rank=2,
        method="hybrid",
        covers=["spbu-os:2.2", "spbu-os:2.2.4", "spbu-os:2.2.4.1"],
    )
    assert first_relevant([hit], ["spbu-os:2.2.4.1"]) == 2
    assert first_relevant([hit], ["spbu-os:3.1"]) is None


def test_golden_set_is_well_formed() -> None:
    questions = load_golden(GOLDEN)
    answerable = [q for q in questions if q.relevant]
    assert len(answerable) >= 30 and len(questions) > len(answerable)  # есть и без ответа
    assert all(q.question and q.direction for q in questions)
    assert len({q.direction for q in answerable}) >= 5  # не одно направление
    # ФГОС — только своего направления; остальные документы общие для всех программ
    assert all(
        r.split(":", 1)[0] == f"fgos-{q.direction}" or not r.startswith("fgos-")
        for q in answerable
        for r in q.relevant
    )
    prefixes = {r.split(":", 1)[0] for q in answerable for r in q.relevant}
    assert {"273-fz", "order-245", "order-885-390", "ps-06.001"} <= prefixes


TYPES = {"fgos-02.03.02": "fgos", "spbu-os": "spbu_standard", "273-fz": "federal_act"}


def test_questions_are_split_by_document_type_of_answers() -> None:
    both = Question("объем", "02.03.02", ["fgos-02.03.02:1.9", "spbu-os:2.8.1", "x-99:1"])
    law = Question("ч. 5 ст. 13", "02.03.02", ["273-fz:ст.13:ч.5"])
    none = Question("стоимость", "02.03.02", [])
    split = split_by_type([both, law, none], TYPES)
    assert [q.relevant for q in split["fgos"]] == [["fgos-02.03.02:1.9"]]
    assert [q.relevant for q in split["spbu_standard"]] == [["spbu-os:2.8.1"]]
    assert [q.question for q in split["federal_act"]] == ["ч. 5 ст. 13"]
    assert split["prof_standard"] == []  # пункт незагруженного документа x-99 не учтён


class FakeSearch:
    """Поиск, который возвращает пункты только запрошенных типов документов."""

    def __init__(self) -> None:
        self.doc_types: list = []

    async def document_types(self) -> dict[str, str]:
        return TYPES

    async def documents_for_direction(self, direction_code: str) -> list[str]:
        return list(TYPES)

    async def search(self, query, *, doc_types=None, top_k=5, **kwargs) -> list[RegulationHit]:
        self.doc_types.append(doc_types)
        found = [("273-fz:ст.13:ч.7", "273-fz"), ("fgos-02.03.02:1.9", "fgos-02.03.02"),
                 ("spbu-os:2.8.1", "spbu-os")]  # fmt: skip
        hits = [(i, d) for i, d in found if doc_types is None or TYPES[d] in doc_types]
        return [
            RegulationHit(id=i, edition="1", doc_id=d, doc_type=TYPES[d], kind="clause", path=i,
                          text="…", rank=n, method="hybrid")
            for n, (i, d) in enumerate(hits, 1)
        ]  # fmt: skip


def test_each_tool_searches_only_its_document_type() -> None:
    search = FakeSearch()
    q = Question("объем", "02.03.02", ["fgos-02.03.02:1.9", "spbu-os:2.8.1"])
    by_tool = asyncio.run(evaluate_by_tool(search, [q], Variant("гибрид", "auto")))
    assert sorted(by_tool) == ["fgos", "spbu_standard"]
    assert sorted(t for (t,) in search.doc_types) == ["fgos", "spbu_standard"]
    assert by_tool["fgos"][0].rank == 1  # закон в выдаче search_fgos не мешает
    everything = [Outcome(q, 2, ["273-fz:ст.13:ч.7", "fgos-02.03.02:1.9"])]
    text = report({"гибрид, И, вес текста 0.25": everything}, by_tool)
    assert "| search_fgos (fgos) | 1 | 100% (1) | 100% (1) |" in text
    assert "| search_all_regulations (все документы) | 1 | 0% (0) | 100% (1) |" in text
