"""Метрики оценки поиска (retrieval.evaluate) без базы."""

from pathlib import Path

from edu_doc_agent.retrieval.evaluate import (
    Outcome,
    Question,
    first_relevant,
    is_relevant,
    load_golden,
    metrics,
    report,
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
    assert all(
        r.split(":", 1)[0] in {f"fgos-{q.direction}", "spbu-os"}
        for q in answerable
        for r in q.relevant
    )
