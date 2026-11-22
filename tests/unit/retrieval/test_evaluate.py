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
    outcomes = [Outcome(q, 1, ["x"]), Outcome(q, 3, ["a", "b", "x"]), Outcome(q, None, ["a"])]
    assert metrics(outcomes) == {"первым": 1 / 3, "в первых 5": 2 / 3}
    text = report({"вектор": outcomes, "гибрид, вес текста 0.25": outcomes})
    assert "| вектор | 33% (1) | 67% (2) |" in text
    assert "Правильный пункт не первым (гибрид, вес текста 0.25)" in text


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
    assert len(questions) >= 15
    assert all(q.question and q.direction and q.relevant for q in questions)
