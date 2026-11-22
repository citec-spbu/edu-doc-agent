"""Метрики оценки поиска (retrieval.evaluate) без базы."""

from pathlib import Path

from edu_doc_agent.retrieval.evaluate import (
    Outcome,
    Question,
    is_relevant,
    load_golden,
    metrics,
    report,
)

GOLDEN = Path(__file__).parents[3] / "data" / "regulations" / "golden.yaml"


def test_relevance_supports_prefix() -> None:
    assert is_relevant("fgos-01.03.02:app:06.001", ["fgos-01.03.02:app:*"])
    assert not is_relevant("fgos-01.03.02:1.9", ["fgos-01.03.02:app:*"])
    assert is_relevant("spbu-os:2.2.4", ["spbu-os:2.2.4"])
    assert not is_relevant("spbu-os:2.2.4.1", ["spbu-os:2.2.4"])


def test_metrics_and_report() -> None:
    q = Question("вопрос", "02.03.02", ["x"])
    outcomes = [Outcome(q, 1, ["x"]), Outcome(q, 3, ["a", "b", "x"]), Outcome(q, None, ["a"])]
    values = metrics(outcomes)
    assert values["Hit@1"] == 1 / 3 and values["Hit@3"] == 2 / 3 and values["Hit@5"] == 2 / 3
    assert abs(values["MRR@10"] - (1 + 1 / 3) / 3) < 1e-9
    text = report({"auto": outcomes, "fts": outcomes})
    assert "| auto | 0.33 | 0.67 | 0.67 | 0.44 |" in text
    assert "Не в первой тройке (auto)" in text


def test_golden_set_is_well_formed() -> None:
    questions = load_golden(GOLDEN)
    assert len(questions) >= 15
    assert all(q.question and q.direction and q.relevant for q in questions)
