"""Регрессионные тесты к замечаниям ревью PR #22 (синтетические тексты)."""

import json

import pytest

from edu_doc_agent.config import Settings
from edu_doc_agent.eval.llm_gold import Report, Score, score_case
from edu_doc_agent.extraction.freetext import _under
from edu_doc_agent.extraction.llm_extract import extract_with_llm, quote_in_text
from edu_doc_agent.extraction.md_utils import table_row_text
from edu_doc_agent.extraction.models import LearningOutcome, RPDExtraction, WorkloadRow
from edu_doc_agent.extraction.rpd import extract_rpd
from edu_doc_agent.llm import FakeLLMClient, LLMError

PREREQ_MD = """## 1.2. Требования к подготовленности обучающегося

Нужны:

знание высокоуровневых языков программирования (С, С++, Java), основ архитектуры.

навыки работы в одной и более операционных системах на уровне пользователя.
"""


def test_settings_empty_llm_rpm_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_RPM", "")
    assert Settings(_env_file=None).llm_rpm is None
    monkeypatch.setenv("LLM_RPM", "0")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_quote_fragments_must_keep_source_order() -> None:
    text = "Основы архитектуры вычислительных машин. Языки: C, Java."
    assert quote_in_text("Основы архитектуры ... Java", text)
    assert not quote_in_text("Java ... Основы архитектуры", text)


def test_llm_error_degrades_to_warning() -> None:
    def fail(call):
        raise LLMError("503 unavailable")

    result = extract_with_llm(PREREQ_MD, "x.md", FakeLLMClient(respond=fail))
    assert any("503 unavailable" in w for w in result.warnings)


def test_llm_source_line_points_to_quote() -> None:
    reply = json.dumps(
        {
            "items": [
                {
                    "name": "операционные системы",
                    "kind": "область знаний",
                    "relation": "предшествует",
                    "quote": "навыки работы в одной и более операционных системах",
                }
            ]
        },
        ensure_ascii=False,
    )
    result = extract_with_llm(PREREQ_MD, "x.md", FakeLLMClient(replies=[reply]), ["prerequisites"])
    assert result.prerequisites[0].source.line == 7


def test_table_row_text_keeps_equal_neighbours() -> None:
    assert table_row_text("| 28 | 28 |  | ИТОГО |") == "28 | 28 | ИТОГО"


def test_section_prefix_compares_components() -> None:
    assert _under("3.1.3.2", "3.1.3") and _under("3.1", "3.1")
    assert not _under("3.10", "3.1") and not _under("2.10.1", "2.1")


def test_prerequisite_keeps_parenthetical_and_bare_skills() -> None:
    names = [p.name for p in extract_rpd(PREREQ_MD, file="x.md").prerequisites]
    assert any("(С, С++, Java)" in n for n in names)
    assert any("операционных системах" in n for n in names)


def test_json_schema_matches_dump() -> None:
    schema = RPDExtraction.model_json_schema()
    assert "total_hours" in schema["$defs"]["WorkloadRow"]["properties"]
    assert "competence_code" in schema["$defs"]["LearningOutcome"]["properties"]
    assert WorkloadRow.model_json_schema()["additionalProperties"] is False
    assert LearningOutcome


def _prereq(name: str, kind: str, relation: str) -> dict:
    return {"name": name, "kind": kind, "relation": relation, "quote": "q"}


def test_inverted_attributes_fail_the_report() -> None:
    gold = {"items": [_prereq("алгебра", "дисциплина", "предшествует")]}
    pred = {"items": [_prereq("алгебра", "область знаний", "параллельно")]}
    report = Report()
    report.by_task["prerequisites"] = score_case("prerequisites", pred, gold)
    assert report.by_task["prerequisites"].f1 == 1.0
    assert not report.passed()


def test_grading_rule_meaning_is_scored() -> None:
    quote = "При 0 баллов за вопрос оценка неудовлетворительно"
    gold = {
        "rules": [
            {
                "condition": "0 баллов за вопрос",
                "effect": "оценка «неудовлетворительно»",
                "quote": quote,
            }
        ]
    }
    good = score_case("grading_rules", gold, gold)
    wrong = {
        "rules": [{"condition": "сданы все задания", "effect": "плюс 5 баллов", "quote": quote}]
    }
    bad = score_case("grading_rules", wrong, gold)
    assert good.attribute_accuracy == 1.0
    assert bad.f1 == 1.0 and bad.attribute_accuracy == 0.0
    assert isinstance(good, Score)
