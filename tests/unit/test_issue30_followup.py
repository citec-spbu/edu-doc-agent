"""Остаточные контрпримеры issue #30: объединения и полнота смысловых метрик."""

import pytest
from docx import Document

from edu_doc_agent.eval.characteristic_llm_gold import score_characteristic_llm
from edu_doc_agent.eval.llm_gold import Report, _same_meaning, score_case
from edu_doc_agent.extraction.characteristic_llm import CharacteristicLLM, LLMEntity
from edu_doc_agent.extraction.rpd import extract_rpd_file
from edu_doc_agent.extraction.source import SourceRef


@pytest.mark.parametrize("merge_competences", [False, True])
@pytest.mark.parametrize("coded", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_merged_outcomes_keep_owners_from_all_covered_competence_cells(
    tmp_path, merge_competences, coded, reverse
):
    doc = Document()
    table = doc.add_table(rows=3, cols=3)
    for cell, text in zip(
        table.rows[0].cells,
        ["Код и наименование компетенции", "Планируемые результаты обучения", "Код индикатора"],
        strict=True,
    ):
        cell.text = text
    codes = ["УК-1", "УК-2"]
    if reverse:
        codes.reverse()
    for row, code in enumerate(codes, 1):
        table.cell(row, 0).text = f"{code}. Способен решать задачи"
        table.cell(row, 2).text = f"{code}.1. Решает задачи"
    if merge_competences:
        table.cell(1, 0).merge(table.cell(2, 0)).text = "\n".join(
            f"{code}. Способен решать задачи" for code in codes
        )
    table.cell(1, 1).merge(table.cell(2, 1)).text = (
        "УК-1.1 З-1. Знает способы поиска.\nУК-2.1 З-1. Знает критерии выбора."
        if coded
        else "знает:\nосновы поиска и оценки информации"
    )
    path = tmp_path / "outcomes.docx"
    doc.save(path)
    result = extract_rpd_file(path)
    assert [c.code for c in result.competences] == codes
    assert {i.code: i.competence_code for i in result.indicators} == {
        "УК-1.1": "УК-1",
        "УК-2.1": "УК-2",
    }
    if coded:
        assert {o.code: o.competence_codes for o in result.learning_outcomes} == {
            "УК-1.1 З-1": ["УК-1"],
            "УК-2.1 З-1": ["УК-2"],
        }
    else:
        assert len(result.learning_outcomes) == 1
        assert result.learning_outcomes[0].competence_codes == codes
    assert all(o.source.file == path.name and o.source.quote for o in result.learning_outcomes)


@pytest.mark.parametrize(
    "expected,predicted",
    [
        ("не выполнена работа и сдан экзамен", "выполнена работа и не сдан экзамен"),
        ("не менее 60 и не более 80 баллов", "не менее 80 и не более 60 баллов"),
        (
            "не менее 60 баллов за работу и не менее 80 за экзамен",
            "не менее 80 баллов за работу и не менее 60 за экзамен",
        ),
        ("не менее -60 баллов", "не менее 60 баллов"),
        ("результат = 60", "результат != 60"),
        ("не менее 60%", "не менее 60"),
    ],
)
@pytest.mark.parametrize("attribute", ["condition", "effect"])
def test_relocated_negations_and_bounds_fail_grading_acceptance(expected, predicted, attribute):
    rule = {"condition": "условие", "effect": "допуск к экзамену", "quote": "одна исходная цитата"}
    gold = dict(rule, **{attribute: expected})
    got = dict(rule, **{attribute: predicted})
    score = score_case("grading_rules", {"rules": [got]}, {"rules": [gold]})
    assert (score.tp, score.fp, score.fn) == (1, 0, 0)
    assert score.attribute_accuracy == 0.5
    report = Report(by_task={"grading_rules": score})
    assert not report.passed()
    assert not _same_meaning(expected, predicted)
    assert not _same_meaning(predicted, expected)


@pytest.mark.parametrize(
    "condition,paraphrase",
    [
        ("не менее 60 баллов", "НЕ МЕНЬШЕ 60 баллов."),
        ("не более 80 баллов", "не больше 80 баллов"),
        ("не выполнена работа и сдан экзамен", "Не выполнена работа\nи сдан экзамен."),
        ("не менее 60 и не более 80 баллов", "не меньше 60 и не больше 80 баллов"),
        ("не менее 60,5 баллов", ">= 60.5 баллов"),
    ],
)
def test_grading_guard_keeps_same_ordered_condition(condition, paraphrase):
    assert _same_meaning(condition, paraphrase)
    assert _same_meaning(paraphrase, condition)


@pytest.mark.parametrize("predicted", ["директор", "колледжа", "директор морского"])
def test_incomplete_position_cannot_pass_with_its_first_word(predicted):
    report = _position_report(predicted, "директор морского колледжа")
    score = report.scores["positions"]
    assert (score.tp, score.fp, score.fn) == (0, 1, 1)
    assert not report.passed()


@pytest.mark.parametrize(
    "predicted,expected",
    [
        ("директор", "директор колледжа"),
        ("директор колледжа", "директор"),
        ("директор колледжа и института", "директор колледжа"),
    ],
)
def test_position_extent_is_checked_in_both_directions(predicted, expected):
    assert not _position_report(predicted, expected).passed()


@pytest.mark.parametrize(
    "predicted", ["ДИРЕКТОР КОЛЛЕДЖА", "директоров колледжей", "директор\nколледжа."]
)
def test_full_position_keeps_existing_normalization(predicted):
    assert _position_report(predicted, "директор колледжа").passed()


def _position_report(predicted, expected):
    gold = {
        "annotation": {
            "roles": [],
            "positions": [{"name": expected}],
            "tracks": [],
            "projects": [],
            "legal_acts": [],
        },
        "mission_principles": [],
        "individualization": {"profiles": [], "means": []},
    }
    doc = CharacteristicLLM(
        positions=[
            LLMEntity(name=predicted, quote=predicted, source=SourceRef(file="synthetic.md"))
        ]
    )
    return score_characteristic_llm(doc, gold)
