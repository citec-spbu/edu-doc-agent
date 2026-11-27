"""Парные регрессии второго ревью PR #45: смысловые правки сохраняют связи."""

import pytest
from docx import Document
from openpyxl import load_workbook

from edu_doc_agent.eval.llm_gold import Report, score_case
from edu_doc_agent.extraction.curriculum import extract_plan_file
from edu_doc_agent.extraction.rpd import extract_rpd_file
from tests.unit.test_extraction_curriculum_xlsx import _small_plan, _without_sources


def test_incomplete_repeated_header_keeps_last_complete_layout(tmp_path):
    path = _small_plan(
        tmp_path,
        [
            ["С02. Семестр 2"],
            ["Блок.1.дисц", 1, "УК-1", "[100] Первый курс", "зачёт", 12, 12, 12],
            ["Блок.1.дисц", 1, "УК-1", "[200] Второй курс", "зачёт", 12, 12, 12],
            ["", "", "", "Second Course"],
        ],
    )
    wb = load_workbook(path)
    ws = wb.active
    for col in ("A", "B", "C", "E"):
        ws.merge_cells(f"{col}5:{col}6")
    wb.save(path)
    before = extract_plan_file(path)
    for merged in list(ws.merged_cells.ranges):
        ws.unmerge_cells(str(merged))
    ws.insert_rows(5)
    header = [ws.cell(2, col).value for col in range(1, 9)]
    header[3] = None
    for col, value in enumerate(header, 1):
        ws.cell(5, col, value)
    # insert_rows не переносит диапазоны объединений автоматически.
    for col in ("A", "B", "C", "E"):
        ws.merge_cells(f"{col}6:{col}7")
    wb.save(path)
    after = extract_plan_file(path)
    assert len(after.entries) == 2
    assert _without_sources(after.model_dump()) == _without_sources(before.model_dump())
    assert (after.entries[1].source.line, after.entries[1].source.end_line) == (6, 7)


def test_merged_choice_header_with_control_keeps_group_name(tmp_path):
    path = _small_plan(
        tmp_path,
        [
            ["Блок.1.дисц", "от 1", "УК-1", "Дисциплины по выбору:", "зачёты:"],
            ["", "до 2", "", "Иностранный язык", "зачёт"],
            ["Блок.1.дисц", 1, "УК-1", "[200] Второй курс", "зачёт", 12, 12, 12],
        ],
    )
    before = extract_plan_file(path)
    wb = load_workbook(path)
    wb.active.merge_cells("A3:A4")
    wb.save(path)
    after = extract_plan_file(path)
    assert _without_sources(after.model_dump()) == _without_sources(before.model_dump())
    assert len(after.entries) == 1
    assert after.entries[0].group == "Блок дисциплин по выбору: Иностранный язык"


def test_sheet_with_different_merges_is_not_reported_as_duplicate(tmp_path):
    path = _small_plan(
        tmp_path,
        [
            ["Блок.1.дисц", 1, "УК-1", "[100] Первый курс", "зачёт", 12, 12, 12],
            ["Блок.1.дисц", 1, "УК-1", "Academic Writing", "зачёт", 12, 12, 12],
        ],
    )
    wb = load_workbook(path)
    ws = wb.active
    wb.copy_worksheet(ws).title = "Без объединений"
    for col in ("A", "B", "C", "E", "F", "G", "H"):
        ws.merge_cells(f"{col}3:{col}4")
    wb.save(path)
    plan = extract_plan_file(path)
    assert [(e.name, e.name_en) for e in plan.entries] == [
        ("Первый курс", "Academic Writing"),
        ("Первый курс", None),
        ("Academic Writing", None),
    ]
    assert not any("полностью совпадает" in w for w in plan.warnings)


def test_note_outside_load_columns_keeps_translation_and_source(tmp_path):
    path = _small_plan(
        tmp_path,
        [
            ["Блок.1.дисц", 1, "УК-1", "[100] Первый курс", "зачёт", 12, 12, 12],
            ["", "", "", "First Course"],
        ],
    )
    wb = load_workbook(path)
    for col in ("A", "B", "C", "E"):
        wb.active.merge_cells(f"{col}3:{col}4")
    wb.save(path)
    before = extract_plan_file(path)
    wb.active["U4"] = "Уточнить даты"
    wb.save(path)
    after = extract_plan_file(path)
    assert _without_sources(after.model_dump()) == _without_sources(before.model_dump())
    assert len(after.entries) == 1 and after.entries[0].name_en == "First Course"
    assert (after.entries[0].source.line, after.entries[0].source.end_line) == (3, 4)
    assert "Уточнить даты" in after.entries[0].source.quote


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("coded", [False, True])
def test_merged_outcome_keeps_explicit_and_inherited_owners(tmp_path, inherited, coded):
    doc = Document()
    table = doc.add_table(rows=4, cols=3)
    for cell, text in zip(
        table.rows[0].cells,
        ["Код и наименование компетенции", "Планируемые результаты обучения", "Код индикатора"],
        strict=True,
    ):
        cell.text = text
    table.cell(1, 0).text = "УК-1. Способен решать задачи"
    table.cell(1, 1).text = "знает:\nотдельные методы"
    table.cell(1, 2).text = "УК-1.1. Решает задачи"
    if not inherited:
        table.cell(2, 0).text = "УК-1. Способен решать задачи"
    table.cell(2, 2).text = "УК-1.2. Применяет методы"
    table.cell(3, 0).text = "УК-2. Способен оценивать информацию"
    table.cell(3, 2).text = "УК-2.1. Оценивает информацию"
    table.cell(2, 1).merge(table.cell(3, 1)).text = (
        "УК-1.2 З-1. Знает способы поиска.\nУК-2.1 З-1. Знает критерии выбора."
        if coded
        else "знает:\nобщие методы поиска"
    )
    path = tmp_path / "outcomes.docx"
    doc.save(path)
    result = extract_rpd_file(path)
    assert [c.code for c in result.competences] == ["УК-1", "УК-2"]
    assert {i.code: i.competence_code for i in result.indicators} == {
        "УК-1.1": "УК-1",
        "УК-1.2": "УК-1",
        "УК-2.1": "УК-2",
    }
    if coded:
        assert {o.code: o.competence_codes for o in result.learning_outcomes[1:]} == {
            "УК-1.2 З-1": ["УК-1"],
            "УК-2.1 З-1": ["УК-2"],
        }
    else:
        assert len(result.learning_outcomes) == 2
        assert result.learning_outcomes[1].competence_codes == ["УК-1", "УК-2"]


@pytest.mark.parametrize("attribute", ["condition", "effect"])
@pytest.mark.parametrize(
    "expected,predicted",
    [
        (
            "не менее 60 баллов за работу и не менее 80 за экзамен",
            "не менее 80 за экзамен и не менее 60 баллов за работу",
        ),
        ("не менее 60 и не более 80 баллов", "не более 80 баллов и не менее 60"),
    ],
)
def test_reordering_whole_conjuncts_preserves_grading_score(expected, predicted, attribute):
    rule = {"condition": "условие", "effect": "допуск к экзамену", "quote": "исходная цитата"}
    score = score_case(
        "grading_rules",
        {"rules": [dict(rule, **{attribute: predicted})]},
        {"rules": [dict(rule, **{attribute: expected})]},
    )
    assert (score.tp, score.fp, score.fn) == (1, 0, 0)
    assert score.attribute_accuracy == 1.0
    assert Report(by_task={"grading_rules": score}).passed()


@pytest.mark.parametrize(
    "predicted",
    [
        "не менее 80 баллов за работу и не менее 60 за экзамен",
        "не менее 60 баллов за работу или не менее 80 за экзамен",
        "не менее 60 баллов за работу и не менее 60 баллов за работу",
        "не менее 60 баллов за работу",
    ],
)
def test_conjunction_normalization_keeps_threshold_owners_and_all_conditions(predicted):
    rule = {
        "condition": "не менее 60 баллов за работу и не менее 80 за экзамен",
        "effect": "допуск к экзамену",
        "quote": "исходная цитата",
    }
    score = score_case(
        "grading_rules", {"rules": [dict(rule, condition=predicted)]}, {"rules": [rule]}
    )
    assert score.attribute_accuracy == 0.5
    assert not Report(by_task={"grading_rules": score}).passed()
