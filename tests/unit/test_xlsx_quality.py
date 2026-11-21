"""Чувствительность XLSX-замера к потерям контекста, перестановкам, дублям и XML."""

import copy
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from edu_doc_agent.eval.program_ner import compare_groups
from edu_doc_agent.eval.xlsx_documents import _sources, main, plan_groups
from edu_doc_agent.eval.xlsx_reference import build_plan_reference, read_xlsx_xml
from edu_doc_agent.extraction.curriculum import CurriculumPlan, PlanEntry


@pytest.fixture
def examples() -> tuple[dict, dict]:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "plan_gold.json"
    gold = json.loads(path.read_text())
    gold["sheet"] = "Синтетика"
    gold["entries"][0]["group"] = "Группа морских систем"
    predicted = copy.deepcopy(gold)
    predicted["entries"] = []
    for row in gold["entries"]:
        entry = copy.deepcopy(row)
        source = {"line": entry.pop("line"), "section": "Синтетика", "file": "synthetic.xlsx"}
        for old, new in (
            ("contact", "contact_hours"),
            ("self", "self_study_hours"),
            ("active", "active_hours"),
        ):
            entry[new] = entry.pop(old)
        predicted["entries"].append(PlanEntry(source=source, **entry).model_dump())
    return gold, predicted


def _compare(gold: dict, predicted: dict) -> dict:
    return compare_groups(plan_groups(predicted), plan_groups(gold, reference=True))


def test_integer_and_float_fields_are_equivalent(examples: tuple[dict, dict]) -> None:
    gold, predicted = examples
    assert all(s["f1"] == 1 for s in _compare(gold, predicted).values())


@pytest.mark.parametrize("field", ["group", "part", "profile", "trajectory", "semester"])
def test_lost_context_is_not_hidden_by_correct_entity_counts(examples, field) -> None:
    gold, predicted = examples
    target = next(e for e in predicted["entries"] if e[field] is not None)
    target[field] = None
    scores = _compare(gold, predicted)
    assert scores["entity.discipline"]["f1"] == 1
    assert scores[f"relation.discipline_{field}"]["fn"] == 1
    assert scores["record.discipline_complete"]["f1"] < 1


def test_swapped_hours_are_errors_even_with_same_sum(examples) -> None:
    gold, predicted = examples
    hours = predicted["entries"][0]["contact_hours"]
    hours["лекции"], hours["семинары"] = hours["семинары"], hours["лекции"]
    scores = _compare(gold, predicted)
    assert scores["entity.discipline"]["f1"] == 1
    assert scores["record.discipline_complete"]["tp"] == 3
    assert scores["record.discipline_complete"]["fp"] == 1
    assert scores["record.discipline_complete"]["fn"] == 1


def test_missing_and_duplicate_entries_are_penalized(examples) -> None:
    gold, predicted = examples
    predicted["entries"].pop()
    predicted["entries"].append(copy.deepcopy(predicted["entries"][0]))
    scores = _compare(gold, predicted)
    assert len(predicted["entries"]) == len(gold["entries"])
    assert scores["entity.discipline"]["fp"] == 1
    assert scores["entity.discipline"]["fn"] == 1
    assert scores["record.discipline_complete"]["fp"] == 1


def test_sources_accept_dash_normalization_but_reject_wrong_rows(examples) -> None:
    gold, predicted = examples
    gold["entries"][0]["name"] = "Модуль – навигация"
    predicted["entries"][0]["name"] = "Модуль - навигация"
    plan = CurriculumPlan(file="synthetic.xlsx", entries=predicted["entries"])
    assert _sources(plan, gold)["accuracy"] == 1
    assert _sources(plan, gold)["literal_identity_matches"] == 3
    plan.entries[0].source.line += 1
    assert _sources(plan, gold)["correct"] == 3


def test_reference_reader_preserves_cached_zero_missing_cache_and_merges() -> None:
    out = io.BytesIO()
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    with ZipFile(out, "w") as z:
        z.writestr(
            "xl/workbook.xml",
            f"<workbook {ns} "
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="План" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships><Relationship Id="rId1" '
            'Target="worksheets/sheet1.xml"/></Relationships>',
        )
        z.writestr("xl/sharedStrings.xml", f"<sst {ns}><si><t>УК-1</t></si></sst>")
        z.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet {ns}><sheetData><row r="3">'
            '<c r="A3"><f>1-1</f><v>0</v></c><c r="B3"><f>1+1</f></c>'
            '<c r="C3" t="s"><v>0</v></c><c r="E3" t="e"><v>#DIV/0!</v></c>'
            '</row></sheetData><mergeCells><mergeCell ref="C3:D4"/>'
            "</mergeCells></worksheet>",
        )
    sheet = read_xlsx_xml(out.getvalue())[0]
    assert sheet.value(3, 1) == 0
    assert sheet.value(3, 2) is None
    assert len(sheet.formulas) == 2
    assert sheet.value(4, 4, inherited=True) == "УК-1"
    assert sheet.errors == {(3, 5): "#DIV/0!"}


def test_reference_cannot_be_applied_to_another_source() -> None:
    with pytest.raises(ValueError, match="нет проверенной разметки"):
        build_plan_reference("другая версия".encode())


def test_missing_local_corpus_is_failure_not_empty_success(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    with pytest.raises(SystemExit) as exc:
        main(["--data-dir", str(tmp_path), "--output", str(target)])
    assert exc.value.code == 2
    assert not target.exists()
