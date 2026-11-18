"""Источники полей новых документов позволяют вернуться к исходной строке."""

from edu_doc_agent.extraction.characteristic import extract_characteristic
from edu_doc_agent.extraction.ckmv import extract_ckmv
from edu_doc_agent.extraction.curriculum import extract_plan
from edu_doc_agent.extraction.matrix import extract_matrix
from tests.unit.test_extraction_characteristic import MD as CHAR_MD
from tests.unit.test_extraction_ckmv import MD as CKMV_MD
from tests.unit.test_extraction_curriculum import PLAN
from tests.unit.test_extraction_matrix import GOOD, _xlsx


def test_plan_title_and_rows_point_to_source():
    plan = extract_plan(PLAN, "plan.md")
    for name in (
        "program",
        "direction",
        "level",
        "study_form",
        "duration",
        "registration_number",
    ):
        ref = plan.sources[name]
        assert ref.file == "plan.md" and ref.line is not None
        assert ref.quote == PLAN.splitlines()[ref.line - 1]
        assert getattr(plan, name) in ref.quote
    for item in [*plan.competences, *plan.entries, *plan.final_attestation]:
        assert item.source.quote == PLAN.splitlines()[item.source.line - 1]


def test_characteristic_title_and_qualification_point_to_source():
    doc = extract_characteristic(CHAR_MD, "char.md")
    fields = (
        "group_code",
        "group_name",
        "direction_code",
        "direction",
        "level",
        "profile",
        "profile_en",
        "program_name",
        "program_name_en",
        "program_code",
        "study_form",
        "languages",
        "duration",
        "annotation",
        "mission",
    )
    assert set(doc.sources) == set(fields)
    for name in fields:
        ref = doc.sources[name]
        assert ref.file == "char.md" and ref.quote == CHAR_MD.splitlines()[ref.line - 1]
        assert getattr(doc, name) in ref.quote
    assert len(doc.qualification_sources) == len(doc.qualification)
    for value, ref in zip(doc.qualification, doc.qualification_sources, strict=True):
        assert value in ref.quote


def test_ckmv_entities_point_to_raw_rows():
    doc = extract_ckmv(CKMV_MD, "ckmv.md")
    for item in [*doc.competences, *doc.indicators, *doc.outcomes]:
        assert item.source.quote == CKMV_MD.splitlines()[item.source.line - 1]


def test_matrix_columns_rows_and_totals_have_sources():
    matrix = extract_matrix(_xlsx(GOOD), "matrix.xlsx")
    for column in matrix.columns:
        assert column.source.file == "matrix.xlsx"
        assert column.source.quote == column.code and column.source.line == 2
    for row in matrix.rows:
        assert row.discipline in row.source.quote
    assert matrix.total_sources.keys() == matrix.totals.keys()
    assert all("Итого" in ref.quote for ref in matrix.total_sources.values())
