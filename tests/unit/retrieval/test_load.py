"""Загрузчик корпуса без базы: строки для эмбеддинга и таблиц, отбор документов."""

from edu_doc_agent.retrieval.corpus.load import (
    _loadable,
    document_row,
    embedding_input,
    fragment_row,
)
from edu_doc_agent.retrieval.corpus.models import ChangeNote, CorpusDocument, DocumentInfo, Fragment

INFO = DocumentInfo(
    doc_id="spbu-os",
    doc_type="spbu_standard",
    title="ОС СПбГУ",
    edition="11144/1 от 24.08.2026",
    approval_act="Приказ № 11144/1",
    source_file="os.pdf",
)
FRAGMENT = Fragment(
    id="spbu-os:2.1",
    doc_type="spbu_standard",
    doc_id="spbu-os",
    edition=INFO.edition,
    kind="clause",
    number="2.1",
    path="2. Требования к структуре › 2.1",
    text="2.1. Образовательная программа включает:",
    changed_by=[ChangeNote(act="Приказ № 1")],
)


def test_embedding_input_is_path_and_text() -> None:
    assert embedding_input(FRAGMENT) == (
        "2. Требования к структуре › 2.1\n2.1. Образовательная программа включает:"
    )


def test_rows_match_table_columns() -> None:
    row = fragment_row(FRAGMENT)
    assert row["changed_by"] == [{"since": None, "act": "Приказ № 1"}]
    assert row["codes"] == [] and row["part"] == 0 and "embedding" not in row
    doc = document_row(CorpusDocument(document=INFO, fragments=[FRAGMENT]))
    assert doc == {
        "doc_id": "spbu-os",
        "edition": "11144/1 от 24.08.2026",
        "doc_type": "spbu_standard",
        "title": "ОС СПбГУ",
        "approval_act": "Приказ № 11144/1",
        "direction_code": None,
        "level": None,
        "source_file": "os.pdf",
    }


def test_documents_with_build_errors_are_not_loaded() -> None:
    good = CorpusDocument(document=INFO, fragments=[FRAGMENT], warnings=["номер не по порядку"])
    bad = CorpusDocument(
        document=INFO.model_copy(update={"doc_id": "broken"}),
        fragments=[],
        warnings=["ОШИБКА: id spbu-os:2.1 повторяется"],
    )
    loadable, errors = _loadable([good, bad])
    assert loadable == [good]
    assert errors == ["broken: ОШИБКА: id spbu-os:2.1 повторяется"]
