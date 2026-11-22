"""Поиск по нормативам на PostgreSQL 17 + pgvector: точный, гибридный, полнотекстовый."""

import asyncio

import pytest

from edu_doc_agent.dao.regulations import RegulationDAO
from edu_doc_agent.llm import FakeEmbedder
from edu_doc_agent.retrieval.corpus.load import load_documents
from edu_doc_agent.retrieval.corpus.models import CorpusDocument, DocumentInfo, Fragment
from edu_doc_agent.retrieval.search import RegulationSearch
from edu_doc_agent.retrieval.tools import RegulationTools

pytestmark = pytest.mark.integration


def document(doc_id: str, doc_type: str, direction: str | None, fragments: list[dict]):
    common = {"doc_type": doc_type, "doc_id": doc_id, "edition": "1"}
    info = DocumentInfo(
        doc_id=doc_id,
        doc_type=doc_type,
        title=doc_id,
        edition="1",
        direction_code=direction,
        source_file=f"{doc_id}.pdf",
    )
    return CorpusDocument(document=info, fragments=[Fragment(**common, **f) for f in fragments])


def clause(doc_id: str, number: str, text: str, **extra) -> dict:
    return {
        "id": f"{doc_id}:{number}",
        "kind": "clause",
        "number": number,
        "path": f"Раздел › {number}",
        "text": text,
        **extra,
    }


FGOS, OTHER, OS = "fgos-02.03.02", "fgos-01.03.02", "spbu-os"
CORPUS = [
    document(
        FGOS,
        "fgos",
        "02.03.02",
        [
            clause(FGOS, "1.9", "1.9. Объем программы бакалавриата составляет 240 зачетных единиц"),
            clause(
                FGOS,
                "3.2",
                "3.2. Программа должна устанавливать универсальные компетенции: УК-1",
                codes=["УК-1"],
            ),
            {
                "id": f"{FGOS}:3.2:УК-1",
                "kind": "competency",
                "number": "3.2",
                "path": "III › 3.2 › УК-1",
                "text": "УК-1. Способен осуществлять поиск, критический анализ и синтез информации",
                "codes": ["УК-1"],
                "parent": f"{FGOS}:3.2",
                "category": "Системное и критическое мышление",
            },
            clause(FGOS, "4.1", "4.1. Требования к материально-техническому ", part=0),
            clause(FGOS, "4.1", "обеспечению программы бакалавриата.", part=1),
        ],
    ),
    document(
        OTHER,
        "fgos",
        "01.03.02",
        [clause(OTHER, "1.9", "1.9. Объем программы бакалавриата составляет 240 зачетных единиц")],
    ),
    document(
        OS,
        "spbu_standard",
        None,
        [
            clause(
                OS, "2.3", "2.3. Образовательная программа содержит рабочие программы дисциплин"
            ),
            clause(OS, "2.4", "2.4. рабочие программы курсов и практик, содержащие:"),
            clause(OS, "2.4.1", "2.4.1. цели и задачи;"),
            clause(OS, "2.4.10", "2.4.10. фонд оценочных средств."),
            clause(OS, "2.4.2", "2.4.2. требования к условиям реализации;"),
        ],
    ),
]


@pytest.fixture
def search(dao: RegulationDAO) -> RegulationSearch:
    asyncio.run(load_documents(dao, CORPUS, FakeEmbedder()))
    return RegulationSearch(dao, FakeEmbedder())


def run(coro):
    return asyncio.run(coro)


def test_code_is_found_exactly_with_definition_first(search: RegulationSearch) -> None:
    hits = run(search.search("что требует ук-1?", doc_ids=[FGOS, OS]))
    assert [(h.id, h.method) for h in hits] == [
        (f"{FGOS}:3.2:УК-1", "exact"),  # сама компетенция раньше пункта, где она упомянута
        (f"{FGOS}:3.2", "exact"),
    ]
    assert hits[0].category == "Системное и критическое мышление"


def test_clause_number_with_document_filter(search: RegulationSearch) -> None:
    hits = run(search.search("что сказано в п. 2.3", doc_ids=[FGOS, OS]))
    assert [(h.id, h.method) for h in hits] == [(f"{OS}:2.3", "exact")]


def test_hybrid_respects_doc_ids_and_joins_parts(search: RegulationSearch) -> None:
    hits = run(search.search("объем программы бакалавриата", doc_ids=[FGOS, OS], top_k=3))
    # порядок задаёт в основном векторный список, а здесь он от FakeEmbedder (мешок слов),
    # поэтому проверяем попадание, а не первое место
    assert f"{FGOS}:1.9" in [h.id for h in hits] and {h.method for h in hits} == {"hybrid"}
    assert all(h.doc_id in {FGOS, OS} for h in hits)  # чужой ФГОС с тем же текстом отсечён
    (clause_4_1,) = [
        h
        for h in run(search.search("материально-техническое обеспечение"))
        if h.id == f"{FGOS}:4.1"
    ]
    assert clause_4_1.text == (
        "4.1. Требования к материально-техническому обеспечению программы бакалавриата."
    )


def test_tools_for_program(search: RegulationSearch) -> None:
    tools = run(RegulationTools.for_program(search, "02.03.02"))
    assert tools.doc_ids == [FGOS, OS]
    assert {h.doc_type for h in run(tools.search_fgos("программы"))} == {"fgos"}
    assert {h.doc_id for h in run(tools.search_spbu_regulations("программы"))} == {OS}
    assert {h.doc_id for h in run(tools.search_all_regulations("программы"))} <= {FGOS, OS}


def test_full_text_only_without_embedder(dao: RegulationDAO, search: RegulationSearch) -> None:
    hits = run(RegulationSearch(dao).search("рабочие программы дисциплин"))
    assert hits[0].id == f"{OS}:2.3" and hits[0].method == "fts"


def test_get_clause_and_documents_for_direction(search: RegulationSearch) -> None:
    clause_hit = run(search.get_clause(f"{FGOS}:4.1", "1"))
    assert clause_hit is not None and clause_hit.text.endswith("программы бакалавриата.")
    assert run(search.get_clause(f"{FGOS}:4.1", "нет такой")) is None
    assert run(search.documents_for_direction("02.03.02")) == [FGOS, OS]


def test_list_item_returns_whole_list_in_document_order(search: RegulationSearch) -> None:
    hits = run(search.search("фонд оценочных средств", doc_ids=[OS]))
    lists = [h for h in hits if h.covers]
    assert [h.id for h in lists] == [f"{OS}:2.4"]  # подпункты не повторяются отдельно
    assert lists[0].covers == [f"{OS}:2.4", f"{OS}:2.4.1", f"{OS}:2.4.2", f"{OS}:2.4.10"]
    assert lists[0].text.splitlines()[0] == "2.4. рабочие программы курсов и практик, содержащие:"
    assert not any(h.id.startswith(f"{OS}:2.4.") for h in hits)
