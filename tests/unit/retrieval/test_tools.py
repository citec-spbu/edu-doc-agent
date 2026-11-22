"""Контракт инструментов RAG для агентов на заглушке в памяти (без базы и модели)."""

import asyncio
import inspect
import json

from edu_doc_agent.retrieval.corpus.models import CorpusDocument, DocumentInfo, Fragment
from edu_doc_agent.retrieval.fake import InMemoryRegulationSearch
from edu_doc_agent.retrieval.tools import RegulationTools

FGOS, OTHER, OS = "fgos-02.03.02", "fgos-01.03.02", "spbu-os"


def doc(doc_id: str, doc_type: str, direction: str | None, fragments: list[dict]):
    common = {"doc_type": doc_type, "doc_id": doc_id, "edition": "1"}
    info = DocumentInfo(
        doc_id=doc_id,
        doc_type=doc_type,
        title=doc_id,
        edition="1",
        direction_code=direction,
        source_file="x.pdf",
    )
    return CorpusDocument(document=info, fragments=[Fragment(**common, **f) for f in fragments])


def clause(doc_id: str, number: str, text: str, **extra) -> dict:
    return {"id": f"{doc_id}:{number}", "kind": "clause", "number": number,
            "path": f"Раздел › {number}", "text": text, **extra}  # fmt: skip


CORPUS = [
    doc(
        FGOS,
        "fgos",
        "02.03.02",
        [
            clause(FGOS, "1.9", "1.9. Объем программы бакалавриата составляет 240 з.е."),
            {
                "id": f"{FGOS}:3.2:УК-1",
                "kind": "competency",
                "number": "3.2",
                "path": "III › 3.2 › УК-1",
                "text": "УК-1. Способен осуществлять поиск информации",
                "codes": ["УК-1"],
            },
            {
                "id": f"{FGOS}:app:06.001",
                "kind": "prof_standard_ref",
                "number": "app",
                "path": "Приложение › 06.001",
                "text": 'Профессиональный стандарт "Программист"',
                "codes": ["06.001"],
            },
        ],
    ),
    doc(OTHER, "fgos", "01.03.02", [clause(OTHER, "1.9", "1.9. Объем программы 240 з.е.")]),
    doc(OS, "spbu_standard", None, [clause(OS, "2.4", "2.4. Объем программы — в з.е.")]),
]


def run(coro):
    return asyncio.run(coro)


def tools() -> RegulationTools:
    return run(RegulationTools.for_program(InMemoryRegulationSearch(CORPUS), "02.03.02"))


def test_program_scope_is_set_by_code_not_by_llm() -> None:
    t = tools()
    assert t.doc_ids == [FGOS, OS]
    hits = run(t.search_all_regulations("объем программы"))
    assert {h.doc_id for h in hits} == {FGOS, OS}  # чужой ФГОС с тем же текстом не виден
    # у инструментов поиска нет аргумента doc_ids — его не может подставить LLM
    params = inspect.signature(t.search_all_regulations).parameters
    assert list(params) == ["query", "top_k"]


def test_each_tool_searches_its_documents() -> None:
    t = tools()
    assert {h.doc_type for h in run(t.search_fgos("объем программы"))} == {"fgos"}
    assert {h.doc_id for h in run(t.search_spbu_regulations("объем программы"))} == {OS}
    (ps,) = run(t.search_prof_standards("программист"))
    assert ps.id == f"{FGOS}:app:06.001" and ps.kind == "prof_standard_ref"


def test_code_is_found_exactly_and_clause_read_by_reference() -> None:
    t = tools()
    (hit,) = run(t.search_fgos("что требует УК-1", top_k=3))
    assert (hit.id, hit.method) == (f"{FGOS}:3.2:УК-1", "exact")
    clause_hit = run(t.get_fgos_clause(hit.id, hit.edition))
    assert clause_hit is not None and clause_hit.text == hit.text
    assert run(t.get_spbu_clause(f"{OS}:9.9", "1")) is None


def test_tools_are_named_as_in_adr_documented_and_json_serializable() -> None:
    functions = tools().as_functions()
    assert set(functions) == {
        "search_fgos",
        "search_spbu_regulations",
        "search_prof_standards",
        "search_all_regulations",
        "get_fgos_clause",
        "get_spbu_clause",
    }
    assert all(inspect.iscoroutinefunction(f) and f.__doc__ for f in functions.values())
    hits = run(functions["search_all_regulations"]("объем программы"))
    payload = json.dumps([h.model_dump() for h in hits], ensure_ascii=False)
    assert '"rank": 1' in payload and '"method": "fts"' in payload


def test_package_exports() -> None:
    import edu_doc_agent.retrieval as retrieval

    assert retrieval.RegulationTools is RegulationTools
    assert retrieval.InMemoryRegulationSearch is InMemoryRegulationSearch
