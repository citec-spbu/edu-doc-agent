"""Федеральные акты и профстандарты в базе (#50): новые типы, точный поиск по статьям и кодам."""

import asyncio

import pytest
from sqlalchemy import text

from edu_doc_agent.dao.regulations import RegulationDAO
from edu_doc_agent.llm import FakeEmbedder
from edu_doc_agent.retrieval.corpus.load import load_documents
from edu_doc_agent.retrieval.corpus.models import CorpusDocument, DocumentInfo, Fragment
from edu_doc_agent.retrieval.search import RegulationSearch
from edu_doc_agent.retrieval.tools import RegulationTools
from tests.integration.dao.support import Database

pytestmark = pytest.mark.integration

FGOS, LAW, ORDER, PS = "fgos-02.03.02", "273-fz", "order-245", "ps-06.001"


def document(doc_id: str, doc_type: str, direction: str | None, fragments: list[dict]):
    info = DocumentInfo(
        doc_id=doc_id,
        doc_type=doc_type,
        title=doc_id,
        edition="1",
        direction_code=direction,
        source_file=f"{doc_id}.pdf",
    )
    common = {"doc_type": doc_type, "doc_id": doc_id, "edition": "1"}
    return CorpusDocument(document=info, fragments=[Fragment(**common, **f) for f in fragments])


def clause(doc_id: str, number: str, text: str, **extra) -> dict:
    return {"id": f"{doc_id}:{number}", "kind": "clause", "number": number,
            "path": f"Раздел › {number}", "text": text, **extra}  # fmt: skip


CORPUS = [
    document(FGOS, "fgos", "02.03.02", [clause(FGOS, "1.9", "1.9. Объем программы 240 з.е.")]),
    document(
        LAW,
        "federal_act",
        None,
        [
            clause(LAW, f"ст.13:ч.{n}", f"{n}. Часть {n} статьи о реализации программ.")
            for n in (1, 2, 10)
        ]
        + [clause(LAW, "ст.2:п.9", "9) образовательная программа - комплекс характеристик")],
    ),
    document(ORDER, "federal_act", None, [clause(ORDER, "15", "15. Трудоемкость программы.")]),
    document(
        PS,
        "prof_standard",
        None,
        [
            {
                "id": f"{PS}:general",
                "kind": "prof_standard",
                "number": "1",
                "path": "Программист › I",
                "text": "Вид профессиональной деятельности: разработка ПО",
                "codes": ["06.001"],
            },
            {
                "id": f"{PS}:D",
                "kind": "generalized_function",
                "number": "3.4",
                "path": "Программист › 3.4",
                "text": "Обобщенная трудовая функция D «Проектирование»",
                "codes": ["06.001:D"],
            },
            {
                "id": f"{PS}:D/03.6",
                "kind": "labor_function",
                "number": "3.4.3",
                "path": "Программист › 3.4.3",
                "text": "Трудовая функция D/03.6 «Проектирование программного обеспечения»",
                "codes": ["06.001:D/03.6"],
            },
        ],
    ),
]


@pytest.fixture
def tools(dao: RegulationDAO) -> RegulationTools:
    asyncio.run(load_documents(dao, CORPUS, FakeEmbedder()))
    return asyncio.run(
        RegulationTools.for_program(RegulationSearch(dao, FakeEmbedder()), "02.03.02")
    )


def run(coro):
    return asyncio.run(coro)


def test_new_types_pass_table_constraints(migrated_database: Database, tools) -> None:
    with migrated_database.engine.connect() as connection:
        kinds = set(connection.scalars(text("SELECT DISTINCT kind FROM regulation_fragments")))
        types = set(connection.scalars(text("SELECT DISTINCT doc_type FROM regulation_documents")))
    assert {"prof_standard", "generalized_function", "labor_function"} <= kinds
    assert "federal_act" in types
    assert tools.doc_ids == [LAW, FGOS, ORDER, PS]  # акты и профстандарты — для любой программы


def test_article_parts_are_found_exactly_in_number_order(tools) -> None:
    hits = run(tools.search_federal_acts("что сказано в статье 13?"))
    exact = [h.id for h in hits if h.method == "exact"]
    assert exact == [f"{LAW}:ст.13:ч.1", f"{LAW}:ст.13:ч.2", f"{LAW}:ст.13:ч.10"]
    (part, *_) = run(tools.search_federal_acts("ч. 10 ст. 13"))
    assert (part.id, part.method) == (f"{LAW}:ст.13:ч.10", "exact")
    (item, *_) = run(tools.search_federal_acts("п. 9 ст. 2"))
    assert (item.id, item.method) == (f"{LAW}:ст.2:п.9", "exact")
    (point, *_) = run(tools.search_all_regulations("п. 15 Порядка"))
    assert (point.id, point.method) == (f"{ORDER}:15", "exact")


def test_labor_functions_are_found_by_code(tools) -> None:
    (function, *_) = run(tools.search_prof_standards("ТФ D/03.6 профстандарта 06.001"))
    assert (function.id, function.kind, function.method) == (
        f"{PS}:D/03.6",
        "labor_function",
        "exact",
    )
    (generalized, *_) = run(tools.search_prof_standards("06.001 ОТФ D"))
    assert (generalized.id, generalized.method) == (f"{PS}:D", "exact")
