"""Хранилище сессии, инструменты агентов и извлечение документов."""

import asyncio
from pathlib import Path

import pytest
from graph_helpers import SRC, entry, plan, rpd
from langgraph.store.memory import InMemoryStore

from edu_doc_agent.extraction.matrix import CompetenceMatrix, MatrixRow
from edu_doc_agent.orchestration import tools as tools_module
from edu_doc_agent.orchestration.ingest import (
    ExtractionIngestor,
    detect_doc_type,
    document_id,
    original_filename,
)
from edu_doc_agent.orchestration.stores import SessionStores, StoredDocument
from edu_doc_agent.orchestration.tools import SessionTools
from edu_doc_agent.validation.findings import Finding

FIXTURES = Path(__file__).parents[2] / "fixtures"
TEXT = """\
# Раздел 1. Характеристики

## 1.1. Цели и задачи

Цель — научить решать дифференциальные уравнения.

## 1.2. Требования к подготовленности

Математический анализ.
"""


def finding(fid: str, **kw) -> Finding:
    values = dict(
        id=fid,
        session_id="s1",
        run_id="r1",
        rule_id="rule",
        rule_title="Правило",
        rule_set="cross_check",
        severity="error",
        fixable=True,
        document_id="d1",
        course="Физика",
        status="violation",
        message="нарушено",
    )
    return Finding(**(values | kw))


@pytest.fixture
def stores() -> SessionStores:
    stores = SessionStores(InMemoryStore())

    async def fill():
        await stores.save_document(
            StoredDocument("d1", "s1", "rpd", rpd(title="Физика"), "f.md", TEXT)
        )
        await stores.save_document(StoredDocument("d2", "s1", "rpd", rpd(), "m.md"))
        await stores.save_document(StoredDocument("x1", "s2", "rpd", rpd(), "x.md"))
        await stores.save_findings("s1", [finding("f1"), finding("f2", status="not_checked")])

    asyncio.run(fill())
    return stores


def tools(stores: SessionStores) -> SessionTools:
    return SessionTools(stores, "s1", [], ["f1", "f2"], [], "02.03.02")


def test_store_round_trip_and_sessions(stores: SessionStores) -> None:
    documents = asyncio.run(stores.list_documents("s1"))

    assert [d.document_id for d in documents] == ["d1", "d2"]
    assert documents[0].data.title == "Физика"
    assert documents[0].text == TEXT
    assert [f.id for f in asyncio.run(stores.get_findings("s1", ["f2", "nope", "f1"]))] == [
        "f2",
        "f1",
    ]
    assert asyncio.run(stores.get_findings("s2", ["f1"])) == []


def test_current_state_and_findings_filters(stores: SessionStores) -> None:
    t = tools(stores)
    state = asyncio.run(t.get_current_state())

    assert [d["title"] for d in state["documents"]] == ["Физика", "Математический анализ"]
    assert state["open_findings"] == {"violation/error": 1, "not_checked/error": 1}
    assert [f["id"] for f in asyncio.run(t.get_findings(status="violation"))] == ["f1"]


def test_find_and_read_course(stores: SessionStores) -> None:
    t = tools(stores)

    (best, *_) = asyncio.run(t.find_course("физике"))
    assert best["course_id"] == "d1"
    # что можно запросить в get_course; у списков — подполя элементов
    assert "credits" in best["fields"] and "competences[code, name, group]" in best["fields"]
    data = asyncio.run(t.get_course("d1", fields=["credits", "competences"]))
    assert data["data"]["credits"] == 4
    assert data["data"]["competences"][0] == {"code": "УК-1", "source": "2.1, строка 10"}
    with pytest.raises(ValueError, match="нет полей"):
        asyncio.run(t.get_course("d1", fields=["кредиты"]))
    codes = asyncio.run(t.get_course("d1", fields=["competences.code"]))
    assert codes["data"]["competences"][0] == {"code": "УК-1"}  # без тяжёлых подполей
    with pytest.raises(ValueError, match="нет подполей"):
        asyncio.run(t.get_course("d1", fields=["competences.role"]))


def test_plan_entries_filtered(stores: SessionStores) -> None:
    curriculum = plan(
        entry("Физика", "001863", 3, ["УК-1"], semester=3, profile="Профиль СТАНДАРТ"),
        entry("Учебная практика (НИР)", "064683", 5, ["УК-1"], block="Блок.2..прки"),
        entry("Производственная практика", "064684", 7, [], block="Блок.2..прки", semester=8),
    )
    asyncio.run(stores.save_document(StoredDocument("p1", "s1", "plan", curriculum, "plan.md")))
    t = tools(stores)

    practices = asyncio.run(t.get_plan_entries(block="Блок 2"))
    assert practices["found"] == 2
    assert practices["credits_total"] == 12
    assert practices["entries"][1] == {
        "name": "Производственная практика",
        "block": "Блок.2..прки",
        "credits": 7,
        "semester": 8,
    }
    physics = asyncio.run(t.get_plan_entries(name="физика", profile="стандарт"))
    assert [e["credits"] for e in physics["entries"]] == [3]


def test_matrix_rows_filtered(stores: SessionStores) -> None:
    def matrix_row(n: int, discipline: str, competences: list[str]) -> MatrixRow:
        return MatrixRow(number=n, discipline=discipline, competences=competences, source=SRC)

    matrix = CompetenceMatrix(
        file="m.xlsx",
        rows=[matrix_row(1, "Физика", ["УК-1", "УНК-03"]), matrix_row(2, "БЖД", ["УК-8"])],
        totals={"ТОП": {"УК-1": 1, "УК-8": 1}},
    )
    asyncio.run(stores.save_document(StoredDocument("m1", "s1", "matrix", matrix, "m.xlsx")))
    t = tools(stores)

    uk8 = asyncio.run(t.get_matrix_rows(competence="ук-8"))
    assert [r["discipline"] for r in uk8["rows"]] == ["БЖД"]
    physics = asyncio.run(t.get_matrix_rows(discipline="физика"))
    assert physics["rows"][0]["competences"] == ["УК-1", "УНК-03"]
    totals = asyncio.run(t.get_matrix_rows())
    assert totals["disciplines_per_competence"] == {"ТОП": {"УК-1": 1, "УК-8": 1}}


def test_course_lists_filtered_and_paged(
    stores: SessionStores, monkeypatch: pytest.MonkeyPatch
) -> None:
    curriculum = plan(*(entry(f"Дисциплина {i}", f"{i:06}", 3, ["УК-1"]) for i in range(30)))
    asyncio.run(stores.save_document(StoredDocument("p1", "s1", "plan", curriculum, "plan.md")))
    t = tools(stores)
    monkeypatch.setattr(tools_module, "MAX_COURSE_DATA", 1500)

    first = asyncio.run(t.get_course("p1", fields=["entries"]))
    page = first["lists"]["entries"]
    assert page["total"] == 30 and 0 < page["returned"] < 30
    assert page["next_offset"] == page["returned"]
    rest = asyncio.run(t.get_course("p1", fields=["entries"], offset=page["next_offset"]))
    assert rest["data"]["entries"][0]["name"] == f"Дисциплина {page['returned']}"
    found = asyncio.run(t.get_course("p1", fields=["entries"], match="дисциплина 7"))
    assert [e["name"] for e in found["data"]["entries"]] == ["Дисциплина 7"]
    assert found["lists"]["entries"] == {"total": 1, "returned": 1, "next_offset": None}


def test_outline_read_and_search(stores: SessionStores) -> None:
    t = tools(stores)

    outline = asyncio.run(t.get_document_outline("d1"))
    assert [s["section_id"] for s in outline] == [
        "Раздел 1. Характеристики",
        "1.1",
        "1.2",
    ]
    section = asyncio.run(t.read_section("d1", "1.1"))
    assert "дифференциальные уравнения" in section["text"]
    (hit, *_) = asyncio.run(t.search_document("дифференциальных уравнений решать"))
    assert hit["section_id"] == "1.1"


def test_detect_doc_type() -> None:
    assert detect_doc_type("### Р А Б О Ч А Я П Р О Г Р А М М А\n### учебной дисциплины") == "rpd"
    assert detect_doc_type("### ОБЩАЯ ХАРАКТЕРИСТИКА\nосновной программы") == "characteristic"
    assert detect_doc_type("### компетентностно-ориентированный учебный план") == "plan"
    assert detect_doc_type("Перечень выбранных профессиональных ролей и компетенций") == "ckmv"
    assert detect_doc_type("Протокол заседания") is None


def test_storage_file_names() -> None:
    path = "/storage/sessions/abc/~" + "0" * 64 + "_РПД Физика.docx"

    assert original_filename(path) == "РПД Физика.docx"
    assert document_id("s1", path) == document_id("s1", path) != document_id("s2", path)


def test_ingest_markdown_rpd() -> None:
    document = asyncio.run(ExtractionIngestor().ingest("s1", str(FIXTURES / "rpd_synthetic.md")))

    assert document.doc_type == "rpd"
    assert document.filename == "rpd_synthetic.md"
    assert document.data.title == "Синтетическая дисциплина"
    assert document.text.startswith("### Университет")
