"""Подграф «Вопросы»: Q&A-агент с инструментами сессии и поиска по нормативам."""

from graph_helpers import Models, call, make_graph, rpd, run
from langchain_core.messages import ToolMessage

from edu_doc_agent.orchestration import make_input


def test_qa_uses_session_and_regulation_tools() -> None:
    asked: list[tuple[str | None, str]] = []

    async def regulations(direction_code):
        async def search_all_regulations(query: str, top_k: int | None = None) -> list[dict]:
            """Поиск по нормативам программы."""
            asked.append((direction_code, query))
            return [{"id": "fgos-02.03.02:2.1", "edition": "808", "text": "Блок 2 | Практика"}]

        return {"search_all_regulations": search_all_regulations}

    models = Models(
        qa=[
            call("find_course", query="Физика"),
            call("search_all_regulations", query="объём практики"),
            call("QAAnswer", answer="Практика — не менее 20 з.е. (fgos-02.03.02:2.1)."),
        ]
    )
    graph, _ = make_graph(models, {"a.md": ("rpd", rpd(title="Физика"))}, regulations=regulations)

    state, nodes = run(
        graph, make_input("r1", "какой объём практики?", action="qa", file_paths=["a.md"])
    )

    assert nodes == ["start", "ingest", "orchestrator", "qa", "remember"]
    assert state.values["reply"] == "Практика — не менее 20 з.е. (fgos-02.03.02:2.1)."
    assert asked == [("02.03.02", "объём практики")]  # код направления из документов
    request = models.calls("qa")[0][0].content  # данные сессии — в системном промпте
    # документы сессии с полями: агент выбирает документ и поле, а не ищет наугад
    assert "- doc:a.md — РПД/РПП — Физика: doc_kind, title, credits, competences" in request
    assert "Правила проверки:" not in request  # перечня правил нет: требования — по нормативам
    history = models.calls("qa")[2]
    found, regulation = history[-3], history[-1]
    assert isinstance(found, ToolMessage) and "doc:a.md" in found.content
    assert isinstance(regulation, ToolMessage) and "Блок 2 | Практика" in regulation.content


def test_qa_without_regulations_component() -> None:
    models = Models(qa=[call("QAAnswer", answer="ok")])
    graph, _ = make_graph(models)

    state, _ = run(graph, make_input("r1", "?", action="qa"))

    assert state.values["reply"] == "ok"
