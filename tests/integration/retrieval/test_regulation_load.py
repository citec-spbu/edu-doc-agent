"""Таблицы нормативов и загрузчик корпуса на PostgreSQL 17 + pgvector (ADR-0005)."""

import asyncio

import pytest
from sqlalchemy import Engine, func, select, text
from sqlalchemy.exc import IntegrityError

from edu_doc_agent.dao.models import RegulationDocument, RegulationFragment
from edu_doc_agent.dao.regulations import RegulationDAO
from edu_doc_agent.llm import FakeEmbedder
from edu_doc_agent.retrieval.corpus.load import embedding_input, load_documents
from edu_doc_agent.retrieval.corpus.models import (
    ChangeNote,
    CorpusDocument,
    DocumentInfo,
    Fragment,
)
from tests.integration.dao.support import Database

pytestmark = pytest.mark.integration


def load(dao: RegulationDAO, documents, embedder, **kwargs):
    return asyncio.run(load_documents(dao, documents, embedder, **kwargs))


def corpus(edition: str = "808 ред. от 08.02.2021", volume: str = "240") -> CorpusDocument:
    doc_id = "fgos-02.03.02"
    common = {"doc_type": "fgos", "doc_id": doc_id, "edition": edition}
    long_text = "Организация обеспечивает условия. " * 3
    fragments = [
        Fragment(
            id=f"{doc_id}:1.9",
            kind="clause",
            number="1.9",
            path="I. Общие положения › 1.9",
            text=f"1.9. Объем программы бакалавриата составляет {volume} зачетных единиц",
            changed_by=[ChangeNote(since="с 1 сентября 2021 г.", act="Приказ N 1456")],
            **common,
        ),
        Fragment(
            id=f"{doc_id}:3.2:УК-1",
            kind="competency",
            number="3.2",
            path="III. Требования к результатам › 3.2 › УК-1",
            text="УК-1. Способен осуществлять поиск, критический анализ и синтез информации",
            codes=["УК-1"],
            parent=f"{doc_id}:3.2",
            category="Системное и критическое мышление",
            **common,
        ),
        *(
            Fragment(
                id=f"{doc_id}:4.1",
                kind="clause",
                number="4.1",
                path="IV. Требования к условиям › 4.1",
                part=part,
                text=chunk,
                **common,
            )
            for part, chunk in enumerate([long_text[:40], long_text[40:]])
        ),
    ]
    info = DocumentInfo(
        doc_id=doc_id,
        doc_type="fgos",
        title="ФГОС ВО — бакалавриат по направлению 02.03.02",
        edition=edition,
        direction_code="02.03.02",
        level="бакалавриат",
        source_file="020302.pdf",
    )
    return CorpusDocument(document=info, fragments=fragments)


def ids(engine: Engine, where: str, **params: object) -> list[str]:
    query = text(f"SELECT id FROM regulation_fragments WHERE {where} ORDER BY id, part")
    with engine.connect() as connection:
        return list(connection.execute(query, params).scalars())


def test_load_and_search_primitives(migrated_database: Database, dao: RegulationDAO) -> None:
    engine, embedder = migrated_database.engine, FakeEmbedder()
    result = load(dao, [corpus()], embedder)
    assert result.loaded == ["fgos-02.03.02"] and result.embedded == 4

    # полнотекстовый поиск по генерируемой колонке tsv (морфология русского)
    assert ids(engine, "tsv @@ plainto_tsquery('russian', :q)", q="объём программ") == [
        "fgos-02.03.02:1.9"
    ]
    # точный поиск по коду
    assert ids(engine, "codes @> ARRAY[:code]::varchar[]", code="УК-1") == [
        "fgos-02.03.02:3.2:УК-1"
    ]
    # векторный поиск: ближайший к эмбеддингу запроса
    (query,) = FakeEmbedder().embed(["поиск и критический анализ информации"])
    with engine.connect() as connection:
        nearest = connection.scalar(
            select(RegulationFragment.id)
            .order_by(RegulationFragment.embedding.cosine_distance(query))
            .limit(1)
        )
        changed_by = connection.scalar(
            select(RegulationFragment.changed_by).where(
                RegulationFragment.id == "fgos-02.03.02:1.9"
            )
        )
    assert nearest == "fgos-02.03.02:3.2:УК-1"
    assert changed_by == [{"since": "с 1 сентября 2021 г.", "act": "Приказ N 1456"}]
    # склейка частей по порядку даёт исходный текст
    with engine.connect() as connection:
        parts = connection.execute(
            select(RegulationFragment.text)
            .where(RegulationFragment.id == "fgos-02.03.02:4.1")
            .order_by(RegulationFragment.part)
        ).scalars()
        assert "".join(parts) == "Организация обеспечивает условия. " * 3


def test_reload_of_same_corpus_writes_nothing(
    migrated_database: Database, dao: RegulationDAO
) -> None:
    engine = migrated_database.engine
    load(dao, [corpus()], FakeEmbedder())
    with engine.connect() as connection:
        loaded_at = connection.scalar(select(RegulationDocument.loaded_at))

    embedder = FakeEmbedder()
    result = load(dao, [corpus()], embedder)
    assert result.unchanged == ["fgos-02.03.02"] and result.loaded == []
    assert embedder.calls == []
    with engine.connect() as connection:
        assert connection.scalar(select(RegulationDocument.loaded_at)) == loaded_at


def test_new_edition_replaces_old_and_reuses_embeddings(
    migrated_database: Database, dao: RegulationDAO
) -> None:
    engine = migrated_database.engine
    load(dao, [corpus()], FakeEmbedder())

    embedder = FakeEmbedder()
    new = corpus(edition="808 ред. от 27.02.2023", volume="241")
    result = load(dao, [new], embedder)
    assert result.loaded == ["fgos-02.03.02"]
    # изменился только пункт 1.9 — остальные эмбеддинги взяты из базы
    assert embedder.calls == [[embedding_input(new.fragments[0])]]
    with engine.connect() as connection:
        editions = connection.execute(
            select(RegulationFragment.edition, func.count()).group_by(RegulationFragment.edition)
        ).all()
        documents = connection.execute(select(RegulationDocument.edition)).scalars().all()
    assert editions == [("808 ред. от 27.02.2023", 4)]
    assert documents == ["808 ред. от 27.02.2023"]


def test_prune_removes_documents_missing_from_corpus(
    migrated_database: Database, dao: RegulationDAO
) -> None:
    engine = migrated_database.engine
    load(dao, [corpus()], FakeEmbedder())
    result = load(dao, [], FakeEmbedder(), prune=True)
    assert result.removed == ["fgos-02.03.02"]
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(RegulationFragment)) == 0


def test_defaults_and_constraints(migrated_database: Database) -> None:
    """Пустые коллекции по умолчанию; закрытые списки и сироты отсекает база."""
    engine = migrated_database.engine
    document = {
        "doc_id": "spbu-os",
        "edition": "1",
        "doc_type": "spbu_standard",
        "title": "ОС",
        "source_file": "os.pdf",
    }
    fragment = {
        "id": "spbu-os:1.1",
        "edition": "1",
        "doc_type": "spbu_standard",
        "doc_id": "spbu-os",
        "kind": "clause",
        "number": "1.1",
        "path": "1. Общие положения › 1.1",
        "text": "1.1. Текст.",
        "embedding": [0.0] * 1023 + [1.0],
    }
    with engine.begin() as connection:
        connection.execute(RegulationDocument.__table__.insert(), [document])
        connection.execute(RegulationFragment.__table__.insert(), [fragment])
        stored = connection.execute(
            select(RegulationFragment.part, RegulationFragment.codes, RegulationFragment.changed_by)
        ).one()
    assert tuple(stored) == (0, [], [])

    bad = [
        (RegulationDocument, {**document, "doc_id": "x", "level": "аспирантура"}),
        (RegulationDocument, {**document, "doc_id": "y", "doc_type": "law"}),
        (RegulationFragment, {**fragment, "id": "a", "kind": "article"}),
        (RegulationFragment, {**fragment, "id": "b", "part": -1}),
        (RegulationFragment, {**fragment, "id": "c", "edition": "нет такой редакции"}),
    ]
    for model, row in bad:
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(model.__table__.insert(), [row])


def test_vector_extension_is_not_moved_into_test_schema(migrated_database: Database) -> None:
    # иначе DROP SCHEMA … CASCADE в конце теста удалил бы расширение и колонки vector вне теста
    with migrated_database.engine.connect() as connection:
        schema = connection.scalar(
            text(
                "SELECT n.nspname FROM pg_extension e "
                "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'vector'"
            )
        )
    assert schema is not None and schema != migrated_database.schema
