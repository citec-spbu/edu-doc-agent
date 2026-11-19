"""Загрузка корпуса нормативов в базу: сборка -> эмбеддинги -> regulation_* (ADR-0005).

    uv run python -m edu_doc_agent.retrieval.corpus.load data/regulations/manifest.yaml \\
        --overrides data/regulations/overrides.yaml

Хранится одна (текущая) редакция документа: новая редакция заменяет старую в одной транзакции.
Эмбеддинги считаются только для фрагментов, чей «путь + текст» ещё не встречался в базе для
этого документа; если корпус не менялся, повторная загрузка ничего не пишет.
"""

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, Engine, Row, create_engine, delete, insert, select, update

from edu_doc_agent.dao.models import RegulationDocument, RegulationFragment
from edu_doc_agent.llm.embeddings import Embedder
from edu_doc_agent.retrieval.corpus.build import build
from edu_doc_agent.retrieval.corpus.models import CorpusDocument, Fragment

_DOCUMENT_FIELDS = ("doc_type", "title", "order", "direction_code", "level", "source_file")
# поля фрагмента, по которым сравнивается содержимое (без эмбеддинга и tsv)
_FRAGMENT_FIELDS = (
    "id",
    "edition",
    "part",
    "doc_type",
    "doc_id",
    "kind",
    "number",
    "path",
    "text",
    "codes",
    "parent",
    "group",
    "changed_by",
)


@dataclass
class LoadResult:
    loaded: list[str] = field(default_factory=list)  # записаны заново
    unchanged: list[str] = field(default_factory=list)  # совпали с базой, не трогали
    removed: list[str] = field(default_factory=list)  # удалены (нет в корпусе, --prune)
    embedded: int = 0  # сколько фрагментов отправлено в модель


def embedding_input(fragment: Fragment) -> str:
    """Строка для эмбеддинга: путь по разделам и текст (ADR-0005, схема хранения)."""
    return f"{fragment.path}\n{fragment.text}"


def fragment_row(fragment: Fragment) -> dict[str, Any]:
    data = fragment.model_dump()
    return {name: data[name] for name in _FRAGMENT_FIELDS}


def document_row(doc: CorpusDocument) -> dict[str, Any]:
    info = doc.document
    return {
        "doc_id": info.doc_id,
        "edition": info.edition,
        **{name: getattr(info, name) for name in _DOCUMENT_FIELDS},
    }


def load_documents(
    engine: Engine,
    documents: Sequence[CorpusDocument],
    embedder: Embedder,
    *,
    prune: bool = False,
) -> LoadResult:
    """Записать документы в базу. Каждый документ — своя транзакция; эмбеддинги — до неё."""
    result = LoadResult()
    for doc in documents:
        doc_id = doc.document.doc_id
        rows = [fragment_row(f) for f in doc.fragments]
        with engine.connect() as connection:
            stored = _stored_fragments(connection, doc_id)
            stored_document = connection.execute(
                select(RegulationDocument.__table__).where(RegulationDocument.doc_id == doc_id)
            ).all()
        if _same_content(stored, rows, doc.document.edition, stored_document):
            if _metadata_changed(stored_document[0], doc):
                with engine.begin() as connection:
                    connection.execute(
                        update(RegulationDocument)
                        .where(RegulationDocument.doc_id == doc_id)
                        .values(**{k: document_row(doc)[k] for k in _DOCUMENT_FIELDS})
                    )
            result.unchanged.append(doc_id)
            continue

        known = {_embedding_key(row): row._mapping["embedding"] for row in stored}
        missing = list(
            dict.fromkeys(key for f in doc.fragments if (key := embedding_input(f)) not in known)
        )
        if missing:
            known |= dict(zip(missing, embedder.embed(missing), strict=True))
            result.embedded += len(missing)
        for row, fragment in zip(rows, doc.fragments, strict=True):
            row["embedding"] = known[embedding_input(fragment)]

        with engine.begin() as connection:
            # каскадом уходят и фрагменты прежних редакций
            connection.execute(
                delete(RegulationDocument).where(RegulationDocument.doc_id == doc_id)
            )
            connection.execute(insert(RegulationDocument), [document_row(doc)])
            if rows:
                connection.execute(insert(RegulationFragment), rows)
        result.loaded.append(doc_id)

    if prune:
        keep = [d.document.doc_id for d in documents]
        with engine.begin() as connection:
            removed = connection.execute(
                delete(RegulationDocument)
                .where(RegulationDocument.doc_id.not_in(keep))
                .returning(RegulationDocument.doc_id)
            ).scalars()
            result.removed = sorted(set(removed))
    return result


def _stored_fragments(connection: Connection, doc_id: str) -> list[Row]:
    table = RegulationFragment.__table__
    columns = [table.c[name] for name in _FRAGMENT_FIELDS] + [table.c.embedding]
    return list(connection.execute(select(*columns).where(table.c.doc_id == doc_id)).all())


def _embedding_key(row: Row) -> str:
    return f"{row._mapping['path']}\n{row._mapping['text']}"


def _same_content(
    stored: list[Row], rows: list[dict[str, Any]], edition: str, documents: list[Row]
) -> bool:
    if len(documents) != 1 or documents[0].edition != edition:
        return False
    previous = [{name: r._mapping[name] for name in _FRAGMENT_FIELDS} for r in stored]
    return sorted(previous, key=_row_key) == sorted(rows, key=_row_key)


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["id"], row["edition"], row["part"]


def _metadata_changed(stored: Row, doc: CorpusDocument) -> bool:
    new = document_row(doc)
    return any(stored._mapping[name] != new[name] for name in _DOCUMENT_FIELDS)


def _loadable(documents: Iterable[CorpusDocument]) -> tuple[list[CorpusDocument], list[str]]:
    """Документы с ошибками сборки (дубли id и т. п.) в базу не пускаем."""
    good, errors = [], []
    for doc in documents:
        fatal = [w for w in doc.warnings if w.startswith("ОШИБКА")]
        if fatal:
            errors += [f"{doc.document.doc_id}: {w}" for w in fatal]
        else:
            good.append(doc)
    return good, errors


def main(argv: list[str] | None = None) -> int:
    from edu_doc_agent.config import get_settings
    from edu_doc_agent.llm.embeddings import OllamaEmbedder

    parser = argparse.ArgumentParser(description="Загрузка корпуса нормативов в базу для RAG")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument(
        "--prune", action="store_true", help="удалить из базы документы, которых нет в манифесте"
    )
    args = parser.parse_args(argv)

    documents, failures = build(args.manifest, args.overrides)
    documents, errors = _loadable(documents)
    errors = failures + errors
    for error in errors:
        print(f"ОШИБКА: {error}", file=sys.stderr)
    if errors and args.prune:
        print("--prune не выполняется: часть документов не собрана", file=sys.stderr)

    engine = create_engine(get_settings().database_connection_url)
    try:
        result = load_documents(
            engine, documents, OllamaEmbedder.from_settings(), prune=args.prune and not errors
        )
    finally:
        engine.dispose()
    print(f"Загружено: {', '.join(result.loaded) or '—'} (эмбеддингов: {result.embedded})")
    print(f"Без изменений: {', '.join(result.unchanged) or '—'}")
    if result.removed:
        print(f"Удалено: {', '.join(result.removed)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
