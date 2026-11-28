"""Загрузка корпуса нормативов в базу: сборка -> эмбеддинги -> regulation_* (ADR-0005).

    uv run python -m edu_doc_agent.retrieval.corpus.load data/regulations/manifest.yaml \\
        --overrides data/regulations/overrides.yaml

Хранится одна (текущая) редакция документа: новая редакция заменяет старую в одной транзакции.
Эмбеддинги считаются только для фрагментов, чей «путь + текст» ещё не встречался в базе для
этого документа; если корпус не менялся, повторная загрузка ничего не пишет.
"""

import argparse
import asyncio
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edu_doc_agent.dao.regulations import DOCUMENT_FIELDS, FRAGMENT_FIELDS, RegulationDAO
from edu_doc_agent.llm.embeddings import Embedder
from edu_doc_agent.retrieval.corpus.build import build
from edu_doc_agent.retrieval.corpus.models import CorpusDocument, Fragment


@dataclass
class LoadResult:
    loaded: list[str] = field(default_factory=list)  # записаны заново
    unchanged: list[str] = field(default_factory=list)  # совпали с базой, не трогали
    removed: list[str] = field(default_factory=list)  # удалены (нет в корпусе, --prune)
    embedded: int = 0  # сколько фрагментов отправлено в модель


def embedding_input(fragment: Fragment) -> str:
    """Строка для эмбеддинга: путь, контекст и текст (ADR-0005, схема хранения)."""
    return _input(fragment.path, fragment.context, fragment.text)


def _input(path: str, context: str, text: str) -> str:
    return "\n".join(part for part in (path, context, text) if part)


def fragment_row(fragment: Fragment) -> dict[str, Any]:
    data = fragment.model_dump()
    return {name: data[name] for name in FRAGMENT_FIELDS}


def document_row(doc: CorpusDocument) -> dict[str, Any]:
    info = doc.document
    return {
        "doc_id": info.doc_id,
        "edition": info.edition,
        **{name: getattr(info, name) for name in DOCUMENT_FIELDS},
    }


async def load_documents(
    dao: RegulationDAO,
    documents: Sequence[CorpusDocument],
    embedder: Embedder,
    *,
    prune: bool = False,
) -> LoadResult:
    """Записать документы в базу. Эмбеддинги считаются до транзакции, она короткая."""
    result = LoadResult()
    for doc in documents:
        doc_id, edition = doc.document.doc_id, doc.document.edition
        rows = [fragment_row(f) for f in doc.fragments]
        stored_document = await dao.get_document(doc_id)
        stored = await dao.list_fragments(doc_id, with_embedding=True)

        if stored_document is not None and stored_document["edition"] == edition:
            previous = [{k: r[k] for k in FRAGMENT_FIELDS} for r in stored]
            if sorted(previous, key=_row_key) == sorted(rows, key=_row_key):
                new = document_row(doc)
                changed = {k: new[k] for k in DOCUMENT_FIELDS if stored_document[k] != new[k]}
                if changed:
                    await dao.update_document(doc_id, changed)
                result.unchanged.append(doc_id)
                continue

        known = {_input(r["path"], r["context"], r["text"]): r["embedding"] for r in stored}
        missing = list(
            dict.fromkeys(key for f in doc.fragments if (key := embedding_input(f)) not in known)
        )
        if missing:
            vectors = await asyncio.to_thread(embedder.embed, missing)
            known |= dict(zip(missing, vectors, strict=True))
            result.embedded += len(missing)
        for row, fragment in zip(rows, doc.fragments, strict=True):
            row["embedding"] = known[embedding_input(fragment)]

        await dao.replace_document(document_row(doc), rows)
        result.loaded.append(doc_id)

    if prune:
        result.removed = await dao.delete_documents_except([d.document.doc_id for d in documents])
    return result


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["id"], row["edition"], row["part"]


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


async def _run(args: argparse.Namespace) -> int:
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from edu_doc_agent.config import get_settings
    from edu_doc_agent.llm.embeddings import OllamaEmbedder
    from edu_doc_agent.llm.openai_compat import LLMError

    documents, failures = build(args.manifest, args.overrides)
    documents, errors = _loadable(documents)
    errors = failures + errors
    for error in errors:
        print(f"ОШИБКА: {error}", file=sys.stderr)
    if errors and args.prune:
        print("--prune не выполняется: часть документов не собрана", file=sys.stderr)

    # своё подключение без пула, как в alembic/env.py: CLI живёт недолго
    settings = get_settings()
    engine = create_async_engine(settings.database_connection_url, poolclass=NullPool)
    embedder = OllamaEmbedder.from_settings(settings)
    try:
        dao = RegulationDAO(async_sessionmaker(engine, expire_on_commit=False))
        result = await load_documents(dao, documents, embedder, prune=args.prune and not errors)
    except LLMError as error:
        print(
            f"ОШИБКА эмбеддингов: {error}\n"
            "Запущена ли Ollama с моделью? docker compose --profile local-llm up -d ollama; "
            f"адрес — EMBEDDING_BASE_URL (сейчас {settings.embedding_base_url or 'по умолчанию'})",
            file=sys.stderr,
        )
        return 2
    except OperationalError as error:
        print(
            f"ОШИБКА базы: {error.orig}\n"
            "Запущен ли PostgreSQL и применены ли миграции? docker compose up -d db; "
            "uv run alembic upgrade head",
            file=sys.stderr,
        )
        return 2
    finally:
        await engine.dispose()
    print(f"Загружено: {', '.join(result.loaded) or '—'} (эмбеддингов: {result.embedded})")
    print(f"Без изменений: {', '.join(result.unchanged) or '—'}")
    if result.removed:
        print(f"Удалено: {', '.join(result.removed)}")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Загрузка корпуса нормативов в базу для RAG")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument(
        "--prune", action="store_true", help="удалить из базы документы, которых нет в манифесте"
    )
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
