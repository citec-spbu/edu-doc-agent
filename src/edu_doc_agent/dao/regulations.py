"""Репозиторий нормативов для RAG (ADR-0005): документы и фрагменты в PostgreSQL + pgvector."""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from edu_doc_agent.dao.database import async_session_maker
from edu_doc_agent.dao.models import RegulationDocument, RegulationFragment

# поля фрагмента без эмбеддинга и tsv — по ним загрузчик сравнивает содержимое
FRAGMENT_FIELDS = (
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
    "category",
    "changed_by",
)
DOCUMENT_FIELDS = ("doc_type", "title", "approval_act", "direction_code", "level", "source_file")

_DOCUMENTS = RegulationDocument.__table__
_FRAGMENTS = RegulationFragment.__table__


class RegulationDAO:
    """Короткие операции с нормативами; каждый метод — своя сессия и транзакция."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = async_session_maker,
    ) -> None:
        self._session_factory = session_factory

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Загруженная редакция документа (их не больше одной) или None."""
        async with self._session_factory() as db:
            rows = (
                await db.execute(select(_DOCUMENTS).where(_DOCUMENTS.c.doc_id == doc_id))
            ).mappings()
            found = [dict(r) for r in rows]
        return found[0] if len(found) == 1 else None

    async def list_fragments(
        self, doc_id: str, *, with_embedding: bool = False
    ) -> list[dict[str, Any]]:
        """Фрагменты документа (всех редакций) в виде словарей полей таблицы."""
        columns = [_FRAGMENTS.c[name] for name in FRAGMENT_FIELDS]
        if with_embedding:
            columns.append(_FRAGMENTS.c.embedding)
        async with self._session_factory() as db:
            rows = (
                await db.execute(select(*columns).where(_FRAGMENTS.c.doc_id == doc_id))
            ).mappings()
            return [dict(r) for r in rows]

    async def replace_document(
        self, document: dict[str, Any], fragments: Sequence[dict[str, Any]]
    ) -> None:
        """Заменить все редакции документа новой — одной транзакцией.

        Строки прежних редакций удаляются из regulation_documents, фрагменты уходят каскадом.
        """
        async with self._session_factory() as db, db.begin():
            await db.execute(
                delete(RegulationDocument).where(RegulationDocument.doc_id == document["doc_id"])
            )
            await db.execute(insert(RegulationDocument), [document])
            if fragments:
                await db.execute(insert(RegulationFragment), list(fragments))

    async def update_document(self, doc_id: str, values: dict[str, Any]) -> None:
        """Обновить метаданные документа (название, акт…) без перезаписи фрагментов."""
        async with self._session_factory() as db, db.begin():
            await db.execute(
                update(RegulationDocument)
                .where(RegulationDocument.doc_id == doc_id)
                .values(**values)
            )

    async def delete_documents_except(self, keep: Sequence[str]) -> list[str]:
        """Удалить документы, которых нет в `keep`; вернуть их doc_id."""
        async with self._session_factory() as db, db.begin():
            removed = await db.execute(
                delete(RegulationDocument)
                .where(RegulationDocument.doc_id.not_in(list(keep)))
                .returning(RegulationDocument.doc_id)
            )
            return sorted(set(removed.scalars()))
