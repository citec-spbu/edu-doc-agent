"""Репозиторий нормативов для RAG (ADR-0005): документы и фрагменты в PostgreSQL + pgvector."""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import (
    Select,
    Text,
    case,
    cast,
    delete,
    func,
    insert,
    literal_column,
    select,
    tuple_,
    update,
)
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
    "context",
    "codes",
    "parent",
    "category",
    "changed_by",
)
DOCUMENT_FIELDS = ("doc_type", "title", "approval_act", "direction_code", "level", "source_file")

_DOCUMENTS = RegulationDocument.__table__
_FRAGMENTS = RegulationFragment.__table__
_RUSSIAN = literal_column("'russian'::regconfig")

# ключ пункта в выдаче поиска: (id, edition); части одного пункта дают один ключ
ClauseKey = tuple[str, str]


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

    # --- поиск (ADR-0005, решение 3) ---------------------------------------------------------

    async def list_documents(self) -> list[dict[str, Any]]:
        """Загруженные документы: doc_id, редакция, тип, название, код направления, уровень."""
        async with self._session_factory() as db:
            rows = (await db.execute(select(_DOCUMENTS).order_by(_DOCUMENTS.c.doc_id))).mappings()
            return [dict(r) for r in rows]

    async def get_parts(self, keys: Sequence[ClauseKey]) -> list[dict[str, Any]]:
        """Все части фрагментов с указанными (id, edition), по порядку частей."""
        if not keys:
            return []
        columns = [_FRAGMENTS.c[name] for name in FRAGMENT_FIELDS]
        query = (
            select(*columns)
            .where(tuple_(_FRAGMENTS.c.id, _FRAGMENTS.c.edition).in_(list(keys)))
            .order_by(_FRAGMENTS.c.id, _FRAGMENTS.c.edition, _FRAGMENTS.c.part)
        )
        async with self._session_factory() as db:
            return [dict(r) for r in (await db.execute(query)).mappings()]

    async def find_by_codes(
        self,
        codes: Sequence[str],
        *,
        doc_ids: Sequence[str] | None = None,
        doc_types: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[ClauseKey]:
        """Фрагменты с любым из кодов; сначала сами компетенции и профстандарты, потом пункты,
        где код упоминается."""
        f = _FRAGMENTS.c
        query = (
            select(f.id, f.edition)
            .where(f.codes.overlap(list(codes)))
            .order_by(case((f.kind == "clause", 1), else_=0), f.doc_id, f.id, f.part)
            .limit(limit)
        )
        return await self._keys(_filtered(query, doc_ids, doc_types))

    async def find_by_numbers(
        self,
        numbers: Sequence[str],
        *,
        doc_ids: Sequence[str] | None = None,
        doc_types: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[ClauseKey]:
        """Пункты с указанными номерами («2.3»)."""
        f = _FRAGMENTS.c
        query = (
            select(f.id, f.edition)
            .where(f.number.in_(list(numbers)), f.kind == "clause")
            .order_by(f.doc_id, f.id, f.part)
            .limit(limit)
        )
        return await self._keys(_filtered(query, doc_ids, doc_types))

    async def full_text(
        self,
        text: str,
        *,
        doc_ids: Sequence[str] | None = None,
        doc_types: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[ClauseKey]:
        """Полнотекстовый поиск по tsv: слова запроса через ИЛИ, порядок — ts_rank_cd.

        ИЛИ, а не И: вопрос на естественном языке редко содержит все слова пункта.
        """
        f = _FRAGMENTS.c
        plain = cast(func.plainto_tsquery(_RUSSIAN, text), Text)
        tsquery = func.to_tsquery(_RUSSIAN, func.replace(plain, " & ", " | "))
        query = (
            select(f.id, f.edition)
            .where(f.tsv.op("@@")(tsquery))
            .order_by(func.ts_rank_cd(f.tsv, tsquery).desc(), f.id, f.part)
            .limit(limit)
        )
        return await self._keys(_filtered(query, doc_ids, doc_types))

    async def nearest(
        self,
        embedding: Sequence[float],
        *,
        doc_ids: Sequence[str] | None = None,
        doc_types: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[ClauseKey]:
        """Ближайшие по косинусному расстоянию (точный перебор, ANN-индекса нет)."""
        f = _FRAGMENTS.c
        query = (
            select(f.id, f.edition)
            .order_by(f.embedding.cosine_distance(list(embedding)), f.id, f.part)
            .limit(limit)
        )
        return await self._keys(_filtered(query, doc_ids, doc_types))

    async def _keys(self, query: Select) -> list[ClauseKey]:
        async with self._session_factory() as db:
            return [(row.id, row.edition) for row in await db.execute(query)]


def _filtered(
    query: Select, doc_ids: Sequence[str] | None, doc_types: Sequence[str] | None
) -> Select:
    if doc_ids is not None:
        query = query.where(_FRAGMENTS.c.doc_id.in_(list(doc_ids)))
    if doc_types is not None:
        query = query.where(_FRAGMENTS.c.doc_type.in_(list(doc_types)))
    return query
