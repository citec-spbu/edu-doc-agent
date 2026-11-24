"""Репозиторий для управления метаданными документов сессии в PostgreSQL."""

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from edu_doc_agent.dao.database import async_session_maker
from edu_doc_agent.dao.models import Document, Run, Session
from edu_doc_agent.dao.repositories import ActiveRunConflictError

ACTIVE_RUN_STATUSES = ("queued", "running", "waiting_for_input")


class DocumentDAO:
    """Короткоживущие операции БД для документов в рамках сессии."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = async_session_maker,
    ) -> None:
        self._session_factory = session_factory

    async def create_document(
        self,
        *,
        session_id: str,
        doc_id: UUID,
        filename: str,
        content_type: str,
        size_bytes: int,
        storage_key: str,
        doc_type: str | None = None,
        derived_from: UUID | None = None,
    ) -> Document:
        """Гарантирует существование Session и сохраняет метаданные документа."""
        async with self._session_factory() as db:
            stmt = (
                pg_insert(Session)
                .values(id=session_id, status="active")
                .on_conflict_do_nothing(index_elements=[Session.id])
            )
            await db.execute(stmt)

            doc = Document(
                id=doc_id,
                session_id=session_id,
                filename=filename,
                content_type=content_type,
                size_bytes=size_bytes,
                storage_key=storage_key,
                doc_type=doc_type,
                derived_from=derived_from,
            )
            db.add(doc)
            await db.commit()
            return doc

    async def list_by_session(self, session_id: str) -> list[Document]:
        """Возвращает список документов сессии в порядке загрузки."""
        async with self._session_factory() as db:
            stmt = (
                select(Document)
                .where(Document.session_id == session_id)
                .order_by(Document.created_at.asc(), Document.id.asc())
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())

    async def get_document(self, session_id: str, doc_id: UUID) -> Document | None:
        """Читает документ строго по паре (session_id, doc_id)."""
        async with self._session_factory() as db:
            stmt = select(Document).where(
                Document.id == doc_id,
                Document.session_id == session_id,
            )
            result = await db.execute(stmt)
            return result.scalar_one_or_none()

    async def delete_document(self, session_id: str, doc_id: UUID) -> str | None:
        """Удаляет запись документа и возвращает storage_key (или None, если не найден).

        Бросает ActiveRunConflictError, если в сессии есть активный Run.
        Блокирует строку Session (FOR UPDATE), чтобы избежать гонки с созданием Run.
        """
        async with self._session_factory() as db:
            session_stmt = select(Session.id).where(Session.id == session_id).with_for_update()
            session_exists = (await db.execute(session_stmt)).scalar_one_or_none()
            if session_exists is None:
                return None

            active_run_stmt = (
                select(Run.id)
                .where(
                    Run.session_id == session_id,
                    Run.status.in_(ACTIVE_RUN_STATUSES),
                )
                .limit(1)
            )
            if (await db.execute(active_run_stmt)).scalar_one_or_none() is not None:
                raise ActiveRunConflictError(session_id)

            delete_stmt = (
                delete(Document)
                .where(Document.id == doc_id, Document.session_id == session_id)
                .returning(Document.storage_key)
            )
            storage_key = (await db.execute(delete_stmt)).scalar_one_or_none()
            await db.commit()
            return storage_key
