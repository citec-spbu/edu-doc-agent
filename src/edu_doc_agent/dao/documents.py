"""Репозиторий для управления метаданными документов сессии в PostgreSQL."""

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from edu_doc_agent.dao.database import async_session_maker
from edu_doc_agent.dao.models import Document, Run, Session
from edu_doc_agent.dao.repositories import ActiveRunConflictError

ACTIVE_RUN_STATUSES = ("queued", "running", "waiting_for_input")


class DocumentCreateRolledBackError(Exception):
    """Создание не дошло до COMMIT; rollback подтверждён, файл можно удалить."""


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
        """Сохраняет документ; только DocumentCreateRolledBackError разрешает компенсацию.

        Любая другая ошибка может означать потерянное подтверждение COMMIT,
        включая ошибку при возврате/закрытии сессии после успешной фиксации.
        """
        async with self._session_factory() as db:
            stmt = (
                pg_insert(Session)
                .values(id=session_id, status="active")
                .on_conflict_do_nothing(index_elements=[Session.id])
            )
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
            try:
                await db.execute(stmt)
                db.add(doc)
                # Все INSERT и ограничения проверяются до отправки COMMIT.
                await db.flush()
            except Exception as exc:
                await db.rollback()
                raise DocumentCreateRolledBackError("Document creation rolled back") from exc
            # Ошибку COMMIT нельзя превращать в подтверждённый отказ даже
            # после успешного rollback(): сервер мог уже зафиксировать INSERT.
            await db.commit()
            return doc

    async def list_by_session(self, session_id: str) -> list[Document]:
        """Возвращает доступный пакет сессии без логически удалённых документов."""
        async with self._session_factory() as db:
            stmt = (
                select(Document)
                .where(Document.session_id == session_id, Document.deletion_pending.is_(False))
                .order_by(Document.created_at.asc(), Document.id.asc())
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())

    async def get_document(self, session_id: str, doc_id: UUID) -> Document | None:
        """Публичное чтение по паре (session_id, doc_id), без удаляемых документов."""
        async with self._session_factory() as db:
            stmt = select(Document).where(
                Document.id == doc_id,
                Document.session_id == session_id,
                Document.deletion_pending.is_(False),
            )
            result = await db.execute(stmt)
            return result.scalar_one_or_none()

    async def get_document_for_cleanup(self, session_id: str, doc_id: UUID) -> Document | None:
        """Служебное чтение незавершённого удаления; публичный GET его не использует."""
        async with self._session_factory() as db:
            return await db.scalar(
                select(Document).where(
                    Document.id == doc_id,
                    Document.session_id == session_id,
                    Document.deletion_pending.is_(True),
                )
            )

    async def delete_document(self, session_id: str, doc_id: UUID) -> str | None:
        """Логически удаляет документ и возвращает ключ для очистки хранилища.

        COMMIT deletion_pending под FOR UPDATE Session — единственный момент
        удаления. Вставка Run берёт конфликтующую FK-блокировку этой же Session:
        если Run выиграл, первый DELETE получает ActiveRunConflictError без правок.
        Уже принятое удаление повторяется с тем же ключом независимо от новых Run.
        Файловый I/O выполняется после завершения этой транзакции.
        """
        async with self._session_factory() as db:
            session_stmt = select(Session.id).where(Session.id == session_id).with_for_update()
            if (await db.execute(session_stmt)).scalar_one_or_none() is None:
                return None

            doc = await db.scalar(
                select(Document).where(Document.id == doc_id, Document.session_id == session_id)
            )
            if doc is None:
                return None
            if doc.deletion_pending:
                return doc.storage_key

            active_run_stmt = (
                select(Run.id)
                .where(Run.session_id == session_id, Run.status.in_(ACTIVE_RUN_STATUSES))
                .limit(1)
            )
            if (await db.execute(active_run_stmt)).scalar_one_or_none() is not None:
                raise ActiveRunConflictError(session_id)

            doc.deletion_pending = True
            storage_key = doc.storage_key
            await db.commit()
            return storage_key

    async def complete_document_deletion(
        self, session_id: str, doc_id: UUID, *, storage_key: str
    ) -> None:
        """Подтверждает удаление метаданных после успешной очистки хранилища.

        Решение уже принято в delete_document: новый Run не может его отменить.
        Если другой DELETE завершил очистку, подтверждение ничего не делает.
        """
        async with self._session_factory() as db:
            delete_stmt = delete(Document).where(
                Document.id == doc_id,
                Document.session_id == session_id,
                Document.storage_key == storage_key,
                Document.deletion_pending.is_(True),
            )
            await db.execute(delete_stmt)
            remaining = await db.scalar(
                select(Document.id).where(Document.id == doc_id, Document.session_id == session_id)
            )
            if remaining is not None:
                raise RuntimeError("Document deletion was not prepared for this storage key")
            await db.commit()
