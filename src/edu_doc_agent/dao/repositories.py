"""Репозиторий для управления каноническим состоянием Session и Run в PostgreSQL."""

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from edu_doc_agent.dao.database import async_session_maker
from edu_doc_agent.dao.models import Run, Session

ACTIVE_RUN_CONSTRAINT = "uq_runs_active_session"


class ActiveRunConflictError(Exception):
    """В рамках данной сессии уже существует активный запуск (run)."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Active run already exists for session '{session_id}'")


def _is_active_run_conflict(exc: IntegrityError) -> bool:
    """Проверяет, что IntegrityError вызван именно индексом uq_runs_active_session."""
    orig = exc.orig
    diag = getattr(orig, "diag", None)
    constraint_name = getattr(diag, "constraint_name", None)
    return constraint_name == ACTIVE_RUN_CONSTRAINT


class SessionRunDAO:
    """Короткоживущие операции БД для жизненного цикла сессий и запусков.

    Каждый метод открывает собственную AsyncSession и фиксирует транзакцию,
    чтобы соединения из пула не удерживались на время работы ARQ или графа.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = async_session_maker,
    ) -> None:
        self._session_factory = session_factory

    async def create_queued_run(
        self,
        session_id: str,
        run_id: UUID | None = None,
    ) -> Run:
        """Конкурентно-безопасно гарантирует существование Session и создаёт Run в 'queued'.

        Бросает ActiveRunConflictError, если в сессии уже есть активный Run.
        Любые другие ошибки целостности БД пробрасываются как есть.
        """
        async with self._session_factory() as db:
            # Идемпотентный upsert сессии: два параллельных POST не упадут на PK sessions_pkey
            stmt = (
                pg_insert(Session)
                .values(id=session_id, status="active")
                .on_conflict_do_nothing(index_elements=[Session.id])
            )
            await db.execute(stmt)

            # Создаём запуск со статусом queued
            run = Run(session_id=session_id, status="queued")
            if run_id is not None:
                run.id = run_id
            db.add(run)

            try:
                await db.commit()
            except IntegrityError as exc:
                await db.rollback()
                if _is_active_run_conflict(exc):
                    raise ActiveRunConflictError(session_id) from exc
                raise

            return run

    async def mark_failed_if_queued(self, run_id: UUID) -> bool:
        """Компенсация при сбое постановки в Redis:
        переводит Run в 'failed', только если он ещё 'queued'.

        Возвращает True, если статус был изменён, или False, если воркер уже успел
        перехватить задачу (статус 'running' или терминальный).
        """
        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(Run.id == run_id, Run.status == "queued")
                .values(status="failed", updated_at=func.clock_timestamp())
                .returning(Run.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def claim_run_for_execution(self, run_id: UUID) -> bool:
        """Атомарный переход 'queued' -> 'running' при старте выполнения задачи в воркере.

        Возвращает True, если переход выполнен и бизнес-логику можно запускать.
        Возвращает False, если Run уже не в 'queued' (например, отменён компенсацией API).
        """
        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(Run.id == run_id, Run.status == "queued")
                .values(
                    status="running",
                    last_heartbeat_at=func.clock_timestamp(),
                    updated_at=func.clock_timestamp(),
                )
                .returning(Run.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def mark_run_completed(self, run_id: UUID) -> bool:
        """Переводит выполняющийся Run ('running') в терминальный статус 'completed'."""
        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(Run.id == run_id, Run.status == "running")
                .values(status="completed", updated_at=func.clock_timestamp())
                .returning(Run.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def mark_run_failed(self, run_id: UUID) -> bool:
        """Переводит активный Run ('queued' или 'running') в терминальный статус 'failed'."""
        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(Run.id == run_id, Run.status.in_(("queued", "running")))
                .values(status="failed", updated_at=func.clock_timestamp())
                .returning(Run.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def requeue_running_run(self, run_id: UUID) -> bool:
        """Возвращает 'running' -> 'queued' при штатной отмене задачи воркера (SIGTERM/retry ARQ)"""
        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(Run.id == run_id, Run.status == "running")
                .values(status="queued", updated_at=func.clock_timestamp())
                .returning(Run.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def get_run(self, session_id: str, run_id: UUID) -> Run | None:
        """Читает Run строго по паре (session_id, run_id)."""
        async with self._session_factory() as db:
            stmt = select(Run).where(Run.id == run_id, Run.session_id == session_id)
            result = await db.execute(stmt)
            return result.scalar_one_or_none()
