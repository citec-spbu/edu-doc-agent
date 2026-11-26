"""Репозиторий для управления каноническим состоянием Session и Run в PostgreSQL."""

from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from edu_doc_agent.dao.database import async_session_maker
from edu_doc_agent.dao.models import Run, Session

ACTIVE_RUN_CONSTRAINT = "uq_runs_active_session"


@dataclass(frozen=True)
class ReconciledRun:
    """Запуск, восстановленный Reconciler'ом для повторной постановки в очередь ARQ."""

    run_id: UUID
    session_id: str
    reconcile_attempts: int
    reconcile_pending: bool = False


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
            stmt = (
                pg_insert(Session)
                .values(id=session_id, status="active")
                .on_conflict_do_nothing(index_elements=[Session.id])
            )
            await db.execute(stmt)

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

    async def claim_run_for_execution(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int = 0,
    ) -> int | None:
        """Атомарный переход 'queued' -> 'running'.

        Возвращает захваченное поколение (reconcile_attempts) как fencing token,
        или None, если Run уже не в 'queued' или поколение задачи устарело.
        Задачи без явно переданного поколения относятся к исходному поколению 0.
        """
        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.status == "queued",
                    Run.reconcile_attempts == expected_reconcile_attempts,
                )
                .values(
                    status="running",
                    # Старт job сам подтверждает доставку, даже если cron отменён
                    # между enqueue и записью подтверждения в PostgreSQL.
                    reconcile_pending=False,
                    last_heartbeat_at=func.clock_timestamp(),
                    updated_at=func.clock_timestamp(),
                )
                .returning(Run.reconcile_attempts)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one_or_none()

    async def touch_heartbeat(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int | None = None,
    ) -> bool:
        """Обновляет last_heartbeat_at для 'running' (с проверкой fencing token, если задан)."""
        conditions = [Run.id == run_id, Run.status == "running"]
        if expected_reconcile_attempts is not None:
            conditions.append(Run.reconcile_attempts == expected_reconcile_attempts)

        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(*conditions)
                .values(last_heartbeat_at=func.clock_timestamp())
                .returning(Run.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def mark_run_completed(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int | None = None,
    ) -> bool:
        """Переводит 'running' -> 'completed' (с проверкой fencing token, если задан)."""
        conditions = [Run.id == run_id, Run.status == "running"]
        if expected_reconcile_attempts is not None:
            conditions.append(Run.reconcile_attempts == expected_reconcile_attempts)

        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(*conditions)
                .values(status="completed", updated_at=func.clock_timestamp())
                .returning(Run.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def mark_run_failed(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int | None = None,
    ) -> bool:
        """Переводит активный Run ('queued' или 'running') в 'failed'."""
        conditions = [Run.id == run_id, Run.status.in_(("queued", "running"))]
        if expected_reconcile_attempts is not None:
            conditions.append(Run.reconcile_attempts == expected_reconcile_attempts)

        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(*conditions)
                .values(status="failed", updated_at=func.clock_timestamp())
                .returning(Run.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def requeue_running_run(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int | None = None,
    ) -> bool:
        """Возвращает 'running' -> 'queued' при штатной отмене задачи воркера."""
        conditions = [Run.id == run_id, Run.status == "running"]
        if expected_reconcile_attempts is not None:
            conditions.append(Run.reconcile_attempts == expected_reconcile_attempts)

        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(*conditions)
                .values(status="queued", updated_at=func.clock_timestamp())
                .returning(Run.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def fail_exhausted_stale_running_runs(
        self,
        *,
        stale_running_seconds: float,
        max_reconcile_attempts: int,
        limit: int = 100,
    ) -> list[UUID]:
        """Переводит зависшие 'running', достигшие лимита восстановлений, в 'failed'."""
        cutoff = func.clock_timestamp() - timedelta(seconds=stale_running_seconds)
        target_ids = (
            select(Run.id)
            .where(
                Run.status == "running",
                func.coalesce(Run.last_heartbeat_at, Run.updated_at) < cutoff,
                Run.reconcile_attempts >= max_reconcile_attempts,
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(Run.id.in_(target_ids), Run.status == "running")
                .values(status="failed", updated_at=func.clock_timestamp())
                .returning(Run.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return list(result.scalars().all())

    async def requeue_stale_running_runs(
        self,
        *,
        stale_running_seconds: float,
        max_reconcile_attempts: int,
        limit: int = 100,
    ) -> list[ReconciledRun]:
        """Резервирует новое поколение для зависших 'running', ожидающее доставки."""
        cutoff = func.clock_timestamp() - timedelta(seconds=stale_running_seconds)
        target_ids = (
            select(Run.id)
            .where(
                Run.status == "running",
                func.coalesce(Run.last_heartbeat_at, Run.updated_at) < cutoff,
                Run.reconcile_attempts < max_reconcile_attempts,
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(Run.id.in_(target_ids), Run.status == "running")
                .values(
                    status="queued",
                    reconcile_attempts=Run.reconcile_attempts + 1,
                    reconcile_pending=True,
                    updated_at=func.clock_timestamp(),
                )
                .returning(Run.id, Run.session_id, Run.reconcile_attempts, Run.reconcile_pending)
            )
            result = await db.execute(stmt)
            await db.commit()
            return [
                ReconciledRun(
                    run_id=row.id,
                    session_id=row.session_id,
                    reconcile_attempts=row.reconcile_attempts,
                    reconcile_pending=row.reconcile_pending,
                )
                for row in result.all()
            ]

    async def list_stale_queued_runs(
        self,
        *,
        stale_queued_seconds: float,
        limit: int = 100,
    ) -> list[ReconciledRun]:
        """Возвращает самых давно не проверявшихся 'queued', старше порога.

        После проверки существующей job mark_queued_run_checked сдвигает
        updated_at: следующие проходы доберутся до кандидатов за пределами LIMIT.
        """
        cutoff = func.clock_timestamp() - timedelta(seconds=stale_queued_seconds)
        async with self._session_factory() as db:
            stmt = (
                select(Run.id, Run.session_id, Run.reconcile_attempts, Run.reconcile_pending)
                .where(
                    Run.status == "queued",
                    Run.updated_at < cutoff,
                )
                .order_by(Run.updated_at, Run.id)
                .limit(limit)
            )
            result = await db.execute(stmt)
            return [
                ReconciledRun(
                    run_id=row.id,
                    session_id=row.session_id,
                    reconcile_attempts=row.reconcile_attempts,
                    reconcile_pending=row.reconcile_pending,
                )
                for row in result.all()
            ]

    async def fail_stale_queued_run(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int,
    ) -> bool:
        """Атомарно переводит потерянный 'queued' run в 'failed' (poison pill)."""
        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.status == "queued",
                    Run.reconcile_attempts == expected_reconcile_attempts,
                    Run.reconcile_pending.is_(False),
                )
                .values(status="failed", updated_at=func.clock_timestamp())
                .returning(Run.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def bump_stale_queued_run(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int,
    ) -> ReconciledRun | None:
        """Резервирует новое поколение потерянного 'queued', если нет ожидающей доставки."""
        async with self._session_factory() as db:
            stmt = (
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.status == "queued",
                    Run.reconcile_attempts == expected_reconcile_attempts,
                    Run.reconcile_pending.is_(False),
                )
                .values(
                    reconcile_attempts=Run.reconcile_attempts + 1,
                    reconcile_pending=True,
                    updated_at=func.clock_timestamp(),
                )
                .returning(Run.id, Run.session_id, Run.reconcile_attempts, Run.reconcile_pending)
            )
            result = await db.execute(stmt)
            await db.commit()
            row = result.one_or_none()
            if row is None:
                return None
            return ReconciledRun(
                run_id=row.id,
                session_id=row.session_id,
                reconcile_attempts=row.reconcile_attempts,
                reconcile_pending=row.reconcile_pending,
            )

    async def confirm_reconcile_enqueue(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int,
    ) -> bool:
        """Подтверждает доставку поколения; повторное подтверждение безопасно."""
        async with self._session_factory() as db:
            result = await db.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.reconcile_attempts == expected_reconcile_attempts,
                    Run.reconcile_pending.is_(True),
                )
                .values(reconcile_pending=False, updated_at=Run.updated_at)
                .returning(Run.id)
            )
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def mark_queued_run_checked(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int,
    ) -> bool:
        """Переносит проверенный queued run в конец обхода, не меняя поколение."""
        async with self._session_factory() as db:
            result = await db.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.status == "queued",
                    Run.reconcile_attempts == expected_reconcile_attempts,
                )
                .values(updated_at=func.clock_timestamp())
                .returning(Run.id)
            )
            await db.commit()
            return result.scalar_one_or_none() is not None

    async def get_run(self, session_id: str, run_id: UUID) -> Run | None:
        """Читает Run строго по паре (session_id, run_id)."""
        async with self._session_factory() as db:
            stmt = select(Run).where(Run.id == run_id, Run.session_id == session_id)
            result = await db.execute(stmt)
            return result.scalar_one_or_none()
