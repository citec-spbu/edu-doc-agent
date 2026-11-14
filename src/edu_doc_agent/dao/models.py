import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    FetchedValue,
    ForeignKey,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Базовый класс для всех моделей."""

    pass


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'archived')", name="ck_sessions_status"),
    )
    # Получаем вычисленный updated_at при flush, без неявного async I/O после commit.
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.clock_timestamp(),
        server_onupdate=FetchedValue(),
    )

    # Связь "один-ко-многим": у одной сессии может быть много запусков
    runs: Mapped[list["Run"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'waiting_for_input', "
            "'completed', 'failed', 'cancelled')",
            name="ck_runs_status",
        ),
        CheckConstraint("reconcile_attempts >= 0", name="ck_runs_reconcile_attempts_nonnegative"),
        Index(
            "uq_runs_active_session",
            "session_id",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running', 'waiting_for_input')"),
        ),
    )
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    reconcile_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.clock_timestamp(),
        server_onupdate=FetchedValue(),
    )

    session: Mapped["Session"] = relationship(back_populates="runs")
