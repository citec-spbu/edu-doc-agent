import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    FetchedValue,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
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
    documents: Mapped[list["Document"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


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
    # Пока доставка не подтверждена, повторяем то же поколение без расхода лимита.
    reconcile_pending: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    last_queue_check_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.clock_timestamp(),
        server_onupdate=FetchedValue(),
    )

    session: Mapped["Session"] = relationship(back_populates="runs")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("size_bytes > 0", name="ck_documents_size_bytes_positive"),
        Index("ix_documents_session_id", "session_id"),
    )
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    # Сохраняем ключ для повтора DELETE после сбоя хранилища или остановки API.
    deletion_pending: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    doc_type: Mapped[str | None] = mapped_column(String, nullable=True)
    derived_from: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["Session"] = relationship(back_populates="documents")


# --- Нормативы для RAG (ADR-0005) -----------------------------------------------------------

EMBEDDING_DIM = 1024  # BGE-M3; другая модель — другая размерность и миграция колонки


class RegulationDocument(Base):
    """Загруженная редакция норматива: ФГОС, ОС СПбГУ, профстандарт.

    Пока хранится одна (текущая) редакция документа: загрузчик заменяет старую.
    """

    __tablename__ = "regulation_documents"
    __table_args__ = (
        CheckConstraint(
            "doc_type IN ('fgos', 'spbu_standard', 'prof_standard')",
            name="ck_regulation_documents_doc_type",
        ),
    )

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)  # «fgos-02.03.02»
    edition: Mapped[str] = mapped_column(String, primary_key=True)  # «808 ред. от 08.02.2021»
    doc_type: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    order: Mapped[str | None] = mapped_column(String, nullable=True)  # утверждающий акт
    direction_code: Mapped[str | None] = mapped_column(String, nullable=True)
    level: Mapped[str | None] = mapped_column(String, nullable=True)
    source_file: Mapped[str] = mapped_column(String)
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RegulationFragment(Base):
    """Фрагмент норматива: пункт, компетенция из пункта или строка перечня профстандартов.

    Длинный пункт хранится частями (`part`); склейка частей по порядку даёт текст дословно.
    """

    __tablename__ = "regulation_fragments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["doc_id", "edition"],
            ["regulation_documents.doc_id", "regulation_documents.edition"],
            ondelete="CASCADE",
            name="fk_regulation_fragments_document",
        ),
        CheckConstraint(
            "kind IN ('clause', 'competency', 'prof_standard_ref')",
            name="ck_regulation_fragments_kind",
        ),
        CheckConstraint("part >= 0", name="ck_regulation_fragments_part_nonnegative"),
        Index("ix_regulation_fragments_doc", "doc_id", "edition"),
        Index("ix_regulation_fragments_tsv", "tsv", postgresql_using="gin"),
        Index("ix_regulation_fragments_codes", "codes", postgresql_using="gin"),
        # по embedding индекса нет: при сотнях–тысячах фрагментов точный перебор (ADR-0005)
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)  # «fgos-02.03.02:3.2:УК-1»
    edition: Mapped[str] = mapped_column(String, primary_key=True)
    part: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_type: Mapped[str] = mapped_column(String)
    doc_id: Mapped[str] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String)
    number: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(String)
    codes: Mapped[list[str]] = mapped_column(ARRAY(String))
    parent: Mapped[str | None] = mapped_column(String, nullable=True)
    group: Mapped[str | None] = mapped_column(String, nullable=True)
    changed_by: Mapped[list[dict]] = mapped_column(JSONB)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # генерирует PostgreSQL; при вставке не передаётся
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('russian', path || ' ' || text)", persisted=True)
    )
