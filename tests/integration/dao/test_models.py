"""Наблюдаемые гарантии моделей и ограничений на PostgreSQL 17."""

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, insert, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from edu_doc_agent.dao.models import Base, Run, Session
from tests.integration.dao.support import (
    ACTIVE_INDEX,
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    Database,
    assert_pg_error,
)

pytestmark = pytest.mark.integration


def seed(database: Database, status: str = "queued") -> UUID:
    identifier = uuid4()
    with database.engine.begin() as connection:
        connection.execute(insert(Session).values(id="session", status="active"))
        connection.execute(insert(Run).values(id=identifier, session_id="session", status=status))
    return identifier


def test_persistence_defaults_foreign_key_and_sql_cascade(migrated_database: Database) -> None:
    database = migrated_database

    async def scenario() -> None:
        from sqlalchemy.ext.asyncio import async_sessionmaker

        engine = create_async_engine(database.url, poolclass=NullPool)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with maker() as session:
                parent = Session(id="session")
                run = Run(session=parent, status="queued")
                session.add(parent)
                await session.commit()
                assert parent.status == "active"
                assert isinstance(run.id, UUID)
                assert run.reconcile_attempts == 0
                assert run.last_heartbeat_at is None
                assert parent.created_at.tzinfo is not None
                assert run.created_at.tzinfo is not None
                identifier = run.id
            async with maker() as session:
                restored = await session.get(Run, identifier)
                assert restored is not None
                assert restored.session_id == "session"
                heartbeat = datetime(2026, 11, 14, tzinfo=UTC)
                restored.last_heartbeat_at = heartbeat
                restored.reconcile_attempts = 2
                await session.commit()
            async with maker() as session:
                restored = await session.get(Run, identifier)
                assert restored.last_heartbeat_at == heartbeat
                assert restored.reconcile_attempts == 2
        finally:
            await engine.dispose()

    asyncio.run(scenario())
    with pytest.raises(IntegrityError) as error, database.engine.begin() as connection:
        connection.execute(insert(Run).values(session_id="missing", status="queued"))
    assert_pg_error(error.value, "23503")
    with database.engine.begin() as connection:
        connection.execute(delete(Session).where(Session.id == "session"))
        assert connection.scalar(select(text("count(*)")).select_from(Run)) == 0


@pytest.mark.parametrize(
    ("existing", "incoming"),
    [(status, "queued") for status in ACTIVE_STATUSES]
    + [("queued", status) for status in ACTIVE_STATUSES[1:]],
)
def test_active_statuses_conflict(
    migrated_database: Database, existing: str, incoming: str
) -> None:
    seed(migrated_database, existing)
    with pytest.raises(IntegrityError) as error, migrated_database.engine.begin() as connection:
        connection.execute(insert(Run).values(session_id="session", status=incoming))
    assert_pg_error(error.value, "23505", ACTIVE_INDEX)


def test_update_cannot_activate_a_second_run(migrated_database: Database) -> None:
    seed(migrated_database)
    identifier = uuid4()
    with migrated_database.engine.begin() as connection:
        connection.execute(
            insert(Run).values(id=identifier, session_id="session", status="completed")
        )
    with pytest.raises(IntegrityError) as error, migrated_database.engine.begin() as connection:
        connection.execute(update(Run).where(Run.id == identifier).values(status="running"))
    assert_pg_error(error.value, "23505", ACTIVE_INDEX)


@pytest.mark.parametrize("terminal", TERMINAL_STATUSES)
def test_terminal_status_releases_slot(migrated_database: Database, terminal: str) -> None:
    identifier = seed(migrated_database, "waiting_for_input")
    with migrated_database.engine.begin() as connection:
        connection.execute(update(Run).where(Run.id == identifier).values(status=terminal))
    with migrated_database.engine.begin() as connection:
        connection.execute(insert(Run).values(session_id="session", status="queued"))
        assert connection.scalar(select(text("count(*)")).select_from(Run)) == 2


def test_resume_preserves_run_id_and_different_sessions_are_independent(
    migrated_database: Database,
) -> None:
    identifier = seed(migrated_database, "waiting_for_input")
    with migrated_database.engine.begin() as connection:
        connection.execute(update(Run).where(Run.id == identifier).values(status="queued"))
        connection.execute(insert(Session).values(id="other", status="active"))
        connection.execute(insert(Run).values(session_id="other", status="running"))
        assert connection.scalar(select(Run.id).where(Run.session_id == "session")) == identifier
        assert connection.scalar(select(Run.status).where(Run.id == identifier)) == "queued"


@pytest.mark.parametrize(
    ("model", "statuses", "typo", "constraint"),
    [
        (Session, ("active", "archived"), "archive", "ck_sessions_status"),
        (Run, ACTIVE_STATUSES + TERMINAL_STATUSES, "waitng_for_input", "ck_runs_status"),
    ],
    ids=["session", "run"],
)
def test_database_accepts_canonical_statuses_and_rejects_invalid_writes(
    migrated_database: Database, model: type, statuses: tuple[str, ...], typo: str, constraint: str
) -> None:
    database = migrated_database
    # Core INSERT/UPDATE обходит ORM-валидацию. Статусы runs разнесены по сессиям.
    identifiers = []
    for status in statuses:
        with database.engine.begin() as connection:
            parent_id = uuid4().hex
            connection.execute(insert(Session).values(id=parent_id, status="active"))
            values = {"status": status}
            if model is Run:
                values.update(session_id=parent_id, id=uuid4())
            else:
                values["id"] = uuid4().hex
            connection.execute(insert(model).values(**values))
            identifiers.append(values["id"])
    for invalid, code in ((typo, "23514"), (None, "23502")):
        with pytest.raises(IntegrityError) as error, database.engine.begin() as connection:
            values = {"id": uuid4().hex if model is Session else uuid4(), "status": invalid}
            if model is Run:
                values["session_id"] = parent_id
            connection.execute(insert(model).values(**values))
        assert_pg_error(error.value, code, constraint if invalid is not None else None)
    with pytest.raises(IntegrityError) as error, database.engine.begin() as connection:
        connection.execute(update(model).where(model.id == identifiers[0]).values(status=typo))
    assert_pg_error(error.value, "23514", constraint)


@pytest.mark.parametrize(
    ("model", "constraint"),
    [(Session, "ck_sessions_status"), (Run, "ck_runs_status")],
    ids=["session", "run"],
)
@pytest.mark.parametrize("operation", ["insert", "update"])
def test_database_rejects_empty_status(
    migrated_database: Database, model: type, constraint: str, operation: str
) -> None:
    identifier = seed(migrated_database)
    if operation == "insert":
        values = {"id": "empty" if model is Session else uuid4(), "status": ""}
        if model is Run:
            values["session_id"] = "session"
        statement = insert(model).values(**values)
    else:
        identifier = "session" if model is Session else identifier
        statement = update(model).where(model.id == identifier).values(status="")
    with pytest.raises(IntegrityError) as error, migrated_database.engine.begin() as connection:
        connection.execute(statement)
    assert_pg_error(error.value, "23514", constraint)


@pytest.mark.parametrize("model", [Session, Run], ids=["session", "run"])
def test_updated_at_is_available_after_commit_on_same_instance(
    migrated_database: Database, monkeypatch: pytest.MonkeyPatch, model: type
) -> None:
    from edu_doc_agent.dao import database as dao

    async def scenario() -> None:
        engine = create_async_engine(migrated_database.url, poolclass=NullPool)
        # Сохраняем конфигурацию фабрики приложения, меняем только подключение.
        monkeypatch.setitem(dao.async_session_maker.kw, "bind", engine)
        try:
            async with dao.async_session_maker() as session:
                parent = Session(id="session")
                instance = parent if model is Session else Run(session=parent, status="queued")
                session.add(instance)
                await session.commit()
                created_at, old_updated_at = instance.created_at, instance.updated_at
                # commit завершил транзакцию создания; UPDATE начинает следующую.
                instance.status = "archived" if model is Session else "running"
                await session.commit()
                assert instance.updated_at > old_updated_at
                assert instance.created_at == created_at
                # Запрашиваем столбцы, чтобы identity map не замаскировал расхождение.
                stored = (
                    await session.execute(
                        select(model.created_at, model.updated_at).where(model.id == instance.id)
                    )
                ).one()
                assert (instance.created_at, instance.updated_at) == tuple(stored)
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_orm_created_schema_enforces_same_guards(database: Database) -> None:
    Base.metadata.create_all(database.engine)
    seed(database)
    with pytest.raises(IntegrityError) as error, database.engine.begin() as connection:
        connection.execute(insert(Run).values(session_id="session", status="waiting_for_input"))
    assert_pg_error(error.value, "23505", ACTIVE_INDEX)
    for model, typo, constraint in (
        (Session, "archive", "ck_sessions_status"),
        (Run, "waitng_for_input", "ck_runs_status"),
    ):
        with pytest.raises(IntegrityError) as error, database.engine.begin() as connection:
            connection.execute(update(model).values(status=typo))
        assert_pg_error(error.value, "23514", constraint)
