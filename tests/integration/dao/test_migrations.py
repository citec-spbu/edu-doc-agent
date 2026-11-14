"""Применение, откат и атомарный отказ корректирующей миграции."""

from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import insert, inspect, select, text
from sqlalchemy.exc import IntegrityError

from edu_doc_agent.dao.models import Base, Run, Session
from tests.integration.dao.support import (
    ACTIVE_INDEX,
    INITIAL_REVISION,
    TERMINAL_STATUSES,
    Database,
    assert_pg_error,
    head_revision,
)

pytestmark = pytest.mark.integration


def rows(database: Database) -> tuple[list, list]:
    with database.engine.connect() as connection:
        return (
            connection.execute(select(Session.__table__).order_by(Session.id)).all(),
            connection.execute(select(Run.__table__).order_by(Run.id)).all(),
        )


def revision(database: Database) -> str:
    with database.engine.connect() as connection:
        return connection.scalar(text("SELECT version_num FROM alembic_version"))


def test_upgrade_preserves_existing_valid_data_and_adds_guards(database: Database) -> None:
    database.migrate(INITIAL_REVISION)
    with database.engine.begin() as connection:
        connection.execute(insert(Session).values(id="session", status="active"))
        connection.execute(
            insert(Run),
            [
                {"id": uuid4(), "session_id": "session", "status": status}
                for status in ("waiting_for_input", *TERMINAL_STATUSES)
            ],
        )
    before = rows(database)
    database.migrate()
    assert revision(database) == head_revision()
    assert rows(database) == before
    with pytest.raises(IntegrityError) as error, database.engine.begin() as connection:
        connection.execute(insert(Run).values(session_id="session", status="queued"))
    assert_pg_error(error.value, "23505", ACTIVE_INDEX)
    for model, typo, constraint in (
        (Session, "archive", "ck_sessions_status"),
        (Run, "waitng_for_input", "ck_runs_status"),
    ):
        with pytest.raises(IntegrityError) as error, database.engine.begin() as connection:
            values = {"id": "invalid", "status": typo}
            if model is Run:
                values.update(id=uuid4(), session_id="session")
            connection.execute(insert(model).values(**values))
        assert_pg_error(error.value, "23514", constraint)


def test_duplicate_active_runs_abort_upgrade_without_data_or_revision_changes(
    database: Database,
) -> None:
    database.migrate(INITIAL_REVISION)
    with database.engine.begin() as connection:
        connection.execute(insert(Session).values(id="session", status="active"))
        connection.execute(
            insert(Run),
            [
                {"id": uuid4(), "session_id": "session", "status": status}
                for status in ("waiting_for_input", "queued")
            ],
        )
    before = rows(database)
    result = database.alembic("upgrade", "head")
    assert result.returncode != 0
    assert ACTIVE_INDEX in result.stderr
    assert rows(database) == before
    assert revision(database) == INITIAL_REVISION
    # PostgreSQL откатил всю ревизию, включая ограничения до создания индекса.
    assert inspect(database.engine).get_check_constraints("runs") == []
    assert inspect(database.engine).get_check_constraints("sessions") == []


def test_fresh_upgrade_matches_orm_and_downgrade_can_be_reapplied(database: Database) -> None:
    database.migrate()
    database.migrate()  # Повторный запуск ничего не ломает.
    with database.engine.connect() as connection:
        context = MigrationContext.configure(connection)
        assert compare_metadata(context, Base.metadata) == []
    assert revision(database) == head_revision()
    result = database.alembic("downgrade", INITIAL_REVISION)
    assert result.returncode == 0, result.stdout + result.stderr
    assert revision(database) == INITIAL_REVISION
    assert inspect(database.engine).get_indexes("runs") == []
    assert inspect(database.engine).get_check_constraints("runs") == []
    assert inspect(database.engine).get_check_constraints("sessions") == []
    database.migrate()
    assert revision(database) == head_revision()
    result = database.alembic("downgrade", "base")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not {"sessions", "runs"} & set(inspect(database.engine).get_table_names())
    database.migrate()
    assert revision(database) == head_revision()
