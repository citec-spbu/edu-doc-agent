"""Применение, откат и атомарный отказ корректирующей миграции."""

from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import insert, inspect, select, text, update
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
COUNTER_CHECK = "ck_runs_reconcile_attempts_nonnegative"


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
                {"id": uuid4(), "session_id": "session", "status": status, "reconcile_attempts": 3}
                for status in ("waiting_for_input", *TERMINAL_STATUSES)
            ],
        )
    before = rows(database)
    database.migrate()
    assert revision(database) == head_revision()
    assert rows(database) == before
    assert COUNTER_CHECK in {
        check["name"] for check in inspect(database.engine).get_check_constraints("runs")
    }
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
    with pytest.raises(IntegrityError) as error, database.engine.begin() as connection:
        connection.execute(update(Run).values(reconcile_attempts=-1))
    assert_pg_error(error.value, "23514", COUNTER_CHECK)

    result = database.alembic("downgrade", INITIAL_REVISION)
    assert result.returncode == 0, result.stdout + result.stderr
    assert revision(database) == INITIAL_REVISION
    assert rows(database) == before
    assert inspect(database.engine).get_indexes("runs") == []
    assert inspect(database.engine).get_check_constraints("runs") == []
    assert inspect(database.engine).get_check_constraints("sessions") == []
    # После отката счётчик снова может быть отрицательным; пробную запись откатываем.
    with database.engine.connect() as connection:
        assert connection.execute(update(Run).values(reconcile_attempts=-1)).rowcount > 0
        connection.rollback()
    database.migrate()
    assert revision(database) == head_revision()
    assert rows(database) == before
    with pytest.raises(IntegrityError) as error, database.engine.begin() as connection:
        connection.execute(update(Run).values(reconcile_attempts=-1))
    assert_pg_error(error.value, "23514", COUNTER_CHECK)


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
        context = MigrationContext.configure(connection, opts={"compare_server_default": True})
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


def test_negative_counter_aborts_upgrade_without_data_or_revision_changes(
    database: Database,
) -> None:
    database.migrate(INITIAL_REVISION)
    with database.engine.begin() as connection:
        connection.execute(insert(Session).values(id="session", status="active"))
        connection.execute(
            insert(Run).values(session_id="session", status="queued", reconcile_attempts=-1)
        )
    before = rows(database)
    previous_checks = inspect(database.engine).get_check_constraints("runs")
    previous_indexes = inspect(database.engine).get_indexes("runs")
    result = database.alembic("upgrade", "head")
    assert result.returncode != 0
    assert COUNTER_CHECK in result.stderr
    assert revision(database) == INITIAL_REVISION
    assert rows(database) == before
    assert inspect(database.engine).get_check_constraints("runs") == previous_checks
    assert inspect(database.engine).get_indexes("runs") == previous_indexes
