"""Percent-encoding проверяется настоящей аутентификацией, а не mock-движком."""

import socket
from uuid import uuid4

import pytest
from psycopg import sql
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from tests.integration.dao.support import Database, head_revision

pytestmark = pytest.mark.integration

PASSWORD = "synthetic@percent%35"
ENCODED_PASSWORD = "synthetic%40percent%2535"


def test_alembic_online_offline_and_failure_do_not_expose_encoded_password(
    database: Database,
) -> None:
    role = f"dao_login_{uuid4().hex}"
    url = database.url.set(username=role, password=PASSWORD)
    assert ENCODED_PASSWORD in url.render_as_string(hide_password=False)
    created = False
    try:
        with database.engine.begin() as connection:
            cursor = connection.connection.driver_connection.cursor()
            cursor.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(PASSWORD)
                )
            )
            cursor.execute(
                sql.SQL("GRANT USAGE, CREATE ON SCHEMA {} TO {}").format(
                    sql.Identifier(database.schema), sql.Identifier(role)
                )
            )
        created = True
        # Проверяем, что тот же URL принимается SQLAlchemy и PostgreSQL.
        engine = create_engine(url, poolclass=NullPool, connect_args={"connect_timeout": 3})
        try:
            with engine.connect() as connection:
                assert connection.scalar(text("SELECT current_user")) == role
        finally:
            engine.dispose()
        # Порт занят сокетом без listen(): подключиться к БД здесь невозможно.
        # Offline обязан сгенерировать SQL, даже если DSN недоступен.
        with socket.socket() as unavailable:
            unavailable.bind(("127.0.0.1", 0))
            offline_url = url.set(
                host="127.0.0.1", port=unavailable.getsockname()[1]
            ).update_query_dict({"connect_timeout": "1"})
            offline = database.alembic("upgrade", "head", "--sql", url=offline_url)
        assert offline.returncode == 0, offline.stdout + offline.stderr
        assert "CREATE TABLE sessions" in offline.stdout
        assert head_revision() in offline.stdout
        online = database.alembic("upgrade", "head", url=url)
        assert online.returncode == 0, online.stdout + online.stderr
        with database.engine.connect() as connection:
            assert (
                connection.scalar(text("SELECT version_num FROM alembic_version"))
                == head_revision()
            )
        # Несуществующая БД даёт настоящий online-сбой после разбора URL.
        failure_url = url.set(database=f"dao_missing_{uuid4().hex}")
        failure = database.alembic("upgrade", "head", url=failure_url)
        assert failure.returncode != 0
        assert "OperationalError" in failure.stderr
        for result, used_url in ((offline, offline_url), (online, url), (failure, failure_url)):
            output = result.stdout + result.stderr
            assert PASSWORD not in output
            assert ENCODED_PASSWORD not in output
            assert used_url.render_as_string(hide_password=False) not in output
    finally:
        if created:
            with database.engine.begin() as connection:
                cursor = connection.connection.driver_connection.cursor()
                # Только собственная роль и созданные ею объекты в своей схеме.
                cursor.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
                cursor.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
