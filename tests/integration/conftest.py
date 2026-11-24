"""Каждый тест владеет только своей случайной схемой в явно указанной БД."""

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema

from tests.integration.dao.support import Database


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.fail("Для PostgreSQL --run-integration нужен явный TEST_DATABASE_URL тестовой БД")
    url = make_url(dsn)
    if url.drivername != "postgresql+psycopg":
        pytest.fail("TEST_DATABASE_URL должен использовать postgresql+psycopg")
    schema = f"dao_test_{uuid4().hex}"
    admin = create_engine(url, poolclass=NullPool, connect_args={"connect_timeout": 3})
    engine = None
    created = False
    try:
        # Недоступная явно указанная БД — ошибка, а не skip.
        with admin.begin() as connection:
            assert connection.dialect.server_version_info[0] == 17
            connection.execute(CreateSchema(schema))
            extensions = _vector_extension_schema(connection)
        created = True
        # Объекты теста — в своей схеме; схема pgvector — второй в пути, чтобы был виден тип
        # vector.
        isolated_url = url.update_query_dict({"options": f"-csearch_path={schema},{extensions}"})
        engine = create_engine(
            isolated_url, poolclass=NullPool, connect_args={"connect_timeout": 3}
        )
        yield Database(isolated_url, engine, tmp_path, schema)
    finally:
        if engine is not None:
            engine.dispose()
        try:
            if created:
                with admin.begin() as connection:
                    connection.execute(DropSchema(schema, cascade=True))
        finally:
            admin.dispose()


def _vector_extension_schema(connection: Connection) -> str:
    """Схема, где стоит pgvector; если расширения нет — ставит его в public.

    Расширение одно на базу и общее для всех схем, поэтому в схему теста его не переносим:
    `DROP SCHEMA … CASCADE` в конце теста удалил бы его вместе с колонками vector вне теста.
    Ставим его здесь суперпользователем; миграция с `CREATE EXTENSION IF NOT EXISTS` тогда
    проходит и под ролью без прав суперпользователя.
    """
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public"))
    schema = connection.scalar(
        text(
            "SELECT n.nspname FROM pg_extension e "
            "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'vector'"
        )
    )
    return connection.dialect.identifier_preparer.quote(schema)


@pytest.fixture
def migrated_database(database: Database) -> Database:
    database.migrate()
    return database
