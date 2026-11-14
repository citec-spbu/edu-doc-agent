"""Каждый тест владеет только своей случайной схемой в явно указанной БД."""

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema

from edu_doc_agent.dao.models import Base
from tests.integration.dao.support import Database


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.fail("Для DAO --run-integration нужен явный TEST_DATABASE_URL тестовой БД")
    url = make_url(dsn)
    if url.drivername != "postgresql+psycopg":
        pytest.fail("TEST_DATABASE_URL должен использовать postgresql+psycopg")
    schema = f"dao_test_{uuid4().hex}"
    admin = create_engine(url, poolclass=NullPool, connect_args={"connect_timeout": 3})
    isolated_url = url.update_query_dict({"options": f"-csearch_path={schema}"})
    engine = create_engine(isolated_url, poolclass=NullPool, connect_args={"connect_timeout": 3})
    created = False
    try:
        # Недоступная явно указанная БД — ошибка, а не skip.
        with admin.begin() as connection:
            assert connection.dialect.server_version_info[0] == 17
            connection.execute(CreateSchema(schema))
        created = True
        yield Database(isolated_url, engine, tmp_path, schema)
    finally:
        engine.dispose()
        try:
            if created:
                with admin.begin() as connection:
                    connection.execute(DropSchema(schema, cascade=True))
        finally:
            admin.dispose()


@pytest.fixture
def migrated_database(database: Database) -> Database:
    database.migrate()
    return database


@pytest.fixture(params=["alembic", "metadata"])
def schema_database(database: Database, request: pytest.FixtureRequest) -> Database:
    if request.param == "alembic":
        database.migrate()
    else:
        Base.metadata.create_all(database.engine)
    return database
