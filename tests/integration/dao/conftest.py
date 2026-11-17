"""Проверяет ограничения и на схеме Alembic, и на SQLAlchemy metadata."""

import pytest

from edu_doc_agent.dao.models import Base
from tests.integration.dao.support import Database


@pytest.fixture(params=["alembic", "metadata"])
def schema_database(database: Database, request: pytest.FixtureRequest) -> Database:
    if request.param == "alembic":
        database.migrate()
    else:
        Base.metadata.create_all(database.engine)
    return database
