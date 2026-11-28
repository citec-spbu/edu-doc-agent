"""Общие фикстуры интеграционных тестов RAG: RegulationDAO на мигрированной схеме теста."""

import asyncio
from collections.abc import Iterator

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from edu_doc_agent.dao.regulations import RegulationDAO
from tests.integration.dao.support import Database


@pytest.fixture
def dao(migrated_database: Database) -> Iterator[RegulationDAO]:
    engine = create_async_engine(migrated_database.url, poolclass=NullPool)
    try:
        yield RegulationDAO(async_sessionmaker(engine, expire_on_commit=False))
    finally:
        asyncio.run(engine.dispose())
