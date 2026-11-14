"""Жизненный цикл зависимости FastAPI с настоящими транзакциями."""

import asyncio

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import create_async_engine

from edu_doc_agent.dao.models import Session
from tests.integration.dao.support import Database

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("failed", [False, True], ids=["normal-exit", "exception"])
def test_get_db_closes_session_and_rolls_back_uncommitted_work(
    migrated_database: Database, monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    from edu_doc_agent.dao import database as dao

    async def scenario() -> None:
        engine = create_async_engine(
            migrated_database.url, pool_size=1, max_overflow=0, pool_timeout=2
        )
        checkouts, checkins = [], []
        event.listen(
            engine.sync_engine,
            "checkout",
            lambda connection, record, proxy: checkouts.append(connection),
        )
        event.listen(
            engine.sync_engine, "checkin", lambda connection, record: checkins.append(connection)
        )
        monkeypatch.setitem(dao.async_session_maker.kw, "bind", engine)
        dependency = dao.get_db()
        try:
            session = await anext(dependency)
            session.add(Session(id="uncommitted"))
            await session.flush()
            assert session.in_transaction()
            assert len(checkouts) == 1
            assert checkouts[0] is not None
            assert checkins == []
            if failed:
                with pytest.raises(RuntimeError, match="synthetic failure"):
                    await dependency.athrow(RuntimeError("synthetic failure"))
            else:
                with pytest.raises(StopAsyncIteration):
                    await anext(dependency)
            assert checkins == checkouts
            assert engine.sync_engine.pool.checkedout() == 0
            assert not session.in_transaction()
            async with dao.async_session_maker() as another:
                assert another is not session
                assert await another.scalar(select(func.count()).select_from(Session)) == 0
        finally:
            await dependency.aclose()
            await engine.dispose()

    asyncio.run(scenario())
