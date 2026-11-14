"""Конфликт проверяется между реально перекрывающимися транзакциями."""

import asyncio
from contextlib import suppress

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from edu_doc_agent.dao.models import Run, Session
from tests.integration.dao.support import ACTIVE_INDEX, Database, assert_pg_error

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("commit_first", [True, False], ids=["commit", "rollback"])
def test_competing_transactions(migrated_database: Database, commit_first: bool) -> None:
    with migrated_database.engine.begin() as connection:
        connection.execute(insert(Session).values(id="session", status="active"))

    async def scenario() -> None:
        engine = create_async_engine(migrated_database.url, poolclass=NullPool)
        try:
            async with (
                engine.connect() as first,
                engine.connect() as second,
                engine.connect() as observer,
            ):
                first_pid = await first.scalar(text("SELECT pg_backend_pid()"))
                second_pid = await second.scalar(text("SELECT pg_backend_pid()"))
                assert first_pid != second_pid
                await second.execute(text("SET LOCAL statement_timeout = '8s'"))
                await first.execute(
                    insert(Run).values(session_id="session", status="waiting_for_input")
                )
                competing = asyncio.create_task(
                    second.execute(insert(Run).values(session_id="session", status="queued"))
                )
                try:
                    # Ждём наблюдаемой блокировки, а не предполагаем её после задержки.
                    async with asyncio.timeout(5):
                        while True:
                            blockers = await observer.scalar(
                                text("SELECT pg_blocking_pids(:pid)"), {"pid": second_pid}
                            )
                            if first_pid in blockers:
                                break
                            assert not competing.done(), "Вторая запись обошла активную транзакцию"
                            await asyncio.sleep(0.01)
                    if commit_first:
                        await first.commit()
                        with pytest.raises(IntegrityError) as error:
                            await competing
                        assert_pg_error(error.value, "23505", ACTIVE_INDEX)
                        await second.rollback()
                    else:
                        await first.rollback()
                        await competing
                        await second.commit()
                    statuses = (await observer.scalars(select(Run.status))).all()
                    assert statuses == ["waiting_for_input" if commit_first else "queued"]
                finally:
                    competing.cancel()
                    with suppress(asyncio.CancelledError, IntegrityError):
                        await competing
        finally:
            await engine.dispose()

    asyncio.run(scenario())
