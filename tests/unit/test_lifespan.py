"""Владение Redis-пулом и освобождение ресурсов приложения."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from redis.exceptions import ConnectionError as RedisConnectionError

from edu_doc_agent import main


class PoolProbe:
    """Модель ресурса: использование подключает, close/aclose освобождает.

    Как настоящий Redis-клиент, допускает использование после закрытия.
    Тесты наблюдают ресурс, не выбирая способ его закрытия и пересоздания.
    """

    def __init__(self) -> None:
        self.connected = False

    async def ping(self) -> bool:
        self.connected = True
        return True

    async def aclose(self, close_connection_pool: bool | None = None) -> None:
        self.connected = False

    async def close(self, close_connection_pool: bool | None = None) -> None:
        await self.aclose(close_connection_pool)


@pytest.mark.parametrize("explicit_none", [False, True])
def test_lifespan_creates_and_closes_its_pool(
    monkeypatch: pytest.MonkeyPatch, explicit_none: bool
) -> None:
    application = FastAPI()
    if explicit_none:
        application.state.redis_pool = None
    pool = PoolProbe()
    create_pool = AsyncMock(return_value=pool)
    monkeypatch.setattr(main, "create_pool", create_pool)

    async def scenario() -> None:
        async with main.lifespan(application):
            assert await application.state.redis_pool.ping()
            create_pool.assert_awaited_once_with(main.get_settings().arq_redis_settings)
            assert pool.connected
        assert not pool.connected

    asyncio.run(scenario())


def test_lifespan_preserves_an_injected_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    application = FastAPI()
    pool = PoolProbe()
    application.state.redis_pool = pool
    create_pool = AsyncMock()
    monkeypatch.setattr(main, "create_pool", create_pool)

    async def scenario() -> None:
        await pool.ping()
        try:
            async with main.lifespan(application):
                assert application.state.redis_pool is pool
            assert application.state.redis_pool is pool
            assert pool.connected
        finally:
            await pool.aclose()

    asyncio.run(scenario())
    create_pool.assert_not_awaited()


def test_failed_redis_connection_aborts_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    application = FastAPI()
    create_pool = AsyncMock(side_effect=RedisConnectionError("synthetic startup failure"))
    monkeypatch.setattr(main, "create_pool", create_pool)

    async def scenario() -> None:
        with pytest.raises(RedisConnectionError, match="synthetic startup failure"):
            async with main.lifespan(application):
                pytest.fail("application started without Redis")
        assert getattr(application.state, "redis_pool", None) is None

    asyncio.run(scenario())
    create_pool.assert_awaited_once()


def test_lifespan_closes_its_pool_when_context_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FastAPI()
    pool = PoolProbe()
    monkeypatch.setattr(main, "create_pool", AsyncMock(return_value=pool))

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            async with main.lifespan(application):
                await application.state.redis_pool.ping()
                raise RuntimeError("synthetic failure")
        assert not pool.connected

    asyncio.run(scenario())


def test_restarting_lifespan_releases_connections_after_each_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FastAPI()
    pools: list[PoolProbe] = []

    def make_pool(*args, **kwargs) -> PoolProbe:
        pool = PoolProbe()
        pools.append(pool)
        return pool

    monkeypatch.setattr(main, "create_pool", AsyncMock(side_effect=make_pool))

    async def scenario() -> None:
        for _ in range(2):
            async with main.lifespan(application):
                assert await application.state.redis_pool.ping()
                assert application.state.redis_pool.connected
            assert all(not pool.connected for pool in pools)

    asyncio.run(scenario())
