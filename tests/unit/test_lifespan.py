"""Владение Redis-пулом и освобождение ресурсов приложения."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from redis.exceptions import ConnectionError as RedisConnectionError

from edu_doc_agent import main


@pytest.mark.parametrize("explicit_none", [False, True])
def test_lifespan_creates_and_closes_its_pool(
    monkeypatch: pytest.MonkeyPatch, redis_pool: AsyncMock, explicit_none: bool
) -> None:
    application = FastAPI()
    if explicit_none:
        application.state.redis_pool = None
    create_pool = AsyncMock(return_value=redis_pool)
    monkeypatch.setattr(main, "create_pool", create_pool)

    async def scenario() -> None:
        async with main.lifespan(application):
            assert application.state.redis_pool is redis_pool
            create_pool.assert_awaited_once_with(main.get_settings().arq_redis_settings)
            redis_pool.close.assert_not_awaited()
        redis_pool.close.assert_awaited_once()

    asyncio.run(scenario())


def test_lifespan_preserves_an_injected_pool(
    monkeypatch: pytest.MonkeyPatch, redis_pool: AsyncMock
) -> None:
    application = FastAPI()
    application.state.redis_pool = redis_pool
    create_pool = AsyncMock()
    monkeypatch.setattr(main, "create_pool", create_pool)

    async def scenario() -> None:
        async with main.lifespan(application):
            assert application.state.redis_pool is redis_pool
        assert application.state.redis_pool is redis_pool

    asyncio.run(scenario())
    create_pool.assert_not_awaited()
    redis_pool.close.assert_not_awaited()


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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="lifespan lacks try/finally cleanup")
def test_lifespan_closes_its_pool_when_context_raises(
    monkeypatch: pytest.MonkeyPatch, redis_pool: AsyncMock
) -> None:
    monkeypatch.setattr(main, "create_pool", AsyncMock(return_value=redis_pool))

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            async with main.lifespan(FastAPI()):
                raise RuntimeError("synthetic failure")
        redis_pool.close.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="lifespan reuses its closed pool on restart"
)
def test_restarting_lifespan_creates_a_fresh_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    application = FastAPI()
    first, second = AsyncMock(), AsyncMock()
    create_pool = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(main, "create_pool", create_pool)

    async def scenario() -> None:
        async with main.lifespan(application):
            assert application.state.redis_pool is first
        first.close.assert_awaited_once()
        async with main.lifespan(application):
            assert application.state.redis_pool is second
        second.close.assert_awaited_once()

    asyncio.run(scenario())
