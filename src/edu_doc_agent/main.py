"""Точка входа FastAPI-приложения."""

from contextlib import asynccontextmanager

from arq import create_pool
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from edu_doc_agent import __version__
from edu_doc_agent.api.router import router as sessions_router
from edu_doc_agent.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом приложения (startup и shutdown).
    Подключает и отключает пул соединений к Redis для ARQ-воркера.
    """
    settings = get_settings()

    redis_pool = getattr(app.state, "redis_pool", None)
    pool_created_here = False

    if redis_pool is None:
        app.state.redis_pool = await create_pool(settings.arq_redis_settings)
        pool_created_here = True

    try:
        yield  # Сервер работает и принимает запросы
    finally:
        if pool_created_here:
            await app.state.redis_pool.close()
            del app.state.redis_pool


app = FastAPI(
    title="edu-doc-agent",
    description=(
        "Агентная ИИ-система интеллектуального анализа, автоматизированного "
        "формирования и проверки учебно-методических документов образовательной программы."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.include_router(sessions_router)


@app.get("/", include_in_schema=False)
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Редирект на интерактивную документацию."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    """Проверка живости сервиса."""
    return {"status": "ok", "version": __version__}
