from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from edu_doc_agent.config import get_settings

settings = get_settings()

# Создаем асинхронный движок (он управляет пулом подключений к БД)
engine = create_async_engine(settings.database_url, echo=settings.debug)

# Фабрика для создания новых сессий БД
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# Зависимость для FastAPI, чтобы получать сессию БД в эндпоинтах
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
