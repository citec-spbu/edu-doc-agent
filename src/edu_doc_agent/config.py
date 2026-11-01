"""Настройки приложения. Значения читаются из .env через pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Все настройки проекта в одном месте.

    Значения можно задать в .env или через переменные окружения
    (переменные окружения имеют приоритет над .env).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Приложение ---
    app_name: str = "edu-doc-agent"
    log_level: str = "INFO"
    debug: bool = False

    # --- База данных ---
    postgres_user: str = "edu"
    postgres_password: str = "edu"
    postgres_db: str = "edu"
    database_url: str = "postgresql+psycopg://edu:edu@localhost:5432/edu"

    # --- Данные ---
    # Путь к папке с реальными документами ОП на машине разработчика.
    # В git документы не попадают (см. data/README.md).
    data_dir: Path | None = None

    # --- LLM ---
    llm_provider: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_rpm: int | None = None  # лимит запросов в минуту у ключа; None — без ограничения
    embedding_model: str | None = None

    @field_validator("data_dir", mode="before")
    @classmethod
    def _empty_string_is_none(cls, value: object) -> object:
        # DATA_DIR= в .env — это пустая строка, а не валидный Path.
        # Приводим её к None, чтобы не падать при старте.
        if value == "":
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Возвращает кэшированный объект настроек.

    Кэш нужен, чтобы не перечитывать .env на каждый вызов.
    В тестах, где окружение подменяется, сбрасывать через
    get_settings.cache_clear().
    """
    return Settings()
