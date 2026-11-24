"""
Data Access Layer (DAO).

Единый модуль для доступа ко всем данным системы.
Изолирует бизнес-логику от деталей реализации инфраструктуры.
Зоны ответственности:
- Реляционная БД: модели SQLAlchemy, миграции Alembic, управление сессиями.
- Векторный поиск: гибридный поиск по нормативам (pgvector).
- Файловое хранилище: интерфейс FileStorage (Docker Volume /storage, в будущем S3/MinIO).
- Каталог правил: чтение и парсинг YAML-файлов с правилами валидации.
- Checkpointer: предоставление пула асинхронных соединений для LangGraph.
"""

from edu_doc_agent.dao.documents import DocumentDAO
from edu_doc_agent.dao.models import Base, Document, Run, Session
from edu_doc_agent.dao.repositories import ActiveRunConflictError, ReconciledRun, SessionRunDAO
from edu_doc_agent.dao.storage import (
    FileStorage,
    FileStorageError,
    InvalidStoragePathError,
    LocalFileStorage,
    StorageFileNotFoundError,
)

__all__ = [
    "ActiveRunConflictError",
    "Base",
    "Document",
    "DocumentDAO",
    "FileStorage",
    "FileStorageError",
    "InvalidStoragePathError",
    "LocalFileStorage",
    "ReconciledRun",
    "Run",
    "Session",
    "SessionRunDAO",
    "StorageFileNotFoundError",
]
