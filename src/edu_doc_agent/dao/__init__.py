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

from edu_doc_agent.dao.models import Base, Run, Session
from edu_doc_agent.dao.repositories import ActiveRunConflictError, SessionRunDAO

__all__ = [
    "ActiveRunConflictError",
    "Base",
    "Run",
    "Session",
    "SessionRunDAO",
]
