"""
Pydantic-схемы данных для API.
Описывают форматы входящих запросов и исходящих ответов.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ProcessResponse(BaseModel):
    """Схема ответа при успешной постановке задачи в очередь."""

    run_id: UUID = Field(..., description="Уникальный идентификатор текущего запуска (run_id)")

    session_id: str = Field(
        ..., description="Идентификатор диалога/сессии пользователя (thread_id)"
    )

    status: str = Field(default="queued", description="Статус задачи на момент ответа API")


class RunStatusResponse(BaseModel):
    """Схема ответа с каноническим статусом запуска из PostgreSQL."""

    run_id: UUID = Field(..., description="Уникальный идентификатор запуска (run_id)")
    session_id: str = Field(
        ..., description="Идентификатор диалога/сессии пользователя (thread_id)"
    )
    status: str = Field(..., description="Канонический статус запуска в БД")
    created_at: datetime = Field(..., description="Время создания запуска")
    updated_at: datetime = Field(..., description="Время последнего обновления статуса")


class DocumentResponse(BaseModel):
    """Публичные метаданные документа сессии (без внутреннего storage_key)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Уникальный идентификатор документа")
    session_id: str = Field(..., description="Идентификатор сессии")
    filename: str = Field(..., description="Исходное имя файла")
    content_type: str = Field(..., description="MIME-тип файла")
    size_bytes: int = Field(..., description="Размер файла в байтах")
    doc_type: str | None = Field(default=None, description="Тип документа ОП")
    derived_from: UUID | None = Field(
        default=None, description="ID исходного документа (для версионирования)"
    )
    created_at: datetime = Field(..., description="Время загрузки")


class DocumentListResponse(BaseModel):
    """Список документов сессии."""

    items: list[DocumentResponse] = Field(default_factory=list)
