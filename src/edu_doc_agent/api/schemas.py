"""
Pydantic-схемы данных для API.
Описывают форматы входящих запросов и исходящих ответов.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


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
