"""Ссылка на место в исходном документе (общая для правил и LLM-извлечения)."""

from pydantic import BaseModel, ConfigDict, Field


class SourceRef(BaseModel):
    """Место в исходном документе."""

    model_config = ConfigDict(extra="forbid")

    file: str
    section: str | None = None
    line: int | None = Field(default=None, description="Номер строки в md-файле, с 1")
