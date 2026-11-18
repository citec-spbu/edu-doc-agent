"""Ссылка на место в исходном документе (общая для правил и LLM-извлечения)."""

from pydantic import BaseModel, ConfigDict, Field


class SourceRef(BaseModel):
    """Место в исходном документе."""

    model_config = ConfigDict(extra="forbid")

    file: str
    section: str | None = None
    line: int | None = Field(default=None, description="Номер строки в md-файле, с 1")
    quote: str | None = Field(default=None, description="Исходная строка Markdown или ячейка XLSX")


def attach_quotes(model: BaseModel, markdown: str) -> None:
    """Дополняет ссылки нового документа исходными строками, сохраняя номера строк."""
    lines = markdown.splitlines()

    def visit(value: object) -> None:
        if isinstance(value, SourceRef):
            if value.quote is None and value.line is not None and 1 <= value.line <= len(lines):
                value.quote = lines[value.line - 1]
        elif isinstance(value, BaseModel):
            for name in type(value).model_fields:
                visit(getattr(value, name))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list | tuple):
            for item in value:
                visit(item)

    visit(model)
