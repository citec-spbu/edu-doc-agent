"""Файловое хранилище слоя Data Access (DAO).

Изолирует API и воркер от прямой работы с файловой системой.
В single-node MVP используется LocalFileStorage поверх общего Docker Volume;
в дальнейшем может быть заменён на S3/MinIO без изменения бизнес-логики.
"""

import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol
from uuid import UUID

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class FileStorageError(Exception):
    """Базовая ошибка файлового хранилища."""


class InvalidStoragePathError(FileStorageError):
    """Некорректный путь или попытка выйти за пределы корневой директории (path traversal)."""


class StorageFileNotFoundError(FileStorageError):
    """Запрошенный файл не найден в хранилище."""


class FileStorage(Protocol):
    """Контракт файлового хранилища для API, воркера и предметных служб."""

    async def save(
        self,
        *,
        session_id: str,
        doc_id: UUID | str,
        filename: str,
        content: bytes,
    ) -> str:
        """Сохраняет файл и возвращает его относительный ключ (`storage_key`)."""
        ...

    async def read(self, storage_key: str) -> bytes:
        """Читает содержимое файла по `storage_key`. Бросает StorageFileNotFoundError."""
        ...

    async def exists(self, storage_key: str) -> bool:
        """Проверяет, существует ли файл по `storage_key`."""
        ...

    async def delete(self, storage_key: str) -> bool:
        """Удаляет файл. Возвращает True, если файл был удалён, False — если его не было."""
        ...


def _validate_segment(value: str, label: str) -> str:
    """Проверяет, что сегмент пути (session_id / doc_id) безопасен и не содержит слэшей."""
    if not value or value in {".", ".."} or not _SAFE_SEGMENT_RE.match(value):
        raise InvalidStoragePathError(f"Некорректный {label}: {value!r}")
    return value


def _sanitize_filename(filename: str) -> str:
    """Оставляет только имя файла без директорий и проверяет, что оно не пустое."""
    name = Path(filename.replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise InvalidStoragePathError(f"Некорректное имя файла: {filename!r}")
    return name


class LocalFileStorage:
    """Локальная реализация FileStorage поверх директории / Docker Volume."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir.resolve()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def resolve_path(self, storage_key: str) -> Path:
        """Преобразует `storage_key` в абсолютный Path и проверяет отсутствие path traversal."""
        if not storage_key or Path(storage_key).is_absolute():
            raise InvalidStoragePathError(f"Некорректный storage_key: {storage_key!r}")

        candidate = (self._base_dir / storage_key).resolve()
        if not candidate.is_relative_to(self._base_dir) or candidate == self._base_dir:
            raise InvalidStoragePathError(f"Путь выходит за пределы хранилища: {storage_key!r}")
        return candidate

    def _build_key(self, session_id: str, doc_id: UUID | str, filename: str) -> str:
        safe_session = _validate_segment(session_id, "session_id")
        safe_doc_id = _validate_segment(str(doc_id), "doc_id")
        safe_name = _sanitize_filename(filename)
        return f"{safe_session}/{safe_doc_id}_{safe_name}"

    async def save(
        self,
        *,
        session_id: str,
        doc_id: UUID | str,
        filename: str,
        content: bytes,
    ) -> str:
        storage_key = self._build_key(session_id, doc_id, filename)
        target = self.resolve_path(storage_key)

        def _write_atomic() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=target.parent, prefix=".tmp_")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(content)
                os.replace(tmp_path, target)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

        await asyncio.to_thread(_write_atomic)
        return storage_key

    async def read(self, storage_key: str) -> bytes:
        target = self.resolve_path(storage_key)

        def _read() -> bytes:
            if not target.is_file():
                raise StorageFileNotFoundError(f"Файл не найден: {storage_key!r}")
            return target.read_bytes()

        return await asyncio.to_thread(_read)

    async def exists(self, storage_key: str) -> bool:
        target = self.resolve_path(storage_key)
        return await asyncio.to_thread(target.is_file)

    async def delete(self, storage_key: str) -> bool:
        target = self.resolve_path(storage_key)

        def _unlink() -> bool:
            try:
                target.unlink()
            except FileNotFoundError:
                return False
            return True

        return await asyncio.to_thread(_unlink)
