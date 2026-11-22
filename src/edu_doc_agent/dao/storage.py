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
_MAX_FILENAME_BYTES = 255


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


def _has_control_chars(value: str) -> bool:
    """Проверяет наличие NUL-байта и управляющих символов (ASCII < 32 или 127)."""
    return any(ord(ch) < 32 or ch == "\x7f" for ch in value)


def _validate_segment(value: str, label: str) -> str:
    """Проверяет, что сегмент пути (session_id / doc_id) безопасен и не содержит слэшей."""
    if not value or value in {".", ".."} or not _SAFE_SEGMENT_RE.match(value):
        raise InvalidStoragePathError(f"Некорректный {label}: {value!r}")
    return value


def _validate_filename(filename: str) -> str:
    """Отклоняет пустые имена, управляющие символы и попытки path traversal в имени файла."""
    if _has_control_chars(filename):
        raise InvalidStoragePathError(f"Управляющие символы в имени файла: {filename!r}")
    if "/" in filename or "\\" in filename:
        raise InvalidStoragePathError(f"Разделители пути запрещены в имени файла: {filename!r}")
    name = filename.strip()
    if not name or name in {".", ".."}:
        raise InvalidStoragePathError(f"Некорректное имя файла: {filename!r}")
    return name


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Обрезает строку до max_bytes в UTF-8, не разрывая многобайтовые символы."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _fit_filename(prefix: str, filename: str, max_bytes: int = _MAX_FILENAME_BYTES) -> str:
    """Ограничивает длину итогового имени `{prefix}{filename}` до 255 байт в UTF-8,
    сохраняя расширение файла (например, `.docx`)."""
    full = f"{prefix}{filename}"
    if len(full.encode("utf-8")) <= max_bytes:
        return full

    p = Path(filename)
    suffix = p.suffix
    stem = p.stem

    budget = max_bytes - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
    if budget <= 0:
        # Если само расширение аномально длинное — обрезаем имя целиком.
        trimmed = _truncate_utf8(filename, max_bytes - len(prefix.encode("utf-8")))
        return f"{prefix}{trimmed}"

    trimmed_stem = _truncate_utf8(stem, budget).rstrip(" .")
    return f"{prefix}{trimmed_stem or 'file'}{suffix}"


class LocalFileStorage:
    """Локальная реализация FileStorage поверх директории / Docker Volume."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir.resolve()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def resolve_path(self, storage_key: str) -> Path:
        """Преобразует `storage_key` в абсолютный Path и проверяет отсутствие path traversal."""
        if (
            not storage_key
            or _has_control_chars(storage_key)
            or "\\" in storage_key
            or Path(storage_key).is_absolute()
        ):
            raise InvalidStoragePathError(f"Некорректный storage_key: {storage_key!r}")

        parts = storage_key.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise InvalidStoragePathError(f"Недопустимые сегменты в storage_key: {storage_key!r}")

        candidate = (self._base_dir / storage_key).resolve()
        if not candidate.is_relative_to(self._base_dir) or candidate == self._base_dir:
            raise InvalidStoragePathError(f"Путь выходит за пределы хранилища: {storage_key!r}")
        return candidate

    def _build_key(self, session_id: str, doc_id: UUID | str, filename: str) -> str:
        safe_session = _validate_segment(session_id, "session_id")
        safe_doc_id = _validate_segment(str(doc_id), "doc_id")
        safe_name = _validate_filename(filename)
        final_name = _fit_filename(f"{safe_doc_id}_", safe_name)
        return f"{safe_session}/{final_name}"

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
                os.chmod(tmp_path, 0o644)
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
            if target.is_dir():
                raise InvalidStoragePathError(f"Ожидался файл, а не директория: {storage_key!r}")
            try:
                target.unlink()
            except FileNotFoundError:
                return False
            return True

        return await asyncio.to_thread(_unlink)
