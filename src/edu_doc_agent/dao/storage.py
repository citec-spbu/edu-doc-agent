"""Файловое хранилище слоя Data Access (DAO).

Изолирует API и воркер от прямой работы с файловой системой.
В single-node MVP используется LocalFileStorage поверх общего Docker Volume;
в дальнейшем может быть заменён на S3/MinIO без изменения бизнес-логики.
"""

import asyncio
import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from uuid import UUID

_SAFE_SEGMENT_RE = re.compile(r"[A-Za-z0-9._-]+")
_MAX_COMPONENT_BYTES = 255


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


def _encode_utf8(value: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidStoragePathError("Компонент пути должен кодироваться в UTF-8") from exc


def _session_directory(session_id: str) -> str:
    if (
        session_id in {"", ".", ".."}
        or "/" in session_id
        or "\\" in session_id
        or _has_control_chars(session_id)
    ):
        raise InvalidStoragePathError(f"Некорректный session_id: {session_id!r}")

    encoded = _encode_utf8(session_id)
    if len(encoded) <= _MAX_COMPONENT_BYTES and _SAFE_SEGMENT_RE.fullmatch(session_id):
        return session_id
    # Символ '~' резервирует отдельное пространство имён для кодированных ID.
    return f"~{sha256(encoded).hexdigest()}"


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
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _fit_filename(prefix: str, filename: str) -> str:
    """Умещает имя с ASCII-префиксом в 255 байт, по возможности сохраняя расширение."""
    budget = _MAX_COMPONENT_BYTES - len(prefix)
    if len(filename.encode("utf-8")) <= budget:
        return f"{prefix}{filename}"

    path = Path(filename)
    suffix = path.suffix
    stem_budget = budget - len(suffix.encode("utf-8"))
    if stem_budget < 0:
        return f"{prefix}{_truncate_utf8(filename, budget)}"

    stem = _truncate_utf8(path.stem, stem_budget).rstrip(" .")
    return f"{prefix}{stem}{suffix}"


class LocalFileStorage:
    """Локальная реализация FileStorage поверх директории / Docker Volume."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir.resolve()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def resolve_path(self, storage_key: str) -> Path:
        """Преобразует ключ в путь внутри хранилища без символических ссылок."""
        if (
            not storage_key
            or _has_control_chars(storage_key)
            or "\\" in storage_key
            or storage_key.startswith("/")
        ):
            raise InvalidStoragePathError(f"Некорректный storage_key: {storage_key!r}")

        candidate = self._base_dir
        for part in storage_key.split("/"):
            if part in {"", ".", ".."} or Path(part).drive:
                raise InvalidStoragePathError(f"Недопустимый сегмент пути: {part!r}")
            if len(_encode_utf8(part)) > _MAX_COMPONENT_BYTES:
                raise InvalidStoragePathError(f"Слишком длинный сегмент пути: {part!r}")
            candidate /= part
            if candidate.is_symlink() or candidate.is_junction():
                raise InvalidStoragePathError(f"Ссылки запрещены в storage_key: {storage_key!r}")
        if candidate.resolve() != candidate:
            raise InvalidStoragePathError(f"Путь меняется при разрешении ключа: {storage_key!r}")
        return candidate

    def _build_key(self, session_id: str, doc_id: UUID | str, filename: str) -> str:
        session = _session_directory(session_id)
        document = str(doc_id)
        if document in {".", ".."} or not _SAFE_SEGMENT_RE.fullmatch(document):
            raise InvalidStoragePathError(f"Некорректный doc_id: {doc_id!r}")
        safe_name = _validate_filename(filename)
        # Хешируем полное имя до нормализации и обрезки, чтобы сохранить различия.
        digest = sha256(document.encode("ascii") + b"\0" + _encode_utf8(filename)).hexdigest()
        final_name = _fit_filename(f"~{digest}_", safe_name)
        return f"{session}/{final_name}"

    async def save(
        self,
        *,
        session_id: str,
        doc_id: UUID | str,
        filename: str,
        content: bytes,
    ) -> str:
        """Сохраняет файл атомарно, заменяя содержимое при повторной записи по тому же ключу."""
        storage_key = self._build_key(session_id, doc_id, filename)

        def _write_atomic() -> None:
            target = self.resolve_path(storage_key)
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
        def _read() -> bytes:
            target = self.resolve_path(storage_key)
            try:
                if not target.is_file():
                    raise StorageFileNotFoundError(f"Файл не найден: {storage_key!r}")
                return target.read_bytes()
            except (FileNotFoundError, NotADirectoryError) as exc:
                raise StorageFileNotFoundError(f"Файл не найден: {storage_key!r}") from exc

        return await asyncio.to_thread(_read)

    async def exists(self, storage_key: str) -> bool:
        def _exists() -> bool:
            return self.resolve_path(storage_key).is_file()

        return await asyncio.to_thread(_exists)

    async def delete(self, storage_key: str) -> bool:
        def _unlink() -> bool:
            target = self.resolve_path(storage_key)
            if target.is_dir():
                raise InvalidStoragePathError(f"Ожидался файл, а не директория: {storage_key!r}")
            try:
                target.unlink()
            except FileNotFoundError:
                return False
            return True

        return await asyncio.to_thread(_unlink)
