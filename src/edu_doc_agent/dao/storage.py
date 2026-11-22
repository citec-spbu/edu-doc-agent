"""Файловое хранилище слоя Data Access (DAO).

Изолирует API и воркер от прямой работы с файловой системой.
В single-node MVP используется LocalFileStorage поверх общего Docker Volume;
в дальнейшем может быть заменён на S3/MinIO без изменения бизнес-логики.
"""

import asyncio
import errno
import os
import re
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

_SAFE_SEGMENT_RE = re.compile(r"[A-Za-z0-9._-]+")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_MAX_COMPONENT_BYTES = 255
_SESSIONS_DIRECTORY = "sessions"


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

    return sha256(_encode_utf8(session_id)).hexdigest()


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


def _key_parts(storage_key: str) -> list[str]:
    if _has_control_chars(storage_key) or "\\" in storage_key:
        raise InvalidStoragePathError(f"Некорректный storage_key: {storage_key!r}")

    parts = storage_key.split("/")
    # Двухкомпонентные ключи сохраняют доступ к файлам прежней раскладки.
    if len(parts) != 2 and not (
        len(parts) == 3 and parts[0] == _SESSIONS_DIRECTORY and _DIGEST_RE.fullmatch(parts[1])
    ):
        raise InvalidStoragePathError(f"Некорректный storage_key: {storage_key!r}")

    for part in parts:
        if part in {"", ".", ".."} or Path(part).drive:
            raise InvalidStoragePathError(f"Недопустимый сегмент пути: {part!r}")
        if len(_encode_utf8(part)) > _MAX_COMPONENT_BYTES:
            raise InvalidStoragePathError(f"Слишком длинный сегмент пути: {part!r}")
    return parts


def _entry_mode(path: Path | str, dir_fd: int | None) -> int:
    mode = os.stat(path, dir_fd=dir_fd, follow_symlinks=False).st_mode
    if stat.S_ISLNK(mode):
        raise InvalidStoragePathError(f"Символические ссылки запрещены: {path!s}")
    return mode


def _open_without_following(path: Path | str, flags: int, dir_fd: int | None) -> int:
    try:
        return os.open(path, flags | os.O_NOFOLLOW, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise InvalidStoragePathError(f"Символические ссылки запрещены: {path!s}") from exc
        raise


def _open_directory(path: Path | str, dir_fd: int | None = None) -> int:
    try:
        return _open_without_following(path, os.O_RDONLY | os.O_DIRECTORY, dir_fd)
    except NotADirectoryError:
        # На Linux сочетание O_DIRECTORY и O_NOFOLLOW даёт ENOTDIR для ссылки.
        with suppress(FileNotFoundError, NotADirectoryError):
            _entry_mode(path, dir_fd)
        raise


class LocalFileStorage:
    """Локальная реализация FileStorage поверх директории / Docker Volume."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir.resolve()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def resolve_path(self, storage_key: str) -> Path:
        """Возвращает путь для осмотра. Файловые операции работают через дескрипторы."""
        candidate = self._base_dir
        for part in _key_parts(storage_key):
            candidate /= part
            if candidate.is_symlink() or candidate.is_junction():
                raise InvalidStoragePathError(f"Ссылки запрещены в storage_key: {storage_key!r}")
        if candidate.resolve() != candidate:
            raise InvalidStoragePathError(f"Путь меняется при разрешении ключа: {storage_key!r}")
        return candidate

    @contextmanager
    def _open_parent(self, parts: list[str], *, create: bool = False) -> Iterator[int]:
        if os.name != "posix":
            raise FileStorageError("LocalFileStorage требует POSIX-операций с дескрипторами")
        if create:
            self._base_dir.mkdir(parents=True, exist_ok=True)

        with ExitStack() as descriptors:
            fd = _open_directory(self._base_dir)
            descriptors.callback(os.close, fd)
            for part in parts[:-1]:
                if create:
                    with suppress(FileExistsError):
                        os.mkdir(part, dir_fd=fd)
                fd = _open_directory(part, fd)
                descriptors.callback(os.close, fd)
            yield fd

    def _build_key(self, session_id: str, doc_id: UUID | str, filename: str) -> str:
        session = _session_directory(session_id)
        document = str(doc_id)
        if document in {".", ".."} or not _SAFE_SEGMENT_RE.fullmatch(document):
            raise InvalidStoragePathError(f"Некорректный doc_id: {doc_id!r}")
        safe_name = _validate_filename(filename)
        # Хешируем полное имя до нормализации и обрезки, чтобы сохранить различия.
        digest = sha256(document.encode("ascii") + b"\0" + _encode_utf8(filename)).hexdigest()
        final_name = _fit_filename(f"~{digest}_", safe_name)
        return f"{_SESSIONS_DIRECTORY}/{session}/{final_name}"

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
        parts = _key_parts(storage_key)
        name = parts[-1]

        def _write_atomic() -> None:
            with self._open_parent(parts, create=True) as parent:
                with suppress(FileNotFoundError):
                    _entry_mode(name, parent)
                temporary_name = f".tmp_{uuid4().hex}"
                fd = os.open(
                    temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent
                )
                try:
                    with os.fdopen(fd, "wb", closefd=False) as stream:
                        stream.write(content)
                    os.replace(temporary_name, name, src_dir_fd=parent, dst_dir_fd=parent)
                except Exception:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary_name, dir_fd=parent)
                    raise
                finally:
                    os.close(fd)

        await asyncio.to_thread(_write_atomic)
        return storage_key

    async def read(self, storage_key: str) -> bytes:
        parts = _key_parts(storage_key)
        name = parts[-1]

        def _read() -> bytes:
            missing = f"Файл не найден: {storage_key!r}"
            try:
                with self._open_parent(parts) as parent:
                    if not stat.S_ISREG(_entry_mode(name, parent)):
                        raise StorageFileNotFoundError(missing)
                    fd = _open_without_following(name, os.O_RDONLY | os.O_NONBLOCK, parent)
                    try:
                        if not stat.S_ISREG(os.fstat(fd).st_mode):
                            raise StorageFileNotFoundError(missing)
                        with os.fdopen(fd, "rb", closefd=False) as stream:
                            return stream.read()
                    finally:
                        os.close(fd)
            except (FileNotFoundError, NotADirectoryError) as exc:
                raise StorageFileNotFoundError(missing) from exc

        return await asyncio.to_thread(_read)

    async def exists(self, storage_key: str) -> bool:
        parts = _key_parts(storage_key)

        def _exists() -> bool:
            try:
                with self._open_parent(parts) as parent:
                    return stat.S_ISREG(_entry_mode(parts[-1], parent))
            except (FileNotFoundError, NotADirectoryError):
                return False

        return await asyncio.to_thread(_exists)

    async def delete(self, storage_key: str) -> bool:
        parts = _key_parts(storage_key)
        name = parts[-1]

        def _unlink() -> bool:
            try:
                with self._open_parent(parts) as parent:
                    if stat.S_ISDIR(_entry_mode(name, parent)):
                        raise InvalidStoragePathError(
                            f"Ожидался файл, а не директория: {storage_key!r}"
                        )
                    os.unlink(name, dir_fd=parent)
            except (FileNotFoundError, NotADirectoryError):
                return False
            except IsADirectoryError as exc:
                raise InvalidStoragePathError(
                    f"Ожидался файл, а не директория: {storage_key!r}"
                ) from exc
            return True

        return await asyncio.to_thread(_unlink)
