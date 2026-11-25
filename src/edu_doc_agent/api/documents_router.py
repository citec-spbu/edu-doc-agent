"""Роутер для загрузки, просмотра, скачивания и удаления документов сессии."""

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path as FilePath
from typing import Annotated, Any
from urllib.parse import quote
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Response,
    UploadFile,
    status,
)

from edu_doc_agent.api.dependencies import get_document_dao, get_file_storage
from edu_doc_agent.api.schemas import DocumentListResponse, DocumentResponse
from edu_doc_agent.config import get_settings
from edu_doc_agent.dao import (
    ActiveRunConflictError,
    Document,
    DocumentDAO,
    FileStorage,
    FileStorageError,
    InvalidStoragePathError,
)
from edu_doc_agent.parsing import supported_suffixes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["documents"])

_READ_CHUNK_SIZE = 64 * 1024  # 64 KiB
_DEFAULT_CONTENT_TYPE = "application/octet-stream"
_CANONICAL_CONTENT_TYPES: dict[str, str] = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def _has_ascii_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 or ch == "\x7f" for ch in value)


def _validate_session_id(session_id: str) -> str:
    if not session_id or not session_id.strip() or "\x00" in session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Некорректный идентификатор сессии.",
        )
    return session_id


def _validate_upload_filename(raw_filename: str | None) -> tuple[str, str]:
    """Проверяет имя файла и расширение до чтения тела запроса в память."""
    if raw_filename is None or _has_ascii_control_chars(raw_filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Некорректное имя файла.",
        )
    if "/" in raw_filename or "\\" in raw_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Имя файла не должно содержать разделители пути.",
        )
    filename = raw_filename.strip()
    if not filename or filename in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не указано имя файла.",
        )

    suffix = FilePath(filename).suffix.lower()
    allowed = supported_suffixes()
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Неподдерживаемый формат файла '{suffix}'. Допустимы: {', '.join(allowed)}.",
        )
    return filename, suffix


def _ascii_fallback_filename(filename: str) -> str:
    """Безопасное ASCII-имя для старых клиентов в заголовке Content-Disposition."""
    path = FilePath(filename)
    stem = "".join(
        ch if 32 <= ord(ch) < 127 and ch not in {'"', "\\", ";"} else "_" for ch in path.stem
    ).strip(" ._")
    suffix = (
        path.suffix if path.suffix.isascii() and not _has_ascii_control_chars(path.suffix) else ""
    )
    safe_suffix = "".join(ch for ch in suffix if ch not in {'"', "\\", ";"})
    return f"{stem or 'document'}{safe_suffix}"


async def _await_uninterruptible[T](operation: Coroutine[Any, Any, T]) -> T:
    """Дожидается завершения операции даже при повторном CancelledError, затем пробрасывает его."""
    task = asyncio.create_task(operation)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
        else:
            if cancellation is not None:
                raise cancellation
            return result


async def _read_upload_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Читает UploadFile чанками, не позволяя загрузить в RAM больше max_bytes + 1."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_READ_CHUNK_SIZE):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Размер файла превышает допустимый лимит ({max_bytes} байт).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _cleanup_storage(storage: FileStorage, storage_key: str) -> None:
    """Компенсация: удаляет файл из хранилища при сбое записи в БД."""
    try:
        await storage.delete(storage_key)
    except Exception:
        logger.exception("Не удалось удалить файл при откате загрузки: storage_key=%s", storage_key)


@router.post(
    "/{session_id}/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentResponse,
)
async def upload_document(
    session_id: Annotated[str, Path(..., description="ID сессии пользователя")],
    file: Annotated[UploadFile, File(..., description="Файл документа ОП (.docx, .xlsx)")],
    dao: Annotated[DocumentDAO, Depends(get_document_dao)],
    storage: Annotated[FileStorage, Depends(get_file_storage)],
    doc_type: Annotated[str | None, Form(description="Тип документа (rpd, plan, ...)")] = None,
) -> DocumentResponse:
    """Загружает документ в хранилище и сохраняет его метаданные в PostgreSQL."""
    _validate_session_id(session_id)
    filename, suffix = _validate_upload_filename(file.filename)
    normalized_doc_type = (doc_type or "").strip() or None

    max_bytes = get_settings().max_upload_file_bytes
    content = await _read_upload_bounded(file, max_bytes)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Загружаемый файл пуст.",
        )

    doc_id = uuid4()
    content_type = _CANONICAL_CONTENT_TYPES.get(suffix, _DEFAULT_CONTENT_TYPE)

    try:
        storage_key = await storage.save(
            session_id=session_id,
            doc_id=doc_id,
            filename=filename,
            content=content,
        )
    except InvalidStoragePathError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Некорректное имя файла или идентификатор сессии.",
        ) from exc

    db_saved = False

    async def _create_metadata() -> Document:
        nonlocal db_saved
        created = await dao.create_document(
            session_id=session_id,
            doc_id=doc_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            storage_key=storage_key,
            doc_type=normalized_doc_type,
        )
        db_saved = True
        return created

    try:
        doc = await _await_uninterruptible(_create_metadata())
    except BaseException:
        if not db_saved:
            await _await_uninterruptible(_cleanup_storage(storage, storage_key))
        raise

    return DocumentResponse.model_validate(doc)


@router.get(
    "/{session_id}/documents",
    status_code=status.HTTP_200_OK,
    response_model=DocumentListResponse,
)
async def list_documents(
    session_id: Annotated[str, Path(..., description="ID сессии пользователя")],
    dao: Annotated[DocumentDAO, Depends(get_document_dao)],
) -> DocumentListResponse:
    """Возвращает список документов, загруженных в рамках сессии."""
    docs = await dao.list_by_session(session_id)
    return DocumentListResponse(items=[DocumentResponse.model_validate(d) for d in docs])


@router.get(
    "/{session_id}/documents/{doc_id}",
    status_code=status.HTTP_200_OK,
)
async def download_document(
    session_id: Annotated[str, Path(..., description="ID сессии пользователя")],
    doc_id: Annotated[UUID, Path(..., description="ID документа")],
    dao: Annotated[DocumentDAO, Depends(get_document_dao)],
    storage: Annotated[FileStorage, Depends(get_file_storage)],
) -> Response:
    """Возвращает бинарное содержимое документа."""
    doc = await dao.get_document(session_id=session_id, doc_id=doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Документ не найден в рамках указанной сессии.",
        )

    try:
        content = await storage.read(doc.storage_key)
    except FileStorageError as exc:
        logger.error(
            "Ошибка чтения файла из хранилища: doc_id=%s, storage_key=%s, error=%s",
            doc_id,
            doc.storage_key,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Файл документа недоступен в хранилище.",
        ) from exc

    suffix = FilePath(doc.filename).suffix.lower()
    media_type = _CANONICAL_CONTENT_TYPES.get(suffix, _DEFAULT_CONTENT_TYPE)
    fallback_name = _ascii_fallback_filename(doc.filename)
    encoded_name = quote(doc.filename, safe="")
    headers = {
        "Content-Disposition": (
            f"attachment; filename=\"{fallback_name}\"; filename*=UTF-8''{encoded_name}"
        ),
        "X-Content-Type-Options": "nosniff",
    }
    return Response(content=content, media_type=media_type, headers=headers)


@router.delete(
    "/{session_id}/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    session_id: Annotated[str, Path(..., description="ID сессии пользователя")],
    doc_id: Annotated[UUID, Path(..., description="ID документа")],
    dao: Annotated[DocumentDAO, Depends(get_document_dao)],
    storage: Annotated[FileStorage, Depends(get_file_storage)],
) -> Response:
    """Удаляет документ из БД и файлового хранилища (запрещено при активном запуске)."""
    try:
        storage_key = await dao.delete_document(session_id=session_id, doc_id=doc_id)
    except ActiveRunConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Нельзя удалить документ во время активного запуска обработки.",
        ) from exc

    if storage_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Документ не найден в рамках указанной сессии.",
        )

    try:
        await storage.delete(storage_key)
    except Exception:
        logger.exception(
            "Запись документа удалена из БД, но не удалось удалить файл: storage_key=%s",
            storage_key,
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
