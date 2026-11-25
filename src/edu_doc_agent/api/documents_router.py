"""Роутер для загрузки, просмотра, скачивания и удаления документов сессии."""

import asyncio
import logging
from pathlib import Path as FilePath
from typing import Annotated
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
    DocumentDAO,
    FileStorage,
    InvalidStoragePathError,
    StorageFileNotFoundError,
)
from edu_doc_agent.parsing import supported_suffixes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["documents"])

_READ_CHUNK_SIZE = 64 * 1024  # 64 KiB


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
    filename = (file.filename or "").strip()
    if not filename:
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

    max_bytes = get_settings().max_upload_file_bytes
    content = await _read_upload_bounded(file, max_bytes)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Загружаемый файл пуст.",
        )

    doc_id = uuid4()
    content_type = file.content_type or "application/octet-stream"

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

    try:
        doc = await dao.create_document(
            session_id=session_id,
            doc_id=doc_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            storage_key=storage_key,
            doc_type=doc_type,
        )
    except BaseException:
        await asyncio.shield(_cleanup_storage(storage, storage_key))
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
    except StorageFileNotFoundError as exc:
        logger.error(
            "Файл отсутствует в хранилище для существующей записи БД: doc_id=%s, storage_key=%s",
            doc_id,
            doc.storage_key,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Файл документа отсутствует в хранилище.",
        ) from exc

    encoded_name = quote(doc.filename, safe="")
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"}
    return Response(content=content, media_type=doc.content_type, headers=headers)


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
