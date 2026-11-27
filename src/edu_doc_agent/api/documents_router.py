"""Роутер для загрузки, просмотра, скачивания и удаления документов сессии."""

import asyncio
import json
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
from fastapi.routing import APIRoute

from edu_doc_agent.api.dependencies import get_document_dao, get_file_storage
from edu_doc_agent.api.multipart import BoundedMultipartRoute
from edu_doc_agent.api.schemas import DocumentListResponse, DocumentResponse
from edu_doc_agent.config import get_settings
from edu_doc_agent.dao import (
    ActiveRunConflictError,
    DocumentDAO,
    FileStorage,
    FileStorageError,
    InvalidStoragePathError,
    LocalFileStorage,
)
from edu_doc_agent.dao.documents import DocumentCreateRolledBackError
from edu_doc_agent.parsing import supported_suffixes
from edu_doc_agent.reconcile_uploads import clear_upload_marker, record_upload_rollback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["documents"], route_class=BoundedMultipartRoute)

_READ_CHUNK_SIZE = 64 * 1024  # 64 KiB
_DEFAULT_CONTENT_TYPE = "application/octet-stream"
_CANONICAL_CONTENT_TYPES: dict[str, str] = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pdf": "application/pdf",
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


async def _reject_empty_session_id() -> None:
    """Пустой сегмент не совпадает с {session_id}, поэтому отклоняется явно."""
    _validate_session_id("")


# Только четыре операции Documents API; без зависимостей и разбора multipart.
# UUID-конвертер оставляет неизвестные пути вне этих обработчиков.
for _path, _methods in (
    ("//documents", ["GET", "POST"]),
    ("//documents/{doc_id:uuid}", ["GET", "DELETE"]),
):
    router.add_api_route(
        _path,
        _reject_empty_session_id,
        methods=_methods,
        include_in_schema=False,
        route_class_override=APIRoute,
    )


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


async def _cleanup_storage(storage: FileStorage, storage_key: str) -> bool:
    """Компенсация: удаляет файл из хранилища при сбое записи в БД."""
    try:
        await storage.delete(storage_key)
        return True
    except Exception:
        logger.exception("Не удалось удалить файл при откате загрузки: storage_key=%s", storage_key)
        return False


@router.post(
    "/{session_id}/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentResponse,
)
async def upload_document(
    session_id: Annotated[str, Path(..., description="ID сессии пользователя")],
    file: Annotated[UploadFile, File(..., description="Файл документа ОП (.docx, .xlsx, .pdf)")],
    dao: Annotated[DocumentDAO, Depends(get_document_dao)],
    storage: Annotated[FileStorage, Depends(get_file_storage)],
    doc_type: Annotated[str | None, Form(description="Тип документа (rpd, plan, ...)")] = None,
) -> DocumentResponse:
    """Загружает документ в хранилище и сохраняет его метаданные в PostgreSQL."""
    _validate_session_id(session_id)
    filename, suffix = _validate_upload_filename(file.filename)
    if doc_type is not None and "\x00" in doc_type:
        raise HTTPException(status_code=400, detail="Некорректный тип документа.")
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

    async def _store_document() -> DocumentResponse:
        # Одна защищённая операция удерживает результат save, метаданные и
        # компенсацию. Внешняя (в том числе повторная) отмена не попадает в
        # LocalFileStorage.save и не скрывает ключ после os.replace.
        storage_key = await storage.save(
            session_id=session_id,
            doc_id=doc_id,
            filename=filename,
            content=content,
        )
        # Записываем намерение ДО обращения к БД: при недоступной БД или
        # перезапуске процесса останутся исходные ID и ключ для сверки.
        # Этот служебный файл не имеет метаданных Document и не виден в API.
        try:
            intent_key = await storage.save(
                session_id=session_id,
                doc_id=doc_id,
                filename="upload-reconciliation.json",
                content=json.dumps(
                    {
                        "version": 1,
                        "session_id": session_id,
                        "doc_id": str(doc_id),
                        "storage_key": storage_key,
                    }
                ).encode("utf-8"),
            )
        except BaseException:
            await _cleanup_storage(storage, storage_key)
            raise

        try:
            doc = await dao.create_document(
                session_id=session_id,
                doc_id=doc_id,
                filename=filename,
                content_type=content_type,
                size_bytes=len(content),
                storage_key=storage_key,
                doc_type=normalized_doc_type,
            )
        except DocumentCreateRolledBackError:
            # Свидетельство rollback должно пережить сбой удаления файла
            # и перезапуск API. Обычного намерения и пустого SELECT недостаточно.
            try:
                await record_upload_rollback(
                    storage,
                    session_id=session_id,
                    doc_id=doc_id,
                    storage_key=storage_key,
                    filename=filename,
                )
            except Exception:
                # Текущий вызов знает rollback и ключ прямо из save, даже
                # если сохранить это свидетельство для нового процесса нельзя.
                logger.exception(
                    "Не удалось записать подтверждение rollback: "
                    "session_id=%s doc_id=%s storage_key=%s intent_key=%s",
                    session_id,
                    doc_id,
                    storage_key,
                    intent_key,
                )
            if await _cleanup_storage(storage, storage_key):
                await _cleanup_storage(storage, intent_key)
            raise
        except BaseException as exc:
            # Новый SELECT способен подтвердить COMMIT только найденной
            # строкой с исходным ключом. Пустой результат не доказывает
            # rollback: транзакция на сервере ещё может завершаться.
            doc = None
            try:
                doc = await dao.get_document_for_reconciliation(
                    session_id=session_id, doc_id=doc_id
                )
            except Exception:
                logger.warning("БД недоступна при сверке загрузки: doc_id=%s", doc_id)
            if doc is None or (doc.session_id, doc.id, doc.storage_key) != (
                session_id,
                doc_id,
                storage_key,
            ):
                logger.error(
                    "Исход загрузки неизвестен; файл и намерение сохранены: "
                    "doc_id=%s storage_key=%s intent_key=%s",
                    doc_id,
                    storage_key,
                    intent_key,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Исход загрузки требует сверки. Идентификатор документа: {doc_id}",
                ) from exc

        response = DocumentResponse.model_validate(doc)
        await _cleanup_storage(storage, intent_key)
        return response

    try:
        return await _await_uninterruptible(_store_document())
    except InvalidStoragePathError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Некорректное имя файла или идентификатор сессии.",
        ) from exc


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
    _validate_session_id(session_id)
    docs = await dao.list_by_session(session_id)
    return DocumentListResponse(items=[DocumentResponse.model_validate(d) for d in docs])


@router.get(
    "/{session_id}/documents/{doc_id}",
    status_code=status.HTTP_200_OK,
    response_class=Response,
    responses={
        200: {
            "description": "Исходное бинарное содержимое документа (DOCX, XLSX или PDF).",
            "content": {
                media_type: {"schema": {"type": "string", "format": "binary"}}
                for media_type in _CANONICAL_CONTENT_TYPES.values()
            },
        },
    },
)
async def download_document(
    session_id: Annotated[str, Path(..., description="ID сессии пользователя")],
    doc_id: Annotated[UUID, Path(..., description="ID документа")],
    dao: Annotated[DocumentDAO, Depends(get_document_dao)],
    storage: Annotated[FileStorage, Depends(get_file_storage)],
) -> Response:
    """Возвращает бинарное содержимое документа."""
    _validate_session_id(session_id)
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
    """Принимает удаление до старта Run; принятую очистку можно повторить позже."""
    _validate_session_id(session_id)
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
        # До физического удаления строки остаётся подтверждение COMMIT POST.
        # Ошибка очистки маркера сохраняет deletion_pending для повтора DELETE
        # или локальной сверки; отсутствие публичного GET не используется.
        if isinstance(storage, LocalFileStorage):
            await clear_upload_marker(
                storage, session_id=session_id, doc_id=doc_id, storage_key=storage_key
            )
    except (OSError, FileStorageError) as exc:
        logger.exception(
            "Не удалось удалить файл; метаданные сохранены для повтора DELETE: storage_key=%s",
            storage_key,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось удалить файл документа. Повторите запрос удаления.",
        ) from exc

    # False от storage.delete означает, что файл уже отсутствует: это также
    # успех, в частности после отмены запроса между удалением файла и commit БД.
    await dao.complete_document_deletion(
        session_id=session_id, doc_id=doc_id, storage_key=storage_key
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
