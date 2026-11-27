"""Локальная сверка маркеров POST: python -m edu_doc_agent.reconcile_uploads.

Удаляет исходные байты только по сохранённому подтверждению rollback.
Отсутствие строки само по себе не доказывает rollback.
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from edu_doc_agent.dao import DocumentDAO, FileStorage, FileStorageError, LocalFileStorage
from edu_doc_agent.dao.storage import StorageFileNotFoundError

INTENT_FILENAME = "upload-reconciliation.json"


class UploadIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    session_id: str = Field(min_length=1)
    doc_id: UUID
    storage_key: str = Field(min_length=1)
    outcome: Literal["unknown", "rolled_back"] = "unknown"


async def record_upload_rollback(
    storage: FileStorage, *, session_id: str, doc_id: UUID, storage_key: str
) -> str:
    """Сохраняет подтверждение DAO до любой попытки удалить исходные байты.

    Вызывать только после DocumentCreateRolledBackError: исключения COMMIT
    и пустой SELECT такого свидетельства не дают. save заменяет маркер атомарно.
    """
    intent = UploadIntent(
        version=1,
        session_id=session_id,
        doc_id=doc_id,
        storage_key=storage_key,
        outcome="rolled_back",
    )
    return await storage.save(
        session_id=session_id,
        doc_id=doc_id,
        filename=INTENT_FILENAME,
        content=intent.model_dump_json().encode("utf-8"),
    )


async def read_intent(storage: LocalFileStorage, key: str) -> UploadIntent:
    intent = UploadIntent.model_validate_json(await storage.read(key))
    if storage.key_for(intent.session_id, intent.doc_id, INTENT_FILENAME) != key:
        raise ValueError("Intent identity does not match its storage key")
    if intent.storage_key == key:
        raise ValueError("Intent cannot refer to itself as document content")
    return intent


async def clear_upload_marker(
    storage: LocalFileStorage, *, session_id: str, doc_id: UUID, storage_key: str
) -> None:
    """Вызывается DELETE после удаления байтов, до удаления подтверждённой строки.

    Ошибка оставляет deletion_pending в БД для повтора. Отсутствующий маркер —
    успех; расхождение идентификаторов не разрешает удалять чужой маркер.
    """
    key = storage.key_for(session_id, doc_id, INTENT_FILENAME)
    try:
        intent = await read_intent(storage, key)
    except StorageFileNotFoundError:
        return
    except (ValueError, ValidationError) as exc:
        raise FileStorageError("Invalid upload reconciliation marker") from exc
    if (intent.session_id, intent.doc_id, intent.storage_key) != (session_id, doc_id, storage_key):
        raise FileStorageError("Upload reconciliation marker identity mismatch")
    await storage.delete(key)


async def reconcile_uploads(dao: DocumentDAO, storage: LocalFileStorage) -> dict[str, int]:
    """Завершает подтверждённые загрузки и компенсации rollback.

    Любая неопределённость сохраняет маркер и байты. При недоступной БД обход
    останавливается: повтор команды безопасен после восстановления сервиса.
    """
    counts: Counter[str] = Counter()
    for key in await storage.list_upload_intents():
        try:
            intent = await read_intent(storage, key)
        except StorageFileNotFoundError:
            counts["already_removed"] += 1
            continue
        except (ValueError, FileStorageError, OSError):
            counts["invalid"] += 1
            continue
        try:
            doc = await dao.get_document_for_reconciliation(intent.session_id, intent.doc_id)
        except Exception:
            counts["database_unavailable"] += 1
            break
        if doc is None:
            if intent.outcome == "rolled_back":
                try:
                    # Маркер удаляется последним: сбой между двумя delete
                    # оставляет свидетельство для нового процесса. Отсутствие
                    # исходного файла при повторе — успешная компенсация.
                    await storage.delete(intent.storage_key)
                    await storage.delete(key)
                except (FileStorageError, OSError):
                    counts["cleanup_failed"] += 1
                    continue
                counts["rolled_back"] += 1
                continue
            counts["unresolved"] += 1
            continue
        if intent.outcome == "rolled_back":
            # Противоречие сохранённому исходу: никакие байты не удаляем.
            counts["mismatch"] += 1
            continue
        if (doc.session_id, doc.id, doc.storage_key) != (
            intent.session_id,
            intent.doc_id,
            intent.storage_key,
        ):
            counts["mismatch"] += 1
            continue
        try:
            await storage.delete(key)
        except (FileStorageError, OSError):
            counts["cleanup_failed"] += 1
            continue
        counts["confirmed"] += 1
    return dict(counts)


def main() -> int:
    from edu_doc_agent.config import get_settings
    from edu_doc_agent.dao.database import engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-dir", type=Path, help="Переопределить STORAGE_DIR")
    args = parser.parse_args()
    root = args.storage_dir or get_settings().storage_dir
    try:
        with os.scandir(root) as entries:
            next(entries, None)
        if not os.access(root, os.R_OK | os.X_OK):
            raise PermissionError("Требуются права чтения и обхода каталога")
    except OSError as exc:
        parser.error(f"Недоступен корень хранилища {root}: {exc}")

    async def run() -> dict[str, int]:
        try:
            return await reconcile_uploads(DocumentDAO(), LocalFileStorage(root))
        finally:
            await engine.dispose()

    try:
        result = asyncio.run(run())
    except OSError as exc:
        print(f"Ошибка доступа к хранилищу {root}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    retained = {"invalid", "database_unavailable", "unresolved", "mismatch", "cleanup_failed"}
    return int(any(result.get(reason, 0) for reason in retained))


if __name__ == "__main__":
    raise SystemExit(main())
