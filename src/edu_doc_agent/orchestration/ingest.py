"""Парсер + извлечение (ADR-0003, шаг 2): файл сессии → данные документа в хранилище.

Тип документа определяется по титулу, затем данные извлекает модуль `extraction/`
(ADR-0002): РПД и РПП, учебный план, общая характеристика, ЦКМВ, матрица компетенций.
Парсинг и извлечение синхронные и тяжёлые, поэтому идут в отдельном потоке.
"""

import asyncio
import re
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from edu_doc_agent.extraction.characteristic import extract_characteristic
from edu_doc_agent.extraction.ckmv import extract_ckmv
from edu_doc_agent.extraction.curriculum import extract_plan
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.matrix import extract_matrix
from edu_doc_agent.extraction.rpd import extract_rpd
from edu_doc_agent.llm import LLMClient
from edu_doc_agent.orchestration.stores import StoredDocument
from edu_doc_agent.parsing import parse_document

_TITLE_LINES = 40
# имя файла в FileStorage: «~<sha256>_<имя>» (dao/storage.py)
_STORAGE_PREFIX = re.compile(r"^~[0-9a-f]{64}_")
# признаки типа в титуле, по порядку проверки; пробелы убраны («р а б о ч а я»)
_TITLE_MARKS = (
    ("rpd", "рабочаяпрограмма"),
    ("characteristic", "общаяхарактеристика"),
    ("ckmv", "профессиональныхролей"),
    ("plan", "учебныйплан"),
)


class UnknownDocumentError(ValueError):
    """Тип документа не удалось определить или он не поддерживается."""


class DocumentIngestor(Protocol):
    async def ingest(self, session_id: str, path: str) -> StoredDocument: ...


def document_id(session_id: str, path: str) -> str:
    """ID документа по имени файла: новая загрузка файла с тем же именем (исправленная
    версия) заменяет данные прежней, ID замечаний по документу сохраняются."""
    return sha256(f"{session_id}\0{original_filename(path)}".encode()).hexdigest()[:16]


def original_filename(path: str) -> str:
    return _STORAGE_PREFIX.sub("", Path(path).name)


def detect_doc_type(markdown: str) -> str | None:
    title = normalize("\n".join(markdown.splitlines()[:_TITLE_LINES])).replace(" ", "")
    for doc_type, mark in _TITLE_MARKS:
        if mark in title:
            return doc_type
    return None


class ExtractionIngestor:
    """Извлечение модулем `extraction/`; `llm` добавляет смысловые поля РПД (ADR-0002)."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    async def ingest(self, session_id: str, path: str) -> StoredDocument:
        return await asyncio.to_thread(self._ingest, session_id, path)

    def _ingest(self, session_id: str, path: str) -> StoredDocument:
        file = original_filename(path)
        doc_id = document_id(session_id, path)
        suffix = Path(path).suffix.lower()
        if suffix == ".xlsx":
            matrix = extract_matrix(Path(path), file=file)
            if not matrix.rows:
                raise UnknownDocumentError(
                    f"{file}: из XLSX поддерживается только матрица компетенций"
                )
            return StoredDocument(doc_id, session_id, "matrix", matrix, file, "", path)
        # Markdown — уже разобранный документ (так хранятся тестовые документы)
        parsed = None if suffix == ".md" else parse_document(path)
        text = parsed.to_markdown() if parsed else Path(path).read_text(encoding="utf-8")
        doc_type = detect_doc_type(text)
        if doc_type == "rpd":
            data = extract_rpd(parsed or text, file=file, llm=self.llm)
        elif doc_type == "characteristic":
            data = extract_characteristic(text, file=file)
        elif doc_type == "ckmv":
            data = extract_ckmv(text, file=file)
        elif doc_type == "plan":
            data = extract_plan(text, file=file)
        else:
            raise UnknownDocumentError(f"{file}: не удалось определить тип документа по титулу")
        return StoredDocument(doc_id, session_id, doc_type, data, file, text, path)
