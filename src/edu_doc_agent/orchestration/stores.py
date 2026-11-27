"""Данные сессии вне состояния графа: извлечённые документы, замечания и изменения.

В состоянии графа лежат только ID (ADR-0003), сами данные хранятся в хранилище LangGraph
(`BaseStore`): в воркере это `AsyncPostgresStore`, в тестах `InMemoryStore`. Пространство
имён у каждой сессии своё, поэтому ID замечания уровня программы не пересекаются между
сессиями.
"""

from dataclasses import dataclass
from typing import Any, Literal

from langgraph.store.base import BaseStore
from pydantic import BaseModel

from edu_doc_agent.extraction.characteristic import ProgramCharacteristic
from edu_doc_agent.extraction.ckmv import CkmvDocument
from edu_doc_agent.extraction.curriculum import CurriculumPlan
from edu_doc_agent.extraction.matrix import CompetenceMatrix
from edu_doc_agent.extraction.models import RPDExtraction
from edu_doc_agent.validation.findings import Finding

# тип документа → модель результата извлечения
DOC_MODELS: dict[str, type[BaseModel]] = {
    "rpd": RPDExtraction,
    "plan": CurriculumPlan,
    "characteristic": ProgramCharacteristic,
    "ckmv": CkmvDocument,
    "matrix": CompetenceMatrix,
}

DOC_TYPE_TITLES = {
    "rpd": "РПД/РПП",
    "plan": "учебный план",
    "characteristic": "общая характеристика",
    "ckmv": "ЦКМВ",
    "matrix": "матрица компетенций",
}

_PAGE = 500


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    session_id: str
    doc_type: str  # ключ DOC_MODELS
    data: BaseModel  # результат extraction: RPDExtraction, CurriculumPlan, ...
    filename: str = ""
    text: str = ""  # Markdown документа: для цитат, поиска и чтения разделов
    path: str = ""  # исходный файл сессии: из него собирается исправленный документ
    file_key: str = ""  # ключ FileStorage исправленной версии; пусто — исходный файл


class ChangeEdit(BaseModel):
    document_id: str
    path: str  # путь к значению в данных документа: «credits», «workload_table/2/credits»
    # set — заменить значение; remove — удалить элемент списка; append — добавить в список
    op: Literal["set", "remove", "append"] = "set"
    old: Any = None
    new: Any = None


class Change(BaseModel):
    """Изменение от подграфа «Генерация»: правки, последствия и статус подтверждения."""

    id: str
    session_id: str
    run_id: str
    goal: str
    edits: list[ChangeEdit]
    rationale: str = ""
    finding_ids: list[str] = []
    skipped_finding_ids: list[str] = []  # замечания цели, которые пользователь просил оставить
    user_paths: list[str] = []  # пути правок, значение которых дал пользователь
    consequences: list[str] = []
    status: Literal["draft", "applied", "rejected", "superseded", "failed"] = "draft"
    problems: list[str] = []  # при status=failed: что не удалось исправить
    document_notes: list[str] = []  # перенос правок в файлы документов: что внести вручную


@dataclass
class SessionStores:
    store: BaseStore

    # --- документы ---------------------------------------------------------------------------

    async def save_document(self, document: StoredDocument) -> None:
        """Повторное сохранение того же ID заменяет прежние данные."""
        await self.store.aput(
            ("documents", document.session_id),
            document.document_id,
            {
                "doc_type": document.doc_type,
                "filename": document.filename,
                "text": document.text,
                "path": document.path,
                "file_key": document.file_key,
                "data": document.data.model_dump(mode="json"),
            },
            index=False,
        )

    async def get_document(self, session_id: str, document_id: str) -> StoredDocument | None:
        item = await self.store.aget(("documents", session_id), document_id)
        return None if item is None else _document(session_id, item.key, item.value)

    async def list_documents(
        self, session_id: str, doc_type: str | None = None
    ) -> list[StoredDocument]:
        """Документы сессии в порядке первой загрузки, при `doc_type` — только этого типа."""
        items = await self._all(("documents", session_id))
        documents = [_document(session_id, i.key, i.value) for i in items]
        return [d for d in documents if doc_type is None or d.doc_type == doc_type]

    # --- замечания ---------------------------------------------------------------------------

    async def save_findings(self, session_id: str, findings: list[Finding]) -> None:
        """Замечание с тем же ID заменяется: повтор шага не создаёт дублей."""
        for f in findings:
            await self.store.aput(
                ("findings", session_id), f.id, f.model_dump(mode="json"), index=False
            )

    async def delete_findings(self, session_id: str, finding_ids: list[str]) -> None:
        for fid in finding_ids:
            await self.store.adelete(("findings", session_id), fid)

    async def get_findings(self, session_id: str, finding_ids: list[str]) -> list[Finding]:
        """Замечания в порядке `finding_ids`; отсутствующие пропускаются."""
        out = []
        for fid in finding_ids:
            item = await self.store.aget(("findings", session_id), fid)
            if item is not None:
                out.append(Finding.model_validate(item.value))
        return out

    # --- изменения ---------------------------------------------------------------------------

    async def save_change(self, change: Change) -> None:
        await self.store.aput(
            ("changes", change.session_id), change.id, change.model_dump(mode="json"), index=False
        )

    async def get_change(self, session_id: str, change_id: str) -> Change | None:
        item = await self.store.aget(("changes", session_id), change_id)
        return None if item is None else Change.model_validate(item.value)

    async def _all(self, namespace: tuple[str, ...]) -> list:
        items, offset = [], 0
        while True:
            page = await self.store.asearch(namespace, limit=_PAGE, offset=offset)
            items += page
            if len(page) < _PAGE:
                break
            offset += _PAGE
        return sorted(items, key=lambda i: (i.created_at, i.key))


def _document(session_id: str, key: str, value: dict) -> StoredDocument:
    model = DOC_MODELS[value["doc_type"]]
    return StoredDocument(
        document_id=key,
        session_id=session_id,
        doc_type=value["doc_type"],
        data=model.model_validate(value["data"]),
        filename=value.get("filename", ""),
        text=value.get("text", ""),
        path=value.get("path", ""),
        file_key=value.get("file_key", ""),
    )
