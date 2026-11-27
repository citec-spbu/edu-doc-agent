"""Документы программы сессии для проверки и код направления."""

import re

from langchain_core.runnables import RunnableConfig

from edu_doc_agent.extraction.characteristic import ProgramCharacteristic
from edu_doc_agent.extraction.ckmv import CkmvDocument
from edu_doc_agent.extraction.curriculum import CurriculumPlan
from edu_doc_agent.extraction.matrix import CompetenceMatrix
from edu_doc_agent.extraction.models import RPDExtraction
from edu_doc_agent.orchestration.deps import GraphDeps
from edu_doc_agent.orchestration.stores import StoredDocument
from edu_doc_agent.validation.units import CourseDoc, ProgramDocuments

_DIRECTION_RE = re.compile(r"\b(\d{2}\.\d{2}\.\d{2})\b")
# документы программы, которые проверяются в одном экземпляре
_PROGRAM_KINDS = {
    "plan": CurriculumPlan,
    "characteristic": ProgramCharacteristic,
    "ckmv": CkmvDocument,
    "matrix": CompetenceMatrix,
}


def session_id_of(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


def _first[T](documents: list[StoredDocument], model: type[T]) -> T | None:
    found = [d.data for d in documents if isinstance(d.data, model)]
    return found[-1] if found else None  # последний загруженный документ этого типа


def program_documents(documents: list[StoredDocument]) -> tuple[ProgramDocuments, dict[str, str]]:
    courses = tuple(
        CourseDoc(d.document_id, d.data, d.filename, d.text)
        for d in documents
        if isinstance(d.data, RPDExtraction)
    )
    texts = {d.filename: d.text for d in documents}
    files_by_kind = {
        kind: tuple(d.filename for d in documents if isinstance(d.data, model))
        for kind, model in _PROGRAM_KINDS.items()
    }
    program = ProgramDocuments(
        courses=courses,
        plan=_first(documents, CurriculumPlan),
        characteristic=_first(documents, ProgramCharacteristic),
        ckmv=_first(documents, CkmvDocument),
        matrix=_first(documents, CompetenceMatrix),
        texts=texts,
        files_by_kind={k: v for k, v in files_by_kind.items() if v},
    )
    return program, texts


async def load_program(deps: GraphDeps, session_id: str) -> tuple[ProgramDocuments, dict[str, str]]:
    """Документы сессии; кэш процесса сбрасывает `invalidate` после записи документов."""
    cached = deps.cache.get(session_id)
    if cached is None:
        cached = program_documents(await deps.stores.list_documents(session_id))
        deps.cache[session_id] = cached
    return cached


def invalidate(deps: GraphDeps, session_id: str) -> None:
    deps.cache.pop(session_id, None)


def direction_codes(documents: list[StoredDocument]) -> list[str]:
    """Коды направления из документов программы: план, характеристика, РПД."""
    codes: list[str] = []
    for d in documents:
        data = d.data
        values: list[str | None] = []
        if isinstance(data, CurriculumPlan):
            values = [data.direction]
        elif isinstance(data, ProgramCharacteristic):
            values = [data.direction_code]
        elif isinstance(data, RPDExtraction) and data.program:
            values = [data.program.direction_code]
        for v in values:
            if v and (m := _DIRECTION_RE.search(v)):
                codes.append(m.group(1))
    return list(dict.fromkeys(codes))
