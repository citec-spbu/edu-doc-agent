"""Корпус нормативов для RAG: нарезка документов на фрагменты (ADR-0005)."""

from edu_doc_agent.retrieval.corpus.chunker import PROFILES, chunk_document, detect_info
from edu_doc_agent.retrieval.corpus.models import (
    ChangeNote,
    CorpusDocument,
    DocumentInfo,
    Fragment,
    Overrides,
)

__all__ = [
    "PROFILES",
    "ChangeNote",
    "CorpusDocument",
    "DocumentInfo",
    "Fragment",
    "Overrides",
    "chunk_document",
    "detect_info",
]
