"""Корпус нормативов для RAG: нарезка документов на фрагменты (ADR-0005).

ФГОС и ОС СПбГУ — chunker.py, федеральные акты из ГАРАНТа — acts.py,
профстандарты — prof_standard.py.
"""

from edu_doc_agent.retrieval.corpus.acts import chunk_act
from edu_doc_agent.retrieval.corpus.chunker import PROFILES, chunk_document, detect_info
from edu_doc_agent.retrieval.corpus.models import (
    ChangeNote,
    CorpusDocument,
    DocumentInfo,
    Fragment,
    Overrides,
)
from edu_doc_agent.retrieval.corpus.prof_standard import chunk_prof_standard

__all__ = [
    "PROFILES",
    "ChangeNote",
    "CorpusDocument",
    "DocumentInfo",
    "Fragment",
    "Overrides",
    "chunk_act",
    "chunk_document",
    "chunk_prof_standard",
    "detect_info",
]
