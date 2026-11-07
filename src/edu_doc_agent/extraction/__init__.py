"""Извлечение сущностей из документов: компетенции, индикаторы, дисциплины, ..."""

from edu_doc_agent.extraction.models import RPDExtraction
from edu_doc_agent.extraction.rpd import extract_rpd, extract_rpd_file

__all__ = ["RPDExtraction", "extract_rpd", "extract_rpd_file"]
