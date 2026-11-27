"""Замечания проверки.

ID замечания детерминированный: правило и единица проверки (документ или программа).
Повторная проверка той же пары заменяет прежнее замечание, поэтому повтор шага не создаёт
дублей (ADR-0003). Хранит замечания `orchestration/stores.py`.
"""

from typing import Literal

from pydantic import BaseModel

from edu_doc_agent.extraction.source import SourceRef
from edu_doc_agent.validation.catalog import Requirement, RuleSet

FindingStatus = Literal["violation", "not_checked"]


def finding_id(rule_id: str, document_id: str | None) -> str:
    return f"{rule_id}:{document_id or 'program'}"


class Evidence(BaseModel):
    """Дословная цитата из документа программы в обоснование нарушения."""

    document: str  # имя файла
    quote: str


class Finding(BaseModel):
    id: str
    session_id: str
    run_id: str
    rule_id: str
    rule_title: str
    rule_set: RuleSet
    severity: Literal["error", "warning"]
    fixable: bool
    method: Literal["code", "llm"] = "code"
    document_id: str | None  # None: правило уровня программы
    course: str | None  # название дисциплины для протокола
    status: FindingStatus
    message: str
    sources: list[SourceRef] = []
    evidence: list[Evidence] = []
    requirement: Requirement | None = None
    needs_human: bool = False  # чекер и ревьюер не согласились за 3 раунда
    opinions: list[str] = []  # мнения агентов при needs_human
