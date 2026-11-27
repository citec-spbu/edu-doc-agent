"""Данные РПД для тестов проверок: только поля, которые нужны правилам."""

from edu_doc_agent.extraction.models import (
    Competence,
    ProgramInfo,
    RPDExtraction,
    WorkloadRow,
)
from edu_doc_agent.extraction.source import SourceRef

SRC = SourceRef(file="rpd.md", section="2.1", line=10)


def row(
    semester: int | None,
    credits: float | None,
    lectures: float,
    self_study: float,
    *,
    study_form: str | None = "очная",
) -> WorkloadRow:
    return WorkloadRow(
        semester=semester,
        is_total=semester is None,
        study_form=study_form,
        contact_hours={"лекции": lectures},
        self_study_hours={"самостоятельная работа": self_study},
        credits=credits,
        source=SRC,
    )


def rpd(
    *,
    title: str = "Математический анализ",
    credits: float | None = 4,
    workload_table: list[WorkloadRow] | None = None,
    competences: list[str] = ("УК-1", "ОПК-2"),
    direction_code: str | None = "02.03.02",
    doc_kind: str | None = "discipline",
    **extra,
) -> RPDExtraction:
    table = (
        workload_table
        if workload_table is not None
        else [row(1, 2, 36, 36), row(2, 2, 36, 36), row(None, 4, 72, 72)]
    )
    return RPDExtraction(
        file="rpd.md",
        title=title,
        credits=credits,
        doc_kind=doc_kind,
        workload_table=table,
        competences=[Competence(code=c, source=SRC) for c in competences],
        program=ProgramInfo(direction_code=direction_code) if direction_code else None,
        **extra,
    )
