"""Фрагменты корпуса нормативов (ADR-0005: фрагмент = пункт норматива).

Поля `id, doc_type, doc_id, edition, path, part, text, codes` — схема таблицы из ADR-0005;
`kind`, `number`, `parent`, `category`, `changed_by` — уточнения для компетенций из таблиц
и отметок об изменениях (вносятся в ADR отдельным изменением).
"""

from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["fgos", "spbu_standard", "prof_standard"]
FragmentKind = Literal["clause", "competency", "prof_standard_ref"]


class ChangeNote(BaseModel):
    """Отметка «Информация об изменениях» (ГАРАНТ): каким актом и с какой даты изменён пункт."""

    since: str | None = None  # «с 1 сентября 2021 г.»
    act: str  # «Приказ Минобрнауки России от 26 ноября 2020 г. N 1456»


class Fragment(BaseModel):
    id: str  # «fgos-02.03.02:2.3», компетенция — «fgos-02.03.02:3.3:ОПК-2»; без редакции
    doc_type: DocType
    doc_id: str
    edition: str
    kind: FragmentKind
    number: str  # номер пункта в документе: «3.3», «2.2.1.6»; у строки приложения — «app»
    path: str  # «III. Требования к результатам освоения программы бакалавриата › 3.3»
    part: int = 0  # часть длинного пункта; склейка частей по порядку = исходный текст
    text: str  # дословно
    codes: list[str] = Field(
        default_factory=list
    )  # канонические, как в данных программы (corpus/codes.py)
    parent: str | None = None  # id пункта, к которому относится компетенция
    category: str | None = None  # категория компетенции или группа профстандартов
    changed_by: list[ChangeNote] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Документ корпуса: метаданные из манифеста или определённые по тексту."""

    doc_id: str  # «fgos-02.03.02», «spbu-os»
    doc_type: DocType
    title: str
    edition: str  # «808 ред. от 27.02.2023»
    approval_act: str | None = None  # реквизиты утверждающего акта
    direction_code: str | None = None
    direction_name: str | None = None
    level: str | None = None  # бакалавриат / магистратура / специалитет
    source_file: str


class Overrides(BaseModel):
    """Ручные правки документа, накладываются после нарезки (исходник не трогаем).

    Нужны там, где автоматика бессильна: ошибки OCR в скане, таблица без рамок.
    `replace`: id -> исправленный текст или словарь полей (`text`, `category`, `path`).
    """

    replace: dict[str, str | dict[str, str]] = Field(default_factory=dict)
    add: list[dict] = Field(default_factory=list)  # новые фрагменты (поля Fragment без doc_*)
    drop: list[str] = Field(default_factory=list)  # id фрагментов, которые убрать


class CorpusDocument(BaseModel):
    """Результат нарезки одного документа: метаданные, фрагменты и замечания сборки."""

    document: DocumentInfo
    fragments: list[Fragment]
    warnings: list[str] = Field(default_factory=list)
