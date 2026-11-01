"""Pydantic-модели результата извлечения из РПД.

Каждая сущность хранит `source` — откуда она взята (файл, раздел, строка),
чтобы в протоколе проверки можно было сослаться на источник.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from edu_doc_agent.extraction.llm_schemas import LLMExtraction
from edu_doc_agent.extraction.source import SourceRef

OutcomeKind = Literal["знает", "умеет", "владеет"]
DocKind = Literal["discipline", "practice"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Competence(_Model):
    code: str
    name: str | None = None
    group: str | None = Field(
        default=None, description="Группа из заголовка списка (РПП 1.7): УК, УНК, УКБ, ОПК, ПК"
    )
    source: SourceRef


class Indicator(_Model):
    code: str
    competence_code: str | None = None
    text: str
    level: str | None = Field(default=None, description="Базовый / Средний / Продвинутый")
    source: SourceRef


class LearningOutcome(_Model):
    kind: OutcomeKind | None = None
    code: str | None = Field(
        default=None, description="Например, «ПКП-ППК-ОС3.1. З-1»; может не быть"
    )
    indicator_code: str | None = None
    competence_codes: list[str] = Field(
        default_factory=list,
        description="Компетенции, которым принадлежит результат; несколько — если ячейка "
        "результатов объединена по вертикали на группу компетенций",
    )
    text: str
    source: SourceRef

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_competence_code(cls, data: Any) -> Any:
        """Старый формат: одно `competence_code` вместо списка `competence_codes`."""
        if isinstance(data, dict) and "competence_code" in data:
            data = dict(data)
            legacy = data.pop("competence_code")
            if "competence_codes" not in data:
                data["competence_codes"] = [legacy] if legacy else []
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def competence_code(self) -> str | None:
        """Устарело: первая из `competence_codes`. Оставлено для обратной совместимости."""
        return self.competence_codes[0] if self.competence_codes else None


class AssessmentLink(_Model):
    """Строка таблицы «индикатор → контрольно-измерительные материалы»."""

    indicator_code: str | None = None
    competence_code: str | None = Field(
        default=None,
        description="Если строка КИМ относится к компетенции целиком, а не к индикатору",
    )
    indicator_text: str | None = None
    materials_raw: str
    materials: list[str] = Field(default_factory=list)
    source: SourceRef


class Attestation(_Model):
    """Одна промежуточная аттестация: вид и уточнения («экзамен», «устно, традиционная форма»)."""

    kind: str = Field(
        description="экзамен / зачёт / зачёт с оценкой (нормализовано: нижний регистр, ё)"
    )
    details: list[str] = Field(default_factory=list)


class ControlForm(_Model):
    semester: int
    trajectory: str | None = Field(
        default=None,
        description="Траектория как в документе: «ОСНОВНАЯ ТРАЕКТОРИЯ Профиль ТОП-ИТ ПРО»",
    )
    study_form: str | None = Field(default=None, description="очная / заочная / очно-заочная")
    current: list[str] = Field(default_factory=list, description="Формы текущего контроля")
    intermediate: list[Attestation] = Field(default_factory=list)
    current_raw: str | None = Field(default=None, description="Ячейка как в документе")
    intermediate_raw: str | None = Field(default=None, description="Ячейка как в документе")
    source: SourceRef


class Prerequisite(_Model):
    """Что нужно знать до курса (раздел 1.2): дисциплина или область знаний."""

    name: str = Field(description="Как в тексте документа, без кавычек")
    kind: Literal["дисциплина", "область знаний"]
    source: SourceRef


class ControlMention(_Model):
    """Форма контроля или оценочное средство, упомянутое в свободном тексте."""

    kind: str = Field(description="Нормализованное название: «контрольная работа», «экзамен»")
    role: Literal["текущий", "промежуточный"]
    count: int | None = Field(default=None, description="«2 контрольные работы» → 2")
    occurrences: int = Field(default=1, description="Сколько раз встречается в разделе")
    source: SourceRef


class Workload(_Model):
    """Трудоёмкость из текста: общая (`semester=None`) или по семестру."""

    semester: int | None = None
    credits: float | None = Field(default=None, description="Зачётные единицы")
    hours: int | None = None
    source: SourceRef


class WorkloadRow(_Model):
    """Строка таблицы «Трудоёмкость, объёмы учебной работы»: один семестр или ИТОГО."""

    trajectory: str | None = None
    study_form: str | None = Field(default=None, description="очная / заочная / очно-заочная")
    semester: int | None = Field(default=None, description="None у строки «ИТОГО»")
    is_total: bool = False
    contact_hours: dict[str, float] = Field(
        default_factory=dict,
        description="Контактная работа по видам: «лекции», «практические занятия», ...",
    )
    self_study_hours: dict[str, float] = Field(
        default_factory=dict, description="Самостоятельная работа по видам"
    )
    active_hours: float | None = Field(
        default=None, description="Объём активных и интерактивных форм"
    )
    credits: float | None = Field(default=None, description="Трудоёмкость, з.е.")
    source: SourceRef

    @model_validator(mode="before")
    @classmethod
    def _drop_computed(cls, data: Any) -> Any:
        """`total_hours` попадает в JSON как вычисляемое поле; при чтении пересчитываем."""
        if isinstance(data, dict) and "total_hours" in data:
            data = {k: v for k, v in data.items() if k != "total_hours"}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_hours(self) -> float:
        """Контактная + самостоятельная работа, часы."""
        return sum(self.contact_hours.values()) + sum(self.self_study_hours.values())


class ActiveForm(_Model):
    """Активная или интерактивная форма занятий из раздела 1.4 («… в активной форме – 60 часов»)."""

    text: str
    hours: float | None = Field(default=None, description="None, если часы не названы")
    is_total: bool = Field(
        default=False, description="Общий объём активных форм («составляет 40 часов: …»)"
    )
    source: SourceRef


class TopicHours(_Model):
    """Строка таблицы 2.2: тема, вид занятий и часы; для сверки с таблицей 2.1."""

    semester: int | None = None
    number: str | None = None
    topic: str
    kind: str = Field(description="Вид занятий как в документе: «лекции», «практические занятия»")
    hours: float
    source: SourceRef


class GradeScaleRow(_Model):
    """Строка шкалы оценивания 3.1.3: оценка СПбГУ, оценка ECTS и критерий или баллы."""

    attestation: str = Field(description="зачёт / экзамен / как в заголовке столбца")
    grade: str
    ects: str | None = Field(default=None, description="A–F")
    criteria: str
    source: SourceRef


class TicketPart(_Model):
    count: int
    item: str = Field(description="Как в тексте: «теоретических вопроса», «задачу»")


class Ticket(_Model):
    """Структура билета: «Билет содержит два теоретических вопроса и одну задачу»."""

    text: str
    parts: list[TicketPart]
    source: SourceRef


class ControlMaterial(_Model):
    """Сами контрольно-измерительные материалы: перечень вопросов или примеры задач."""

    kind: Literal["вопросы к экзамену", "вопросы к зачёту", "примеры задач", "контрольная работа"]
    title: str
    items: list[str]
    source: SourceRef


class Developer(_Model):
    """Разработчик программы (раздел 4)."""

    name: str
    degree: str | None = None
    title: str | None = None
    position: str | None = None
    contacts: str | None = None
    source: SourceRef


class ProgramInfo(_Model):
    """Направление и образовательная программа, если они названы в тексте документа."""

    direction_code: str | None = None
    direction: str | None = None
    program: str | None = Field(default=None, description="«ООП:» / «Образовательная программа:»")
    profiles: list[str] = Field(
        default_factory=list, description="Профили из строк траекторий таблиц: «Профиль ТОП-ИТ ПРО»"
    )
    source: SourceRef | None = None


class PracticeInfo(_Model):
    """РПП: вид, тип и способ проведения практики (разделы 1.2, 1.2.1, 1.3)."""

    kind: str | None = Field(default=None, description="Учебная / Производственная / …")
    type: str | None = None
    method: str | None = Field(default=None, description="стационарная / выездная")
    source: SourceRef


class ProfStandard(_Model):
    """Профессиональный стандарт и его трудовые функции (РПП 1.6)."""

    code: str = Field(description="«06.001»")
    name: str
    functions: list[str] = Field(default_factory=list, description="«A.3 Разработка и отладка…»")
    source: SourceRef


class StandardLink(_Model):
    """РПП 1.8: компетенции → обобщённые трудовые функции по профстандартам."""

    competence_codes: list[str]
    functions: dict[str, list[str]] = Field(description="«06.003» → [«A.6», «B.7»]")
    source: SourceRef


class RPDExtraction(_Model):
    file: str
    doc_kind: DocKind | None = None
    title: str | None = None
    title_en: str | None = None
    registration_number: str | None = None
    discipline_code: str | None = Field(
        default=None, description="Код дисциплины = рег. номер программы («001856» в плане)"
    )
    language: str | None = None
    credits: float | None = None
    competences: list[Competence] = Field(default_factory=list)
    indicators: list[Indicator] = Field(default_factory=list)
    learning_outcomes: list[LearningOutcome] = Field(default_factory=list)
    assessments: list[AssessmentLink] = Field(default_factory=list)
    control_forms: list[ControlForm] = Field(default_factory=list)
    prerequisites: list[Prerequisite] = Field(default_factory=list)
    control_mentions: list[ControlMention] = Field(default_factory=list)
    workload: list[Workload] = Field(default_factory=list)
    workload_table: list[WorkloadRow] = Field(
        default_factory=list, description="Таблица трудоёмкости по семестрам"
    )
    semesters: list[int] = Field(
        default_factory=list, description="Семестры, названные в тексте раздела 2.1"
    )
    goals: str | None = Field(default=None, description="Раздел 1.1 целиком")
    active_forms: list[ActiveForm] = Field(default_factory=list)
    topic_hours: list[TopicHours] = Field(default_factory=list)
    grade_scales: list[GradeScaleRow] = Field(default_factory=list)
    tickets: list[Ticket] = Field(default_factory=list)
    control_materials: list[ControlMaterial] = Field(default_factory=list)
    developers: list[Developer] = Field(default_factory=list)
    program: ProgramInfo | None = None
    practice: PracticeInfo | None = None
    prof_standards: list[ProfStandard] = Field(default_factory=list)
    standard_links: list[StandardLink] = Field(default_factory=list)
    llm: LLMExtraction | None = Field(
        default=None, description="Смысловые поля от LLM; None, если LLM не подключали"
    )
    warnings: list[str] = Field(default_factory=list)
