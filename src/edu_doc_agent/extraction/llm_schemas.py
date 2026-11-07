"""Схемы ответа LLM для смысловых полей РПД, которые правилами и Yargy берутся плохо.

Модели `*Item` — то, что возвращает модель (по ним строится JSON-схема для структурированного
вывода). Модели `LLM*` — то же самое плюс `source`, который добавляет код после проверки цитаты.
Во всех схемах есть `quote`: дословная цитата из текста, по ней код проверяет, что модель
ничего не выдумала.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from edu_doc_agent.extraction.source import SourceRef


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- пререквизиты (РПД 1.2) ----------------------------------------------------------------


class PrerequisiteItem(_Schema):
    name: str = Field(description="Название в именительном падеже: «аналитическая геометрия»")
    kind: Literal["дисциплина", "область знаний", "навык"]
    relation: Literal["предшествует", "параллельно"] = Field(
        description="«предшествует» — нужно знать до курса; «параллельно» — изучается одновременно"
    )
    quote: str = Field(description="Дословный фрагмент текста, из которого взято требование")


class PrerequisitesOutput(_Schema):
    items: list[PrerequisiteItem] = Field(
        description="Пусто, если требуется только общее образование или ничего"
    )


# --- активные и интерактивные формы (РПД 1.4) ----------------------------------------------


class ActiveFormItem(_Schema):
    kind: str = Field(description="Вид занятий или метод: «практические занятия», «семинары»")
    hours: float | None = Field(description="Часы, если названы; иначе null")
    mandatory: bool = Field(
        description="true — формулировка обязывающая («необходимо проводить», «составляют»); "
        "false — допускающая («могут применяться»)"
    )
    quote: str


class ActiveFormsOutput(_Schema):
    total_hours: float | None = Field(
        description="Общий объём, только если назван в тексте явно («составляет 32 часа»); "
        "сам не считай"
    )
    items: list[ActiveFormItem]


# --- правила оценивания (РПД 3.1.3) --------------------------------------------------------


class GradingRuleItem(_Schema):
    condition: str = Field(description="При каком условии, своими словами и кратко")
    effect: str = Field(description="Что из этого следует для оценки или баллов")
    quote: str


class GradingRulesOutput(_Schema):
    rules: list[GradingRuleItem] = Field(
        description="Только правила, влияющие на оценку; организационные сведения не включай"
    )


# --- результат с источником ----------------------------------------------------------------


class LLMPrerequisite(PrerequisiteItem):
    source: SourceRef


class LLMActiveForm(ActiveFormItem):
    source: SourceRef


class LLMGradingRule(GradingRuleItem):
    source: SourceRef


class LLMExtraction(_Schema):
    file: str
    prerequisites: list[LLMPrerequisite] = Field(default_factory=list)
    active_forms: list[LLMActiveForm] = Field(default_factory=list)
    active_total_hours: float | None = None
    grading_rules: list[LLMGradingRule] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list, description="Невалидные ответы, цитаты не из текста, сбои вызова"
    )
