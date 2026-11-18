"""LLM-извлечение сущностей из свободного текста общей характеристики ОП (ADR-0002, часть 2).

Правила берут таблицы и перечни характеристики, но не повествовательные разделы. Здесь по
одному вызову на раздел выделяются: из «Аннотации» — профессиональные роли, целевые должности
выпускников, траектории, проекты и модели, нормативные акты; из «Миссии» — принципы стратегии;
из раздела 5 — профили и средства индивидуализации. Каркас тот же, что у РПД (`llm_extract`):
структурированный вывод, повторы при невалидном JSON, дословная цитата-доказательство, которую
код проверяет по тексту раздела; вызовы только через `llm/`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from edu_doc_agent.extraction.llm_extract import (
    LLMTask,
    _locate,
    run_task,
    section_text,
)
from edu_doc_agent.extraction.md_utils import MdSection, split_sections
from edu_doc_agent.extraction.models import _Model
from edu_doc_agent.extraction.source import SourceRef, attach_quotes
from edu_doc_agent.llm import LLMClient, LLMError

# --- схемы ответа модели ---------------------------------------------------------------------


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleItem(_Schema):
    name: str = Field(description="Название роли как в тексте: «Аналитик данных»")
    kind: Literal["основная", "опережающая", "дополнительная"] = Field(
        description="«основная» — главная роль программы; «опережающая» — роль, к которой "
        "готовят с опережением; иначе «дополнительная»"
    )
    quote: str = Field(description="Дословный фрагмент текста с ролью")


class NamedItem(_Schema):
    name: str
    quote: str = Field(description="Дословный фрагмент текста")


class ProjectItem(_Schema):
    name: str = Field(
        description="Название без слов «федеральный проект», кавычек: «Цифровые кадры»"
    )
    kind: Literal["федеральный проект", "национальный проект", "модель", "программа", "другое"]
    quote: str


class AnnotationOutput(_Schema):
    roles: list[RoleItem]
    positions: list[NamedItem] = Field(
        description="Должности и уровни специалистов, которых достигают выпускники"
    )
    tracks: list[NamedItem] = Field(
        description="Траектории и направления специализации внутри роли"
    )
    projects: list[ProjectItem]
    legal_acts: list[NamedItem] = Field(description="Нормативные акты: указ, приказ, закон")


class PrincipleItem(_Schema):
    text: str = Field(description="Принцип кратко, как в тексте")
    quote: str


class MissionOutput(_Schema):
    principles: list[PrincipleItem]


class IndividualizationOutput(_Schema):
    profiles: list[NamedItem] = Field(description="Профили (потоки) обучения программы")
    means: list[NamedItem] = Field(
        description="Средства индивидуализации: выбор элективов, онлайн-курсы и т. п."
    )


# --- результат с источником ------------------------------------------------------------------


class LLMEntity(_Model):
    name: str
    kind: str | None = None
    quote: str
    source: SourceRef


class CharacteristicLLM(_Model):
    roles: list[LLMEntity] = Field(default_factory=list)
    positions: list[LLMEntity] = Field(default_factory=list)
    tracks: list[LLMEntity] = Field(default_factory=list)
    projects: list[LLMEntity] = Field(default_factory=list)
    legal_acts: list[LLMEntity] = Field(default_factory=list)
    principles: list[LLMEntity] = Field(default_factory=list)
    profiles: list[LLMEntity] = Field(default_factory=list)
    individualization_means: list[LLMEntity] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --- промпты ---------------------------------------------------------------------------------
# Примеры синтетические и с эталоном `characteristic_llm_gold.json` не пересекаются.

SYSTEM = """Ты извлекаешь именованные сущности из текста общей характеристики образовательной
программы вуза.
Правила:
1. Используй только переданный текст. Ничего не выдумывай: нет в тексте — пустой список.
2. В поле quote — один сплошной дословный фрагмент текста (подстрока без изменений и без
   многоточий), по которому видно, откуда взят элемент. Не пересказывай.
   name можно привести к именительному падежу, но quote сохраняет исходные окончания:
   «посредством выбора факультативов» → name «выбор факультативов», quote «выбора факультативов».
3. Ответь только JSON строго по схеме, без пояснений."""

ANNOTATION = """Раздел «Аннотация» образовательной программы. Выдели:
- roles: профессиональные роли, на которые ориентирована программа («основная роль», «опережающая
  роль»); название роли — как в тексте, без слов «роль» и кавычек. Должность вообще и уровень
  квалификации ролями не считаются.
- positions: должности и уровни специалистов, которых достигают выпускники («уровня Junior»,
  «на позиции руководителей лабораторий»); каждый вариант — отдельный элемент.
  В сочинённом перечне восстанавливай общее главное слово в каждом названии:
  «директоров школ и колледжей» → «директор школы», «директор колледжа».
  Не выдавай «колледжей» как должность. Цитата для обоих элементов должна включать
  общее главное слово и соответствующую часть перечня дословно, без сокращений.
- tracks: траектории и направления специализации внутри роли; перечень через запятую — по
  элементу на пункт; «и др.» и «и т. д.» не элементы.
- projects: все названные проекты, программы и модели, в рамках которых реализуется программа
  или на основе которых она построена. Модель, положенная в основу программы, тоже отдельный
  элемент, даже если у неё нет названия в кавычках. Не пропускай её после перечня проектов;
  название — без родовых слов («федеральный проект «Цифровые кадры»» → name «Цифровые кадры»).
  kind определяется родовыми словами в цитате: «федерального проекта» → «федеральный проект»,
  «национального проекта» → «национальный проект», «программы» → «программа», «модели» → «модель».
  Проекты не помечай как «программа» только потому, что они упомянуты в образовательной программе.
- legal_acts: нормативные акты со ссылкой на номер и дату.
Пример. Текст: «Программа готовит биоинформатиков по основной роли «Аналитик геномных данных» и
опережающей роли «Руководитель лаборатории». Траектории: протеомика, метагеномика и др.
Выпускники достигают уровня Middle и выходят на позиции руководителей лабораторий. Программа
реализуется в рамках национальной программы «Цифровая медицина» согласно приказу Минздрава от
01.02.2021 № 55.»
Ответ: {"roles": [{"name": "Аналитик геномных данных", "kind": "основная", "quote": "основной роли
«Аналитик геномных данных»"}, {"name": "Руководитель лаборатории", "kind": "опережающая", "quote":
"опережающей роли «Руководитель лаборатории»"}], "positions": [{"name": "Middle", "quote":
"достигают уровня Middle"}, {"name": "руководитель лаборатории", "quote": "на позиции
руководителей лабораторий"}], "tracks": [{"name": "протеомика", "quote": "протеомика"}, {"name":
"метагеномика", "quote": "метагеномика"}], "projects": [{"name": "Цифровая медицина", "kind":
"программа", "quote": "национальной программы «Цифровая медицина»"}], "legal_acts": [{"name":
"приказ Минздрава от 01.02.2021 № 55", "quote": "приказу Минздрава от 01.02.2021 № 55"}]}

Текст раздела:
{text}"""

MISSION = """Раздел «Миссия образовательной программы (стратегия развития)».
Выпиши принципы, на которых строится стратегия развития программы: каждый пункт перечня после
слов вроде «базируется на следующих принципах» — отдельный принцип. Саму формулировку миссии,
вводные и заключительные предложения принципами не считай. quote — пункт перечня целиком или его
начало одним фрагментом.
Пример. Текст: «Миссия — готовить кадры. Стратегия базируется на принципах: - открытость данных
исследований; - ежегодное обновление курсов с участием работодателей.» Ответ: {"principles":
[{"text": "открытость данных исследований", "quote": "открытость данных исследований"}, {"text":
"ежегодное обновление курсов с участием работодателей", "quote": "ежегодное обновление курсов с
участием работодателей"}]}

Текст раздела:
{text}"""

INDIVIDUALIZATION = """Раздел «Способы и варианты индивидуализации обучения».
- profiles: профили (потоки) обучения, по которым идёт программа; название как в тексте
  («базовый профиль», «профиль «Х»»).
- means: средства индивидуализации траектории — выбор элективных дисциплин, онлайн-курсы и т. п.
  (отдельный элемент на каждое средство). Тестирование для распределения по профилям и ссылки на
  порядок приёма средствами индивидуализации не считай.
Пример. Текст: «Обучение идёт по профилям: исследовательский и прикладной. Траектория
выстраивается через выбор факультативов и включение дистанционных модулей.»
Ответ: {"profiles": [{"name": "исследовательский", "quote": "исследовательский"}, {"name":
"прикладной", "quote": "прикладной"}], "means": [{"name": "выбор факультативов", "quote": "выбор
факультативов"}, {"name": "дистанционные модули", "quote": "включение дистанционных модулей"}]}

Текст раздела:
{text}"""

# --- задачи ----------------------------------------------------------------------------------


def _by_heading(*starts: str) -> Callable[[list[MdSection]], list[MdSection]]:
    def select(sections: list[MdSection]) -> list[MdSection]:
        return [s for s in sections if s.heading.lower().startswith(starts)]

    return select


def select_individualization(sections: list[MdSection]) -> list[MdSection]:
    """Раздел 5 вместе с подразделами до следующего заголовка того же уровня."""
    picked: list[MdSection] = []
    level: int | None = None
    for section in sections:
        if section.number == "5":
            level = section.level
        elif level is not None and section.level <= level:
            break
        if level is not None:
            picked.append(section)
    return picked


TASKS = [
    LLMTask("annotation", ANNOTATION, AnnotationOutput, _by_heading("аннотация")),
    LLMTask("mission", MISSION, MissionOutput, _by_heading("миссия")),
    LLMTask(
        "individualization", INDIVIDUALIZATION, IndividualizationOutput, select_individualization
    ),
]


@dataclass
class _Sink:
    result: CharacteristicLLM
    sections: list[MdSection]
    file: str
    text: str

    def add(self, target: list[LLMEntity], items: list, name: str, kind: bool = False) -> None:
        for item in items:
            # Проверяем именно дословную цитату; сокращения с многоточиями не принимаем.
            if not item.quote.strip() or item.quote not in self.text:
                self.result.warnings.append(f"{name}: цитата не найдена в тексте раздела")
                continue
            label = item.text if isinstance(item, PrincipleItem) else item.name
            if not label.strip():
                self.result.warnings.append(f"{name}: пустое название сущности")
                continue
            target.append(
                LLMEntity(
                    name=label,
                    kind=getattr(item, "kind", None) if kind else None,
                    quote=item.quote,
                    source=_locate(item.quote, self.sections, self.file, False),
                )
            )


def _dedupe(items: list[LLMEntity]) -> list[LLMEntity]:
    seen, kept = set(), []
    for e in items:
        if (key := (" ".join(e.name.casefold().split()), e.kind)) not in seen:
            seen.add(key)
            kept.append(e)
    return kept


def extract_characteristic_llm(
    md: str, file: str, client: LLMClient, tasks: list[str] | None = None
) -> CharacteristicLLM:
    """Один вызов на задачу; неподтверждённые цитаты отбрасываются с предупреждением."""
    result = CharacteristicLLM()
    unknown = set(tasks or ()) - {t.name for t in TASKS}
    if unknown:
        raise ValueError(f"неизвестные LLM-задачи: {', '.join(sorted(unknown))}")
    sections = split_sections(md)
    for task in (t for t in TASKS if tasks is None or t.name in tasks):
        picked = task.select(sections)
        text = section_text(picked, False) if picked else ""
        if not text:
            result.warnings.append(f"{task.name}: раздел не найден")
            continue
        try:
            output, problem = run_task(task, text, client, system=SYSTEM)
        except LLMError:
            result.warnings.append(f"{task.name}: сбой вызова LLM, задача пропущена")
            continue
        if output is None:
            result.warnings.append(str(problem))
            continue
        sink = _Sink(result, picked, file, text)
        if isinstance(output, AnnotationOutput):
            sink.add(result.roles, output.roles, "roles", kind=True)
            sink.add(result.positions, output.positions, "positions")
            sink.add(result.tracks, output.tracks, "tracks")
            sink.add(result.projects, output.projects, "projects", kind=True)
            sink.add(result.legal_acts, output.legal_acts, "legal_acts")
        elif isinstance(output, MissionOutput):
            sink.add(result.principles, output.principles, "principles")
        elif isinstance(output, IndividualizationOutput):
            sink.add(result.profiles, output.profiles, "profiles")
            sink.add(result.individualization_means, output.means, "means")
    for name in CharacteristicLLM.model_fields:
        if name == "warnings":
            continue
        setattr(result, name, _dedupe(getattr(result, name)))
    attach_quotes(result, md)
    return result
