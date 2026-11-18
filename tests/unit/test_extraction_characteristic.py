"""Извлечение сущностей из общей характеристики ОП: синтетический Markdown и реальные файлы."""

# ruff: noqa: E501  (строки таблиц и длинные формулировки)

import pytest

from edu_doc_agent.extraction.characteristic import (
    extract_characteristic,
    function_key,
)
from edu_doc_agent.extraction.crossdoc import check_program_consistency
from edu_doc_agent.extraction.curriculum import extract_plan
from tests.real_data import real_characteristic_markdown, real_plan_markdown

MD = """Приложение

### Санкт-Петербургский государственный университет

### ОБЩАЯ ХАРАКТЕРИСТИКА

Код, наименование укрупненной группы специальностей и направлений

### 02.00.00 Компьютерные и информационные науки

Код, наименование направления подготовки/ специальности

### 02.03.02 Фундаментальная информатика и информационные технологии

Уровень бакалавриат

| Направленность образовательной программы /профиль | Направленность образовательной программы /профиль (англ.) |
| --- | --- |
| Программирование | Programming |

Шифр программы СВ.5003.2026

| Наименование программы | Наименование программы (англ.) |
| --- | --- |
| Программирование и ИТ | Programming and IT |

Форма(ы) обучения: очная

Язык(и) обучения: русский, английский

Срок(и) обучения: 4 года

### Аннотация

Программа готовит инженеров.

### Миссия образовательной программы (стратегия развития)

Подготовка кадров.

# 1. Характеристика профессиональной деятельности

## 1.1. Квалификация, присваиваемая выпускникам:

Бакалавр

## 1.2. Области (сферы) профессиональной деятельности выпускников

06 Связь, информационные и коммуникационные технологии;

40 Сквозные виды деятельности.

Выпускники могут работать и в других областях.

## 1.3. Объекты профессиональной деятельности выпускников

- модели сложных систем;

- алгоритмы и библиотеки.

## 1.4. Виды профессиональной деятельности выпускников

Научно-исследовательская деятельность:

Код ОКВЭД 62.09 – Деятельность, связанная с вычислительной техникой, прочая;

Код ОКВЭД 72.1 – Научные исследования и разработки.

Производственная деятельность:

Код ОКВЭД 62.01 – Разработка программного обеспечения.

## 1.5. Задачи профессиональной деятельности выпускников

Научно-исследовательская деятельность:

- участие в научных проектах;

- подготовка публикаций.

Производственная деятельность:

- разработка алгоритмов.

Для профиля «ТОП» приоритетны задачи ИИ.

## 1.6. Перечень применяемых профессиональных стандартов

Код 06.001 Профессиональный стандарт «Программист» (приказ Минтруда от 20.07.2022 № 424н, регистрационный № 69720).

Код 40.011 Профессиональный стандарт «Специалист по НИОКР» (Приказ от 04.03.2014 № 121н).

## 1.7. Сведения о работодателях

ООО «Яндекс»;

АНО «Астра Академия».

# 2. Планируемые результаты

## 2.1. Перечень универсальных компетенций, предусмотренных ФГОС3++ – бакалавриат:

УК-1 Способен искать информацию.

УК-2 Способен определять круг задач.

## 2.2. Перечень общепрофессиональных компетенций, предусмотренных ФГОС 3++ – бакалавриат:

ОПК-1 Способен применять знания.

## 2.3. Перечень универсальных надпрофессиональных компетенций, предусмотренных моделью:

УНК-01 Способен применять коммуникационные компетенции.

Указанные компетенции реализуются в каждом семестре.

## 2.4. Перечень профессиональных компетенций, формирующих практическую составляющую результатов освоения

ПКП-ППК-Р1 Способен разрабатывать код.

# 3. Сопоставление компетенций с профессиональными стандартами

| Перечень компетенций | Обобщенные трудовые функции, трудовые функции в соответствии с профессиональным стандартом |
| --- | --- |
| ПКП-ППК-Р1 | 06.001.A.3 Разработка и отладка кода.<br>06.015.А.4 Написание кода ИС (кириллическая А) |
| ПКП-ППК-Х9 | 06.042.B/04.7 Разработка требований. |
| ОПК-5 | 99.999.A.1 Чужая функция |

# 4. Требования к поступающим

В соответствии с Порядком приема.
"""


def _doc():
    return extract_characteristic(MD, "x.md")


def test_title_block() -> None:
    d = _doc()
    assert (d.group_code, d.group_name) == ("02.00.00", "Компьютерные и информационные науки")
    assert (d.direction_code, d.direction) == (
        "02.03.02",
        "Фундаментальная информатика и информационные технологии",
    )
    assert (d.level, d.program_code) == ("бакалавриат", "СВ.5003.2026")
    assert (d.profile, d.profile_en) == ("Программирование", "Programming")
    assert (d.program_name, d.program_name_en) == ("Программирование и ИТ", "Programming and IT")
    assert (d.study_form, d.languages, d.duration) == ("очная", "русский, английский", "4 года")
    assert d.annotation == "Программа готовит инженеров." and d.mission == "Подготовка кадров."


def test_lists_of_part_one() -> None:
    d = _doc()
    assert d.qualification == ["Бакалавр"]
    assert [(a.code, a.name) for a in d.areas] == [
        ("06", "Связь, информационные и коммуникационные технологии"),
        ("40", "Сквозные виды деятельности"),
    ]  # абзац без кода области в список не попадает
    assert [o.text.rstrip(";.") for o in d.objects] == [
        "модели сложных систем",
        "алгоритмы и библиотеки",
    ]
    assert [(s.code, s.name) for s in d.standards] == [
        ("06.001", "Программист"),
        ("40.011", "Специалист по НИОКР"),
    ]
    assert d.standards[0].details == "приказ Минтруда от 20.07.2022 № 424н, регистрационный № 69720"
    assert [e.text for e in d.employers] == ["ООО «Яндекс»", "АНО «Астра Академия»"]


def test_activities_join_okved_and_tasks_by_kind() -> None:
    d = _doc()
    nir, prod = d.activities
    assert nir.name == "Научно-исследовательская деятельность"
    assert [(o.code, o.name) for o in nir.okved] == [
        ("62.09", "Деятельность, связанная с вычислительной техникой, прочая"),
        ("72.1", "Научные исследования и разработки"),
    ]
    assert [t.text for t in nir.tasks] == ["участие в научных проектах;", "подготовка публикаций."]
    assert [o.code for o in prod.okved] == ["62.01"] and len(prod.tasks) == 1


def test_competence_groups_by_heading_text_not_number() -> None:
    d = _doc()
    got = [(c.code, c.group, c.section) for c in d.competences]
    assert got == [
        ("УК-1", "универсальные (ФГОС)", "2.1"),
        ("УК-2", "универсальные (ФГОС)", "2.1"),
        ("ОПК-1", "общепрофессиональные", "2.2"),  # заголовок ОПК тоже «предусмотренных ФГОС»
        ("УНК-01", "универсальные надпрофессиональные", "2.3"),
        ("ПКП-ППК-Р1", "профессиональные практические", "2.4"),
    ]
    assert d.competences[0].name == "Способен искать информацию"  # без точки в конце


def test_mapping_function_code_formats() -> None:
    d = _doc()
    r1, x9, opk = d.mappings
    assert [(f.code, f.standard_code, f.function_letter, f.level) for f in r1.functions] == [
        ("06.001.A.3", "06.001", "A", 3),
        ("06.015.А.4", "06.015", "A", 4),  # код хранится как в тексте (кириллическая А)
    ]
    assert r1.functions[1].name == "Написание кода ИС (кириллическая А)"
    assert [(f.code, f.function_letter, f.level) for f in x9.functions] == [
        ("06.042.B/04.7", "B", 7)
    ]
    assert function_key("06.015.А.4") == function_key("06.015.A.4")
    assert opk.functions[0].standard_code == "99.999"


def test_sections_four_and_up_are_kept_as_text() -> None:
    d = _doc()
    assert [(s.number, s.text) for s in d.sections] == [("4", "В соответствии с Порядком приема.")]


def test_warnings_for_inconsistencies() -> None:
    text = "\n".join(_doc().warnings)
    assert "ПКП-ППК-Х9 нет в разделе 2" in text and "ОПК-5 нет в разделе 2" in text
    assert "профстандарта 06.015 нет в разделе 1.6" in text
    assert "профстандарта 06.042 нет в разделе 1.6" in text
    assert "профстандарта 99.999 нет в разделе 1.6" in text
    assert "ПКП-ППК-Р1" not in text  # эта компетенция есть и в разделе 2, и в таблице 3


def test_empty_document_is_reported() -> None:
    text = "\n".join(extract_characteristic("# текст", "x.md").warnings)
    assert "направление" in text and "раздел 2" in text and "раздел 3" in text


def test_program_consistency_with_plan() -> None:
    plan = extract_plan(
        "Программирование и ИТ Programming and IT\n\n| по уровню | бакалавриат |  |\n| --- | --- | --- |\n"
        "| по направлению (специальности) | 02.03.02 Фундаментальная информатика и информационные технологии |  |\n"
        "| Форма обучения: |  | заочная |\n| Срок обучения по основной образовательной программе |  | 4 года |\n",
        "plan.md",
    )
    warnings = check_program_consistency(plan, _doc())
    assert len(warnings) == 1 and "форма обучения" in warnings[0] and "заочная" in warnings[0]


needs_data = pytest.mark.skipif(
    real_characteristic_markdown() is None, reason="нет data/Характеристика.zip"
)


@needs_data
@pytest.mark.parametrize("corrected", [True, False], ids=["corr", "original"])
def test_real_documents_have_no_warnings_and_group_by_heading(corrected: bool) -> None:
    d = extract_characteristic(real_characteristic_markdown(corrected) or "", "char.docx")
    assert d.warnings == []
    groups = {c.group for c in d.competences}
    assert "общепрофессиональные" in groups and len(d.competences) == 49
    assert sum(c.group == "универсальные (ФГОС)" for c in d.competences) == 10
    # в исходной версии УНК вынесены в отдельный раздел, в исправленной входят в практические
    unk = sum(c.group == "универсальные надпрофессиональные" for c in d.competences)
    assert unk == (0 if corrected else 3)


@needs_data
def test_real_characteristic_agrees_with_plan() -> None:
    if real_plan_markdown() is None:
        pytest.skip("нет data/Учебный план.zip")
    plan = extract_plan(real_plan_markdown() or "", "plan.docx")
    char = extract_characteristic(real_characteristic_markdown() or "", "char.docx")
    assert check_program_consistency(plan, char) == []
