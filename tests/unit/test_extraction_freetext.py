"""Тесты извлечения из свободного текста (Yargy): пререквизиты, контроль, трудоёмкость, семестры."""

import pytest

from edu_doc_agent.extraction import RPDExtraction, extract_rpd, extract_rpd_file
from tests.real_data import find_md

MATH = find_md("Математический анализ I")
PRACTICE = find_md("С2-5_ТОП__Учебная практика")


def _extract(*sections: tuple[str, str], title: str = "Тестовая дисциплина") -> RPDExtraction:
    """Собирает md из пар (заголовок, текст) и прогоняет извлечение."""
    parts = [
        "### Р А Б О Ч А Я П Р О Г Р А М М А",
        "",
        "### учебной дисциплины",
        "",
        title,
        "",
        "### Язык(и) обучения",
        "",
    ]
    for heading, body in sections:
        parts += [f"## {heading}", "", body, ""]
    return extract_rpd("\n".join(parts), file="synthetic.md")


def _prereqs(body: str, title: str = "Тестовая дисциплина") -> list[tuple[str, str]]:
    result = _extract(("1.2. Пререквизиты", body), title=title)
    return [(p.kind, p.name) for p in result.prerequisites]


def _control(body: str, number: str = "3.1.3. Методика контроля"):
    return {m.kind: m for m in _extract((number, body)).control_mentions}


# --- пререквизиты --------------------------------------------------------------------------


def test_prerequisite_disciplines_in_quotes() -> None:
    body = (
        "Обучающийся должен иметь знания по дисциплинам: «Алгебра и геометрия», "
        "«Основы дискретной математики», которые создают основы для курса."
    )
    assert _prereqs(body) == [
        ("дисциплина", "Алгебра и геометрия"),
        ("дисциплина", "Основы дискретной математики"),
    ]


def test_quoted_name_with_conjunction_is_not_split() -> None:
    assert _prereqs("Рекомендуются «Теория вероятностей и математическая статистика».") == [
        ("дисциплина", "Теория вероятностей и математическая статистика")
    ]


def test_prerequisite_list_after_colon_without_quotes() -> None:
    body = "Освоение дисциплин: основы программирования, дискретная математика."
    assert _prereqs(body) == [
        ("дисциплина", "основы программирования"),
        ("дисциплина", "дискретная математика"),
    ]


def test_prerequisite_disciplines_after_preposition_in_any_case() -> None:
    body = "Нужны знания по курсам линейной алгебры, математического анализа и геометрии."
    assert _prereqs(body) == [
        ("дисциплина", "линейной алгебры"),
        ("дисциплина", "математического анализа"),
        ("дисциплина", "геометрии"),
    ]


def test_prerequisite_areas_of_knowledge() -> None:
    body = "Необходимы базовые знания по аналитической геометрии и математическому анализу."
    assert _prereqs(body) == [
        ("область знаний", "аналитической геометрии"),
        ("область знаний", "математическому анализу"),
    ]


def test_current_discipline_title_is_not_a_prerequisite() -> None:
    body = "Знания по дисциплинам «Алгебра» нужны для учебной дисциплины «Матан I»."
    assert _prereqs(body, title="Матан I") == [("дисциплина", "Алгебра")]


def test_prerequisites_only_from_section_1_2() -> None:
    result = _extract(("1.1. Цели", "Курс опирается на «Алгебра и геометрия»."))
    assert result.prerequisites == []


def test_prerequisite_has_source_line() -> None:
    result = _extract(("1.2. Пререквизиты", "Нужна «Алгебра»."))
    ref = result.prerequisites[0].source
    assert ref.section == "1.2. Пререквизиты"
    assert ref.line == 11


# --- формы контроля и оценочные средства ---------------------------------------------------


def test_control_terms_in_any_case_are_normalized() -> None:
    found = _control(
        "Результаты контрольных работ и сдача коллоквиума. "
        "Задания контрольной работе выдаёт преподаватель."
    )
    assert {"контрольная работа", "коллоквиум"} <= set(found)
    assert found["контрольная работа"].occurrences == 2
    assert found["контрольная работа"].role == "текущий"


def test_control_counts_are_taken_from_number_before_term() -> None:
    found = _control("В семестре предусмотрено 2 контрольные работы и 1 коллоквиум.")
    assert found["контрольная работа"].count == 2
    assert found["коллоквиум"].count == 1


def test_intermediate_forms_have_intermediate_role() -> None:
    found = _control(
        "Промежуточная аттестация проводится в форме экзамена, затем зачёта с оценкой."
    )
    assert found["экзамен"].role == "промежуточный"
    assert found["зачёт с оценкой"].role == "промежуточный"
    assert "зачёт" not in found  # «зачёт с оценкой» — одна форма, а не две


def test_grade_credited_is_not_a_credit_form() -> None:
    assert "зачёт" not in _control("Выставляется оценка «зачтено», если работа сдана.")


def test_checkboxes_keep_only_marked_forms() -> None:
    found = _control("Промежуточная аттестация 🗹 зачет □ экзамен")
    assert set(found) == {"зачёт"}


def test_listed_activities_in_practice_are_kept_whole() -> None:
    found = _control(
        "Виды: демонстрация результатов по проекту, ретроспективный анализ достижений и проблем",
        number="3.1.1. Текущий контроль успеваемости",
    )
    assert set(found) == {
        "демонстрация результатов по проекту",
        "ретроспективный анализ достижений и проблем",
    }


def test_control_is_ignored_outside_control_sections() -> None:
    assert _control("Преподаватель может использовать презентации и тест.", "3.1.2. Методика") == {}
    assert _control("Пример: тест из 12 вопросов.", "3.1.4. Примеры КИМ") == {}


# --- трудоёмкость и семестры ---------------------------------------------------------------

LOAD_TEXT = """Практика проводится в течение 2-го, 3-го, 4-го и 5-го семестров.

Общая трудоемкость практики составляет 18 зачетных единиц (648 академических часов).

- Семестр 2: 3 з.е. (108 часов)
- Семестр 4: 1,5 з.е. (54 часа)"""


def test_total_and_per_semester_workload() -> None:
    result = _extract(("2.1. Организация учебных занятий", LOAD_TEXT))
    got = [(w.semester, w.credits, w.hours) for w in result.workload]
    assert got == [(None, 18.0, 648), (2, 3.0, 108), (4, 1.5, 54)]


def test_semesters_from_ordinal_list_and_workload() -> None:
    result = _extract(("2.1. Организация", LOAD_TEXT))
    assert result.semesters == [2, 3, 4, 5]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Изучается во 2-3 семестрах.", [2, 3]),
        ("Занятия идут с 3 по 5 семестр.", [3, 4, 5]),
        ("Проводится в 3 семестре.", [3]),
        ("Курс читается в 1-м и 2-м семестрах.", [1, 2]),
    ],
)
def test_semester_phrasings(text: str, expected: list[int]) -> None:
    assert _extract(("2.1. Организация", text)).semesters == expected


def test_hours_without_credits_are_not_workload() -> None:
    result = _extract(("2.1. Организация", "Занятия в интерактивной форме – 60 часов."))
    assert result.workload == []


def test_workload_only_from_section_2_1() -> None:
    assert _extract(("1.1. Цели", "Курс занимает 3 з.е. (108 часов).")).workload == []


# --- реальные документы --------------------------------------------------------------------


@pytest.mark.skipif(MATH is None, reason="нет md «Математический анализ I»")
def test_real_math_analysis_free_text() -> None:
    assert MATH is not None
    result = extract_rpd_file(MATH)
    assert [p.name for p in result.prerequisites] == [
        "Алгебра и геометрия",
        "Основы дискретной математики",
    ]
    controls = {m.kind: m for m in result.control_mentions}
    assert controls["контрольная работа"].count == 2
    assert controls["коллоквиум"].count == 1
    assert controls["зачёт"].role == "промежуточный"


@pytest.mark.skipif(PRACTICE is None, reason="нет md учебной практики")
def test_real_practice_free_text() -> None:
    assert PRACTICE is not None
    result = extract_rpd_file(PRACTICE)
    got = [(w.semester, w.credits, w.hours) for w in result.workload]
    assert got == [(None, 18.0, 648), (2, 3.0, 108), (3, 3.0, 108), (4, 6.0, 216), (5, 6.0, 216)]
    assert result.semesters == [2, 3, 4, 5]
    kinds = {m.kind for m in result.control_mentions}
    assert "зачёт" in kinds
    assert "экзамен" not in kinds  # в документе отмечен только зачёт
