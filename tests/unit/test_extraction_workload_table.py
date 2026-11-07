"""Тесты разбора таблицы «Трудоёмкость, объёмы учебной работы»."""

import pytest

from edu_doc_agent.extraction import RPDExtraction, extract_rpd, extract_rpd_file
from tests.real_data import find_md


def _row(*cells: str, width: int = 12) -> str:
    padded = [*cells, *[""] * (width - len(cells))]
    return "| " + " | ".join(padded) + " |"


def _table(
    group_row: list[str], leaf_row: list[str], rows: list[list[str]], width: int = 12
) -> str:
    sep = "| " + " | ".join(["---"] * width) + " |"
    lines = [
        _row("Трудоёмкость, объёмы учебной работы", width=width),
        sep,
        _row(*group_row, width=width),
        _row(*leaf_row, width=width),
    ]
    lines += [_row(*r, width=width) for r in rows]
    return "\n".join(lines)


def _extract(table: str) -> RPDExtraction:
    return extract_rpd(f"## 2.1. Организация\n\n{table}\n", file="synthetic.md")


GROUP = [
    "Период",
    "Контактная работа обучающихся",
    "",
    "",
    "",
    "Самостоятельная работа",
    "",
    "",
    "Объём активных",
    "Трудоёмкость",
]
LEAF = [
    "Период",
    "лекции",
    "консультации",
    "практические<br>занятия",
    "промежуточная<br>аттестация",
    "под руководством преподавателя",
    "сам.раб. с использованием методических материалов",
    "промежуточная аттестация (сам.раб.)",
    "Объём активных и интерактивных форм",
    "Трудоёмкость",
]


def test_columns_are_recognized_by_header_meaning() -> None:
    rows = [
        ["ОСНОВНАЯ ТРАЕКТОРИЯ"],
        ["очная форма обучения"],
        ["Семестр 1", "34", "2", "28", "4", "", "45", "31", "30", "4"],
        ["ИТОГО", "34", "2", "28", "4", "", "45", "31", "30", "4"],
    ]
    result = _extract(_table(GROUP, LEAF, rows))
    sem, total = result.workload_table
    assert (sem.semester, sem.is_total, total.is_total) == (1, False, True)
    assert sem.contact_hours == {
        "лекции": 34,
        "консультации": 2,
        "практические занятия": 28,
        "промежуточная аттестация": 4,
    }
    assert sem.self_study_hours == {
        "с использованием методических материалов": 45,
        "промежуточная аттестация": 31,
    }
    assert (sem.active_hours, sem.credits, sem.total_hours) == (30, 4, 144)
    assert (sem.trajectory, sem.study_form) == ("ОСНОВНАЯ ТРАЕКТОРИЯ", "очная")
    assert not [w for w in result.warnings if w.startswith("трудоёмкость")]


def test_same_header_name_in_two_groups_is_told_apart() -> None:
    """«промежуточная аттестация» есть и в контактной, и в самостоятельной работе."""
    rows = [["Семестр 2", "10", "", "", "4", "", "", "20", "", "1"]]
    row = _extract(_table(GROUP, LEAF, rows)).workload_table[0]
    assert row.contact_hours["промежуточная аттестация"] == 4
    assert row.self_study_hours["промежуточная аттестация"] == 20


def test_header_variants_and_abbreviations() -> None:
    """«Аудиторная» вместо «Контактная», «Трудоёмкость, ед.», «В т. ч. … учебно-методич.»."""
    group = [
        "Код модуля",
        "Аудиторная работа",
        "",
        "Самостоятельная работа",
        "",
        "Объём занятий",
        "Трудоёмкость, ед.",
    ]
    leaf = [
        "Код модуля",
        "Лекции",
        "Практические занятия",
        "Под руководством преподавателя",
        "В т. ч. с использованием учебно-методич. материалов",
        "Объём занятий в активных и интерактивных формах",
        "Трудоёмкость, ед.",
    ]
    rows = [["Семестр 4", "28", "28", "2", "49", "26", "4"]]
    row = _extract(_table(group, leaf, rows)).workload_table[0]
    assert row.contact_hours == {"лекции": 28, "практические занятия": 28}
    assert row.self_study_hours["с использованием методических материалов"] == 49
    assert (row.credits, row.active_hours) == (4, 26)


def test_group_size_rows_are_skipped() -> None:
    rows = [
        ["Семестр 1", "30", "", "", "", "", "", "", "30", "4"],
        ["Семестр 1", "1-60", "", "", "", "", "", "", "30", "4"],
    ]
    assert len(_extract(_table(GROUP, LEAF, rows)).workload_table) == 1


def test_decimal_comma_in_hours_and_credits() -> None:
    rows = [["Семестр 1", "13,5", "", "", "", "", "", "", "", "1,5"]]
    row = _extract(_table(GROUP, LEAF, rows)).workload_table[0]
    assert (row.contact_hours["лекции"], row.credits) == (13.5, 1.5)


def test_several_trajectories_are_kept_apart() -> None:
    rows = [
        ["ТРАЕКТОРИЯ 7 СЕМЕСТРА"],
        ["Форма обучения: очная"],
        ["Семестр 7", "36", "", "", "", "", "", "", "", "1"],
        ["ТРАЕКТОРИЯ 8 СЕМЕСТРА"],
        ["Форма обучения: заочная"],
        ["Семестр 8", "36", "", "", "", "", "", "", "", "1"],
    ]
    first, second = _extract(_table(GROUP, LEAF, rows)).workload_table
    assert (first.trajectory, first.study_form) == ("ТРАЕКТОРИЯ 7 СЕМЕСТРА", "очная")
    assert (second.trajectory, second.study_form) == ("ТРАЕКТОРИЯ 8 СЕМЕСТРА", "заочная")


def test_hours_not_matching_credits_give_warning() -> None:
    rows = [["Семестр 1", "30", "", "", "", "", "", "", "", "4"]]
    warnings = _extract(_table(GROUP, LEAF, rows)).warnings
    assert any("30 ч не равно 4 з.е." in w for w in warnings)


def test_total_not_matching_semesters_gives_warning() -> None:
    rows = [
        ["Семестр 1", "72", "", "", "", "", "", "", "", "2"],
        ["Семестр 2", "72", "", "", "", "", "", "", "", "2"],
        ["ИТОГО", "144", "", "", "", "", "", "", "", "5"],
    ]
    warnings = _extract(_table(GROUP, LEAF, rows)).warnings
    assert any("з.е. по семестрам (4) не равны ИТОГО (5)" in w for w in warnings)


def test_unrelated_table_is_not_taken_as_workload() -> None:
    text = "| Код | Название |\n| --- | --- |\n| УК-1 | Способен |\n"
    assert extract_rpd(text, file="x.md").workload_table == []


def test_missing_table_gives_warning() -> None:
    assert (
        "не найдена таблица трудоёмкости по видам занятий" in extract_rpd("text", "x.md").warnings
    )


@pytest.mark.skipif(find_md("Математический анализ I") is None, reason="нет md мат. анализа")
def test_real_math_analysis_workload() -> None:
    path = find_md("Математический анализ I")
    assert path is not None
    result = extract_rpd_file(path)
    sem, total = result.workload_table
    assert sem.semester == 1
    assert sem.contact_hours == {
        "лекции": 62,
        "консультации": 2,
        "практические занятия": 60,
        "контрольные работы": 4,
        "коллоквиумы": 2,
        "промежуточная аттестация": 4,
    }
    assert sem.self_study_hours == {
        "с использованием методических материалов": 51,
        "промежуточная аттестация": 31,
    }
    assert (sem.credits, sem.total_hours) == (6, 216)
    assert total.is_total
    assert not [w for w in result.warnings if w.startswith("трудоёмкость")]
