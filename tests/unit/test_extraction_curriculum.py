"""Извлечение сущностей из учебного плана: синтетический Markdown и (если есть) реальный план."""

# ruff: noqa: E501  (строки таблиц Markdown)

import pytest

from edu_doc_agent.extraction.curriculum import extract_plan
from tests.real_data import real_plan_markdown

_HEAD1 = "| Код Блока | Трудоёмкость | Код компетенции | Наименование | Виды текущего контроля успеваемости и (или) форма промежуточной аттестации | Аудиторная работа | | | Самостоятельная работа | | Объём занятий в активных и интерактивных формах |"  # noqa: E501
_HEAD2 = "| Код Блока | Трудоёмкость | Код компетенции | Наименование | Виды текущего контроля успеваемости и (или) форма промежуточной аттестации | Лекции | Практические занятия | Промежуточная аттестация | Текущий контроль | Промежуточная аттестация | Объём занятий в активных и интерактивных формах |"  # noqa: E501
_SEP = "| " + " | ".join(["---"] * 11) + " |"


def _ctx(text: str) -> str:
    return f"| {text} |" + " |" * 10


PLAN = f"""Программирование и информационные технологии Programming and Information Technology

| по уровню | бакалавриат |  |
| --- | --- | --- |
| по направлению (специальности) | 02.03.02 Фундаментальная информатика |  |
| Форма обучения: |  | очная |
| Срок обучения по основной образовательной программе |  | 4 года |

| Регистрационный номер | | 26/5003/1 |
| --- | --- | --- |
|  |  |  |

# Раздел 1. Формируемые компетенции

| Код компетенции | Наименование и (или) описание компетенции |
| --- | --- |
| УК-1 | Способен осуществлять поиск информации |
| ПКП-УНК-03 | Способен применять методы стратегического планирования |

# Раздел 2. Организация обучения

{_HEAD1}
{_HEAD2}
{_SEP}
{_ctx("1 год обучения")}
{_ctx("С01. Семестр 1")}
{_ctx("Базовая часть периода обучения")}
| Блок.1.дисц | 2 | УК-1, ПКП-УНК-03 | [000160] Алгебра<br>Algebra | зачёт, экзамен | 30 | 20 | 4 | 14 | 4 | 20 |
{_HEAD2}
{_ctx("Блок дисциплин Иностранный язык")}
{_ctx("Траектория 1 (0-B2)")}
| Блок.1.дисц | 1 | УК-1 | [700000] Английский язык<br>English | зачёт | 0 | 36 | 0 | 0 | 0 | 36 |
{_ctx("С02. Семестр 2")}
| Блок.1.дисц | 0 | УК-1 | [900000] Физкультура<br>PE | текущий контроль | 0 | 4 | 0 | 0 | 0 | 0 |
| Блок.1.дисц | 2 | УК-9 | [000161] Геометрия<br>Geometry | зачёт | 30 | 20 | 4 | 14 | 3 | 20 |

# Раздел 3. Структура и форма итоговой аттестации

| Код Блока | Трудоёмкость | Форма и наименование процедуры итоговой аттестации | Перечень кодов компетенций, проверяемых при проведении итоговой аттестации |
| --- | --- | --- | --- |
| И. Итоговая аттестация |  |  |  |
| Блок.3.гиа | 9 | Защита ВКР<br>Thesis Defense | УК-1, ОПК-1 |
"""


def test_title_competences_and_attestation() -> None:
    plan = extract_plan(PLAN, "plan.md")
    assert plan.level == "бакалавриат" and plan.study_form == "очная"
    assert plan.duration == "4 года" and plan.registration_number == "26/5003/1"
    assert (
        plan.program
        == "Программирование и информационные технологии Programming and Information Technology"
    )
    assert [c.code for c in plan.competences] == ["УК-1", "ПКП-УНК-03"]
    gia = plan.final_attestation[0]
    assert (gia.block, gia.credits, gia.name_en) == ("Блок.3.гиа", 9.0, "Thesis Defense")
    assert gia.competences == ["УК-1", "ОПК-1"]


def test_entries_keep_hours_groups_and_context() -> None:
    plan = extract_plan(PLAN, "plan.md")
    algebra, english, pe, geometry = plan.entries
    assert (algebra.code, algebra.name, algebra.name_en) == ("000160", "Алгебра", "Algebra")
    assert algebra.credits == 2 and algebra.control == ["зачёт", "экзамен"]
    assert algebra.competences == ["УК-1", "ПКП-УНК-03"]
    # «Промежуточная аттестация» есть и в аудиторной, и в самостоятельной работе
    assert algebra.contact_hours["промежуточная аттестация"] == 4
    assert algebra.self_study_hours["промежуточная аттестация"] == 4
    assert algebra.self_study_hours["текущий контроль"] == 14
    assert algebra.active_hours == 20
    assert (algebra.semester, algebra.year) == (1, 1)
    assert algebra.part == "Базовая часть периода обучения"
    assert english.group == "Блок дисциплин Иностранный язык"
    assert english.trajectory == "Траектория 1 (0-B2)"
    assert (pe.semester, pe.group, pe.trajectory) == (2, None, None)
    assert geometry.semester == 2 and geometry.part is None


def test_repeated_header_and_context_rows_are_not_entries() -> None:
    plan = extract_plan(PLAN, "plan.md")
    assert len(plan.entries) == 4


def test_warnings_hours_vs_credits_and_unknown_competence() -> None:
    plan = extract_plan(PLAN, "plan.md")
    text = "\n".join(plan.warnings)
    # у Геометрии 30+20+4+14+3 = 71 ч вместо 72, а УК-9 нет в таблице 1.1
    assert "000161" in text and "71 ч не равно 2 з.е." in text
    assert "УК-9" in text and "ОПК-1" in text
    # 0 з.е. (физкультура) — не ошибка; Алгебра сходится: 30+20+4+14+4 = 72
    assert "900000" not in text and "000160" not in text


def test_missing_tables_are_reported() -> None:
    plan = extract_plan("# Раздел 1\n\nтекст", "x.md")
    assert any("1.1" in w for w in plan.warnings) and any("раздел 2" in w for w in plan.warnings)


@pytest.mark.skipif(real_plan_markdown() is None, reason="нет data/Учебный план.zip")
def test_real_plan_has_no_warnings() -> None:
    plan = extract_plan(real_plan_markdown() or "", "plan.docx")
    assert plan.registration_number == "26/5003/1"
    assert len(plan.competences) == 49 and len(plan.entries) == 226
    assert plan.final_attestation[0].credits == 9
    assert plan.warnings == []
