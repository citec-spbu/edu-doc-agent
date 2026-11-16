"""Извлечение сущностей из ЦКМВ: синтетический Markdown и (если есть) реальный документ."""

# ruff: noqa: E501  (строки таблиц Markdown)

import pytest

from edu_doc_agent.extraction.ckmv import extract_ckmv
from tests.real_data import real_ckmv_markdown

H1 = "| Основная профессиональная роль: | Профессиональные компетенции базового уровня | Индикаторы профессиональных компетенций базового уровня | Результаты обучения по индикаторам профессиональных компетенций базового уровня |"
H2 = "| | Профессиональные перекрестные компетенции топ-уровня | Индикаторы профессиональных перекрестных компетенций топ-уровня | Результаты обучения по индикаторам профессиональных перекрестных компетенций топ-уровня |"
SEP = "| --- | --- | --- | --- |"
LEAD = "| Профессиональная опережающая роль: | | | |"

MD = f"""### Перечень ролей

{H1}
{SEP}
| MLOps / Data Scientist | ПКП-МО-1. Способен проектировать модели | ПКП-МО-1.1. Применяет методы (МО). (Базовый) | ПКП-МО-1.1 З-1. Знает алгоритмы. Линейные методы. |
| | | | ПКП-МО-1.1 У-1. Умеет применять библиотеки. |
| | ПКП-МО-2. Способен внедрять MLOps | ПКП-МО-2.1. Настраивает CI/CD (Средний) | ПКП-МО-2.1. З-1. Знает принципы CI/CD. |
| | | | ПКП-МО-2.1. У-1. Умеет настраивать.<br>ПКП-МО-2.1. У-2. Умеет отлаживать. |
{H2}
| | ПКП-2-ППК-Р2. Способен проверять код | ПКП-2-ППК-Р2.1. Разрабатывает тесты (Продвинутый) | ПКП-2-ППК-Р2.1. 3-1. Знает методы тестов. |
| | | | ПКП-2-ППК-Р2.1. У-1 Владеет навыками. |
{LEAD}
| Руководитель группы | Профессиональные опережающие компетенции | Индикаторы опережающих | Результаты обучения опережающих |
| | ПКП-22-РГ-2. Способен организовать процессы | ПКП-22-РГ-2.1. Управляет проектированием | ПКП-22-РГ-2.1. З-1. Знает принципы. |
"""


def test_roles_groups_and_codes() -> None:
    doc = extract_ckmv(MD, "x.md")
    codes = [(c.code, c.group, c.role_section, c.role) for c in doc.competences]
    base = "Профессиональные компетенции базового уровня"
    assert codes[0] == (
        "ПКП-МО-1",
        base,
        "Основная профессиональная роль",
        "MLOps / Data Scientist",
    )
    assert codes[1][1:] == (base, "Основная профессиональная роль", "MLOps / Data Scientist")
    # роль и секция сохраняются, группа меняется строкой-шапкой; код с номером хранится как есть
    assert codes[2] == (
        "ПКП-2-ППК-Р2",
        "Профессиональные перекрестные компетенции топ-уровня",
        "Основная профессиональная роль",
        "MLOps / Data Scientist",
    )
    assert codes[3] == (
        "ПКП-22-РГ-2",
        "Профессиональные опережающие компетенции",
        "Профессиональная опережающая роль",
        "Руководитель группы",
    )


def test_indicators_levels_and_empty_cells_carry_over() -> None:
    doc = extract_ckmv(MD, "x.md")
    mo1, mo2 = doc.competences[0], doc.competences[1]
    assert [i.code for i in mo1.indicators] == ["ПКП-МО-1.1"]
    # уровень вынесен из названия, точка после «(МО)» убрана, сама скобка остаётся в названии
    assert (mo1.indicators[0].name, mo1.indicators[0].level) == ("Применяет методы (МО)", "Базовый")
    assert mo2.indicators[0].level == "Средний"
    assert doc.competences[2].indicators[0].level == "Продвинутый"
    assert doc.competences[3].indicators[0].level is None


def test_outcomes_split_by_br_and_normalised() -> None:
    doc = extract_ckmv(MD, "x.md")
    mo2 = doc.competences[1].indicators[0]
    assert [o.code for o in mo2.outcomes] == ["ПКП-МО-2.1 З-1", "ПКП-МО-2.1 У-1", "ПКП-МО-2.1 У-2"]
    first = doc.competences[0].indicators[0].outcomes[0]
    assert (first.letter, first.number, first.kind) == ("З", 1, "знает")
    assert first.text == "Знает алгоритмы. Линейные методы."
    # опечатка «3-1» (цифра вместо З) исправляется на З
    typo = doc.competences[2].indicators[0].outcomes[0]
    assert (typo.code, typo.letter, typo.kind) == ("ПКП-2-ППК-Р2.1 З-1", "З", "знает")
    assert first.source.line is not None


def test_warnings_for_contradictions_and_missing_data() -> None:
    doc = extract_ckmv(MD, "x.md")
    text = "\n".join(doc.warnings)
    assert "ПКП-2-ППК-Р2.1 У-1" in text and "владеет" in text  # «У-1 Владеет …»
    assert "ПКП-22-РГ-2.1" in text and "не указан уровень" in text


def test_foreign_indicator_and_outcome_codes_are_reported() -> None:
    md = f"""{H1}
{SEP}
| Роль | ПКП-МО-1. Способен | ПКП-МО-2.1. Другой код (Базовый) | ПКП-МО-3.1. З-1. Знает. |
"""
    text = "\n".join(extract_ckmv(md, "x.md").warnings)
    assert "не относится к компетенции ПКП-МО-1" in text
    assert "не относится к индикатору ПКП-МО-2.1" in text


def test_no_table_is_reported() -> None:
    assert "не найдена" in "\n".join(extract_ckmv("# текст", "x.md").warnings)


@pytest.mark.skipif(real_ckmv_markdown() is None, reason="нет data/ЦКМВ.zip")
def test_real_ckmv_counts_and_known_issue() -> None:
    doc = extract_ckmv(real_ckmv_markdown() or "", "ckmv.docx")
    assert (len(doc.competences), len(doc.indicators), len(doc.outcomes)) == (31, 85, 409)
    # единственное найденное противоречие документа: «У-1 Владеет навыками презентации…»
    assert len(doc.warnings) == 1 and "УНК-01.01 У-1" in doc.warnings[0]
