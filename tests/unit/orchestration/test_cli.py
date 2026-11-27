"""Терминальный чат: вывод вопроса графа с вариантами и выбор варианта номером."""

from edu_doc_agent.orchestration.cli import answer_text, render
from edu_doc_agent.orchestration.session import Turn


def test_question_is_shown_with_numbered_options() -> None:
    turn = Turn(reply=None, question={"question": "Применить?", "options": ["Да", "Нет"]})

    text, options = render(turn)

    assert text == "[вопрос] Применить?\n  1. Да\n  2. Нет\n  (номер варианта или свой ответ)"
    assert options == ["Да", "Нет"]
    assert render(Turn(reply="Готово.", question=None)) == ("Готово.", [])


def test_number_selects_option_and_text_stays_own_answer() -> None:
    options = ["Применить", "Отклонить"]

    assert answer_text("2", options) == "Отклонить"
    assert answer_text("3", options) == "3"  # нет такого варианта: свой ответ
    assert answer_text("оставь 8 з.е.", options) == "оставь 8 з.е."
