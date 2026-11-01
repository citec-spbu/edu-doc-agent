"""Прогон LLM-извлечения по документу: схема, повторы, проверка цитат, источники."""

import json
from pathlib import Path

from edu_doc_agent.extraction.llm_extract import MAX_RETRIES, extract_with_llm
from edu_doc_agent.llm import FakeLLMClient, LLMCall

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rpd_full_synthetic.md"
MD = FIXTURE.read_text(encoding="utf-8")

PREREQ = {
    "items": [
        {
            "name": "алгебра и геометрия",
            "kind": "дисциплина",
            "relation": "предшествует",
            "quote": "«Алгебра и геометрия»",
        }
    ]
}
ACTIVE = {
    "total_hours": None,
    "items": [
        {
            "kind": "консультации",
            "hours": 2,
            "mandatory": True,
            "quote": "Консультации в активной форме – 2 часа",
        }
    ],
}
RULES = {"rules": []}


def _by_prompt(call: LLMCall) -> str:
    if "пререквизиты" in call.user:
        return json.dumps(PREREQ, ensure_ascii=False)
    if "активных и интерактивных" in call.user:
        return json.dumps(ACTIVE, ensure_ascii=False)
    return json.dumps(RULES)


def test_all_tasks_run_and_items_get_sources() -> None:
    client = FakeLLMClient(respond=_by_prompt)
    result = extract_with_llm(MD, "rpd.md", client)
    assert [p.name for p in result.prerequisites] == ["алгебра и геометрия"]
    assert result.prerequisites[0].source.section.startswith("1.2.")
    assert result.prerequisites[0].source.line is not None
    assert [(f.kind, f.hours) for f in result.active_forms] == [("консультации", 2)]
    assert result.warnings == []
    assert len(client.calls) == 3  # по одному вызову на задачу: 1.2, 1.4, 3.1.3


def test_each_call_has_schema_and_section_text_only() -> None:
    client = FakeLLMClient(respond=_by_prompt)
    extract_with_llm(MD, "rpd.md", client, tasks=["prerequisites"])
    (call,) = client.calls
    assert "items" in call.schema["properties"]
    assert "Алгебра и геометрия" in call.user and "Консультации в активной" not in call.user


def test_invalid_json_is_retried_with_error_in_prompt() -> None:
    client = FakeLLMClient(replies=["{не json", json.dumps(PREREQ, ensure_ascii=False)])
    result = extract_with_llm(MD, "rpd.md", client, tasks=["prerequisites"])
    assert len(result.prerequisites) == 1 and result.warnings == []
    assert "не принят" in client.calls[1].user and "не принят" not in client.calls[0].user


def test_gives_up_after_retries_with_warning() -> None:
    client = FakeLLMClient(respond=lambda call: "{}")  # нет обязательного поля items
    result = extract_with_llm(MD, "rpd.md", client, tasks=["prerequisites"])
    assert len(client.calls) == MAX_RETRIES + 1
    assert result.prerequisites == []
    assert "после 3 попыток" in result.warnings[0]


def test_hallucinated_quote_is_dropped_with_warning() -> None:
    bad = {"items": [{**PREREQ["items"][0], "quote": "курс квантовой механики"}]}
    client = FakeLLMClient(replies=[json.dumps(bad, ensure_ascii=False)])
    result = extract_with_llm(MD, "rpd.md", client, tasks=["prerequisites"])
    assert result.prerequisites == []
    assert "цитата не найдена" in result.warnings[0]


def test_total_hours_not_in_text_is_not_trusted() -> None:
    reply = json.dumps({**ACTIVE, "total_hours": 999}, ensure_ascii=False)
    result = extract_with_llm(MD, "rpd.md", FakeLLMClient(replies=[reply]), tasks=["active_forms"])
    assert result.active_total_hours is None
    assert any("не найден в тексте" in w for w in result.warnings)


def test_document_without_sections_makes_no_calls() -> None:
    client = FakeLLMClient(respond=_by_prompt)
    result = extract_with_llm("# Пусто\n\nтекст\n", "x.md", client)
    assert client.calls == [] and result.warnings == []


# --- подключение к extract_rpd -------------------------------------------------------------


def test_extract_rpd_without_llm_is_offline_and_has_no_llm_part() -> None:
    from edu_doc_agent.extraction import extract_rpd

    assert extract_rpd(MD, "x.md").llm is None


def test_extract_rpd_with_llm_adds_semantic_fields_and_keeps_rules() -> None:
    from edu_doc_agent.extraction import extract_rpd

    client = FakeLLMClient(respond=_by_prompt)
    result = extract_rpd(MD, "x.md", llm=client)
    assert result.llm is not None
    assert [p.name for p in result.llm.prerequisites] == ["алгебра и геометрия"]
    # правила (Yargy) работают независимо от LLM
    assert {"Алгебра и геометрия", "Основы дискретной математики"} <= {
        p.name for p in result.prerequisites
    }
    assert len(client.calls) == 3


def test_llm_warnings_and_disagreement_with_table_reach_result_warnings() -> None:
    from edu_doc_agent.extraction import extract_rpd

    bad = {"items": [{**PREREQ["items"][0], "quote": "курс квантовой механики"}]}
    active = {**ACTIVE, "total_hours": 34}  # в 1.4 такого числа нет
    replies = {"пререквизиты": bad, "активных и интерактивных": active}

    def respond(call: LLMCall) -> str:
        for key, data in replies.items():
            if key in call.user:
                return json.dumps(data, ensure_ascii=False)
        return json.dumps(RULES)

    result = extract_rpd(MD, "x.md", llm=FakeLLMClient(respond=respond))
    assert any(w.startswith("LLM: prerequisites: цитата не найдена") for w in result.warnings)
    assert any("не найден в тексте" in w for w in result.warnings)


def test_llm_total_that_differs_from_table_2_1_is_reported() -> None:
    from edu_doc_agent.extraction import extract_rpd

    text = MD.replace("Консультации в активной форме – 2 часа.", "Объём составляет 10 часов.")
    active = {"total_hours": 10, "items": []}

    def respond(call: LLMCall) -> str:
        if "активных и интерактивных" in call.user:
            return json.dumps(active)
        return json.dumps(PREREQ if "пререквизиты" in call.user else RULES, ensure_ascii=False)

    result = extract_rpd(text, "x.md", llm=FakeLLMClient(respond=respond))
    assert result.llm.active_total_hours == 10
    assert any("не совпадает с таблицей 2.1 (34)" in w for w in result.warnings)


def test_sum_of_form_hours_is_checked_against_table_when_total_is_not_stated() -> None:
    from edu_doc_agent.extraction import extract_rpd

    active = {"total_hours": None, "items": [{**ACTIVE["items"][0], "hours": 10}]}

    def respond(call: LLMCall) -> str:
        if "активных и интерактивных" in call.user:
            return json.dumps(active)
        return json.dumps(PREREQ if "пререквизиты" in call.user else RULES, ensure_ascii=False)

    result = extract_rpd(MD, "x.md", llm=FakeLLMClient(respond=respond))
    assert result.llm.active_total_hours is None  # итог модель не называла и не считала
    assert any(
        "сумма часов активных форм в 1.4 (10 ч)" in w and "(34)" in w for w in result.warnings
    )

    active["items"][0]["hours"] = None  # часов нет — сумма неполная, сравнивать нечего
    result = extract_rpd(MD, "x.md", llm=FakeLLMClient(respond=respond))
    assert not any("не совпадает" in w for w in result.warnings)


# --- выбор разделов, таблицы, постобработка ---------------------------------------------------

from edu_doc_agent.extraction.llm_extract import (  # noqa: E402
    postprocess,
    section_text,
    select_active_forms,
    select_grading,
    select_prerequisites,
)
from edu_doc_agent.extraction.llm_schemas import (  # noqa: E402
    ActiveFormsOutput,
    PrerequisitesOutput,
)
from edu_doc_agent.extraction.md_utils import split_sections  # noqa: E402

_PRACTICE = """### 1.5. Требования подготовленности к прохождению практики
Нужны знания по алгебре.
### 3.1.3. Методические материалы для руководителей практики
#### 3.1.3.1. Методика проведения текущего контроля
Отчёт принимается, если оформлен по образцу.
#### 3.1.3.2. Методика и критерии оценивания
Просрочка снижает оценку.
#### 3.1.3.3. Оценочные средства: контрольно-измерительные материалы
Примеры вопросов.
#### 3.1.3.4. Рекомендуемая форма отчета о практике
Титульный лист.
### 3.2. Кадровое обеспечение
Текст.
"""

_COURSE = """### 3.1.3. Методика проведения текущего контроля
Работа зачтена, если выполнена верно.
### Критерии оценивания контрольной работы
| Критерий | Баллы |
| --- | --- |
| Все задания верно | 3 |
| Все задания верно | 3 |
| Нет решения | 0 |
### Таблица. Критерии выставления оценок за экзамен
Ответ полный — отлично.
### 3.1.4. Методические материалы для проведения контроля
Билеты.
### Раздел 4. Разработчики
Иванов.
"""


def test_prerequisites_of_practice_are_found_by_heading_not_number() -> None:
    (section,) = select_prerequisites(split_sections(_PRACTICE))
    assert section.number == "1.5"


def test_practice_has_no_active_forms_section() -> None:
    assert select_active_forms(split_sections(_PRACTICE)) == []


def test_grading_of_practice_takes_methodology_and_criteria_only() -> None:
    numbers = [s.number for s in select_grading(split_sections(_PRACTICE))]
    assert numbers == ["3.1.3", "3.1.3.1", "3.1.3.2"]


def test_grading_includes_unnumbered_subsections_until_next_numbered_heading() -> None:
    picked = select_grading(split_sections(_COURSE))
    assert [s.heading[:8] for s in picked] == ["3.1.3. М", "Критерии", "Таблица."]


def test_tables_reach_the_model_as_text_rows_without_duplicate_cells() -> None:
    picked = select_grading(split_sections(_COURSE))
    text = section_text(picked, with_tables=True)
    assert "Все задания верно | 3\n" in text and "Нет решения | 0" in text
    assert "Критерии оценивания контрольной работы" in text  # заголовок подраздела сохранён
    assert "---" not in text
    assert "Нет решения" not in section_text(picked, with_tables=False)


def test_split_sections_keeps_line_numbers_and_levels() -> None:
    first, second, *_ = split_sections(_COURSE)
    assert first.level == 3 and second.line == 3
    assert len(second.text.split("\n")) == len(second.full_text.split("\n"))


def test_grading_rules_get_source_of_the_subsection_where_quote_is_found() -> None:
    answer = {
        "rules": [
            {"condition": "нет решения", "effect": "0 баллов", "quote": "Нет решения | 0"},
            {"condition": "работа верна", "effect": "зачёт", "quote": "Работа зачтена, если"},
        ]
    }
    client = FakeLLMClient(respond=lambda call: json.dumps(answer, ensure_ascii=False))
    result = extract_with_llm(_COURSE, "r.md", client, tasks=["grading_rules"])
    assert [r.source.section[:8] for r in result.grading_rules] == ["Критерии", "3.1.3. М"]
    assert len(client.calls) == 1


def test_same_quote_and_hours_are_one_form_and_kinds_are_lowercase() -> None:
    quote = "Практические и лабораторные занятия (24 часа) проводятся в активной форме"
    forms = [
        {"kind": "Практические занятия", "hours": 24, "mandatory": True, "quote": quote},
        {"kind": "лабораторные занятия", "hours": 24, "mandatory": True, "quote": quote},
        {"kind": "Лекции", "hours": 10, "mandatory": True, "quote": "Лекции-10"},
    ]
    out = postprocess(ActiveFormsOutput.model_validate({"total_hours": None, "items": forms}))
    assert [(f.kind, f.hours) for f in out.items] == [
        ("практические занятия и лабораторные занятия", 24),
        ("лекции", 10),
    ]


def test_forms_with_own_equal_hours_in_one_sentence_are_not_merged() -> None:
    quote = "Объём составляет 32 часа: семинары — 16 часов, практические занятия — 16 часов"
    forms = [
        {"kind": "семинары", "hours": 16, "mandatory": True, "quote": quote},
        {"kind": "практические занятия", "hours": 16, "mandatory": True, "quote": quote},
    ]
    out = postprocess(ActiveFormsOutput.model_validate({"total_hours": 32, "items": forms}))
    assert [(f.kind, f.hours) for f in out.items] == [
        ("семинары", 16),
        ("практические занятия", 16),
    ]


def test_forms_with_unknown_hours_are_not_merged() -> None:
    forms = [
        {"kind": "семинары", "hours": None, "mandatory": True, "quote": "общая цитата"},
        {"kind": "консультации", "hours": None, "mandatory": True, "quote": "общая цитата"},
    ]
    out = postprocess(ActiveFormsOutput.model_validate({"total_hours": None, "items": forms}))
    assert len(out.items) == 2


def test_repeated_prerequisite_names_are_collapsed() -> None:
    item = {"name": "Алгебра", "kind": "дисциплина", "relation": "предшествует", "quote": "алгебра"}
    out = postprocess(
        PrerequisitesOutput.model_validate({"items": [item, {**item, "name": "алгебра"}]})
    )
    assert len(out.items) == 1
