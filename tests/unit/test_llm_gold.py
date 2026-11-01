"""Эталонные кейсы LLM-извлечения: корректность набора и работа замера на подменной модели."""

import json

import pytest

from edu_doc_agent.eval.llm_gold import (
    BLIND,
    BLIND2,
    HOLDOUT,
    THRESHOLDS,
    GoldCase,
    evaluate,
    load_gold,
    score_case,
)
from edu_doc_agent.extraction import llm_prompts
from edu_doc_agent.extraction.llm_extract import TASKS_BY_NAME, normalize, quote_in_text
from edu_doc_agent.llm import FakeLLMClient, LLMCall
from tests.real_data import find_md

DEV_CASES = load_gold()
HOLDOUT_CASES = load_gold(HOLDOUT)
BLIND_CASES = load_gold(BLIND)
BLIND2_CASES = load_gold(BLIND2)
CASES = DEV_CASES + HOLDOUT_CASES + BLIND_CASES + BLIND2_CASES
_ITEMS = {"prerequisites": "items", "active_forms": "items", "grading_rules": "rules"}


def _quotes(case: GoldCase) -> list[str]:
    return [i["quote"] for i in case.expected[_ITEMS[case.task]]]


def test_gold_covers_every_task_with_positive_and_negative_cases() -> None:
    assert len({c.id for c in CASES}) == len(CASES)
    assert {c.task for c in CASES} == set(TASKS_BY_NAME)
    for task in TASKS_BY_NAME:
        own = [c for c in CASES if c.task == task]
        assert len(own) >= 3, task
        assert any(not c.expected[_ITEMS[task]] for c in own) or task == "active_forms", task
        assert any(c.expected[_ITEMS[task]] for c in own), task


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_expected_matches_output_schema(case: GoldCase) -> None:
    TASKS_BY_NAME[case.task].output.model_validate(case.expected)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_expected_quotes_are_verbatim_substrings(case: GoldCase) -> None:
    for quote in _quotes(case):
        assert quote in case.text, quote


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_case_text_is_verbatim_from_real_document(case: GoldCase) -> None:
    path = find_md(case.source)
    if path is None:
        pytest.skip(f"нет документа {case.source} в data/")
    document = normalize(path.read_text(encoding="utf-8"))
    for line in case.text.split("\n"):
        assert normalize(line) in document, line


def test_prompts_do_not_contain_gold_texts() -> None:
    """Примеры в промптах не должны совпадать с эталоном, иначе замер завышен."""
    prompts = normalize(
        " ".join(
            (
                llm_prompts.SYSTEM,
                llm_prompts.PREREQUISITES,
                llm_prompts.ACTIVE_FORMS,
                llm_prompts.GRADING_RULES,
            )
        )
    )
    for case in CASES:
        for line in case.text.split("\n"):
            assert normalize(line) not in prompts, line


# --- замер на подменной модели ----------------------------------------------------------------


def _oracle(call: LLMCall) -> str:
    case = next(c for c in CASES if c.text in call.user)
    return json.dumps(case.expected, ensure_ascii=False)


def test_oracle_model_gets_perfect_score() -> None:
    report = evaluate(FakeLLMClient(respond=_oracle), CASES)
    assert report.passed(), report.summary()
    assert all(s.f1 == 1.0 and s.attribute_accuracy == 1.0 for s in report.by_task.values())


def test_missing_and_hallucinated_items_lower_scores() -> None:
    def noisy(call: LLMCall) -> str:
        data = json.loads(_oracle(call))
        key = "rules" if "rules" in data else "items"
        data[key] = data[key][1:]  # теряем первый элемент
        if key == "items" and "total_hours" not in data:
            data[key].append(
                {
                    "name": "выдумка",
                    "kind": "дисциплина",
                    "relation": "предшествует",
                    "quote": "выдумка",
                }
            )
        return json.dumps(data, ensure_ascii=False)

    report = evaluate(FakeLLMClient(respond=noisy))
    assert not report.passed()
    assert report.by_task["prerequisites"].fn > 0 and report.by_task["prerequisites"].f1 < 1


def test_quote_not_from_text_counts_as_not_found() -> None:
    case = next(c for c in CASES if c.id == "prereq-algebra-parallel")
    reply = json.dumps(
        {
            "items": [
                {
                    "name": "математический анализ",
                    "kind": "дисциплина",
                    "relation": "параллельно",
                    "quote": "цитаты нет в тексте",
                }
            ]
        },
        ensure_ascii=False,
    )
    report = evaluate(FakeLLMClient(replies=[reply]), [case])
    assert report.by_case[case.id].tp == 0 and report.by_case[case.id].fn == 1


def test_wrong_attribute_is_reported_separately() -> None:
    expected = {
        "items": [
            {"name": "алгебра", "kind": "дисциплина", "relation": "предшествует", "quote": "q"}
        ]
    }
    predicted = {
        "items": [
            {"name": "Алгебра", "kind": "дисциплина", "relation": "параллельно", "quote": "q"}
        ]
    }
    score = score_case("prerequisites", predicted, expected)
    assert (score.tp, score.fp, score.fn) == (1, 0, 0)
    assert score.attribute_accuracy == 0.5


def test_unparsable_answers_are_failures_after_retries() -> None:
    case = CASES[0]
    client = FakeLLMClient(replies=["не json"] * 3)
    report = evaluate(client, [case])
    assert case.id in report.failures and len(client.calls) == 3
    assert report.by_task[case.task].fn == len(case.expected[_ITEMS[case.task]])


def test_thresholds_match_adr_target_for_text_fields() -> None:
    assert set(THRESHOLDS.values()) == {0.8}


def test_quote_matching_ignores_case_dashes_and_yo() -> None:
    assert quote_in_text("ПРАКТИЧЕСКИЕ занятия – 48 час", "практические занятия - 48 час.")
    assert quote_in_text("учёта", "без учета")
    assert not quote_in_text("", "любой текст")


@pytest.mark.llm
def test_real_model_meets_thresholds_on_gold() -> None:
    """Настоящая модель из .env (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL): `pytest --run-llm`."""
    from edu_doc_agent.llm import OpenAICompatibleClient

    report = evaluate(OpenAICompatibleClient.from_settings())
    assert report.passed(), report.summary()


def test_quote_with_ellipsis_is_checked_piece_by_piece() -> None:
    text = "обладали знаниями по курсам: алгебра, математический анализ, аналитическая геометрия"
    assert quote_in_text("знаниями по курсам: ... математический анализ", text)
    assert not quote_in_text("знаниями по курсам: ... квантовая механика", text)


def test_metric_ignores_case_endings_and_trailing_dot_in_quotes() -> None:
    forms = score_case(
        "active_forms",
        {"items": [{"kind": "Лабораторных работ", "hours": 50, "mandatory": True, "quote": "q"}]},
        {"items": [{"kind": "лабораторные работы", "hours": 50, "mandatory": True, "quote": "q"}]},
    )
    assert (forms.tp, forms.fp, forms.fn) == (1, 0, 0)
    rules = score_case(
        "grading_rules",
        {"rules": [{"condition": "c", "effect": "e", "quote": "Работа зачтена, если все верно."}]},
        {"rules": [{"condition": "c", "effect": "e", "quote": "Работа зачтена, если все верно"}]},
    )
    assert (rules.tp, rules.fp, rules.fn) == (1, 0, 0)


def test_sets_are_disjoint() -> None:
    sets = [DEV_CASES, HOLDOUT_CASES, BLIND_CASES, BLIND2_CASES]
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            assert not {c.id for c in a} & {c.id for c in b}
            assert not {c.text for c in a} & {c.text for c in b}
            assert not {c.source for c in a} & {c.source for c in b} or a is not sets[0]
    assert len(HOLDOUT_CASES) >= 20 and len(BLIND_CASES) >= 20


@pytest.mark.llm
def test_real_model_on_holdout() -> None:
    """Честная оценка: на этих кейсах промпты не настраивались."""
    from edu_doc_agent.llm import OpenAICompatibleClient

    report = evaluate(OpenAICompatibleClient.from_settings(), HOLDOUT_CASES)
    assert report.passed(), report.summary()


@pytest.mark.llm
def test_real_model_on_blind2() -> None:
    """Четвёртый набор: размечен после правок выбора разделов и таблиц, до запуска модели."""
    from edu_doc_agent.llm import OpenAICompatibleClient

    report = evaluate(OpenAICompatibleClient.from_settings(), BLIND2_CASES)
    assert report.passed(), report.summary()


def test_prerequisite_names_match_by_words_one_to_one() -> None:
    gold = {
        "items": [
            {
                "name": "разработка простых программ",
                "kind": "навык",
                "relation": "предшествует",
                "quote": "q",
            }
        ]
    }
    longer = {"items": [{**gold["items"][0], "name": "разработка простых программ на языках"}]}
    assert score_case("prerequisites", longer, gold).tp == 1
    two = {
        "items": [
            {**gold["items"][0], "name": "аналитическая геометрия"},
            {**gold["items"][0], "name": "дифференциальная геометрия"},
        ]
    }
    merged = {"items": [{**gold["items"][0], "name": "аналитическая и дифференциальная геометрия"}]}
    s = score_case("prerequisites", merged, two)
    assert (s.tp, s.fp, s.fn) == (1, 0, 1)  # склейка двух названий не прощается целиком


def test_grading_rule_cut_out_of_gold_sentence_is_not_a_false_positive() -> None:
    gold = {
        "rules": [
            {"condition": "c", "effect": "e", "quote": "Вопросы — по 3 балла, задача — 4 балла"}
        ]
    }
    pred = {
        "rules": [
            {"condition": "c", "effect": "e", "quote": "Вопросы — по 3 балла, задача — 4 балла"},
            {"condition": "c", "effect": "e", "quote": "задача — 4 балла"},
            {"condition": "c", "effect": "e", "quote": "штраф за опоздание"},
        ]
    }
    s = score_case("grading_rules", pred, gold)
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)  # лишней считается только выдумка
