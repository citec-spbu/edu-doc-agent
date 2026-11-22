"""Разбор запроса и RRF (retrieval.ranking) — без базы и модели."""

import pytest

from edu_doc_agent.retrieval.ranking import (
    QueryKeys,
    clause_of,
    collapse_lists,
    number_order,
    query_keys,
    rrf,
)


@pytest.mark.parametrize(
    ("query", "codes", "numbers"),
    [
        ("Что требует УК-1?", ("УК-1",), ()),
        ("что требует ук-1 и укб-2", ("УК-1", "УКБ-2"), ()),
        ("опк 2 и ОПК‑2, а также 06.001", ("ОПК-2", "06.001"), ()),
        ("Что сказано в п. 2.3 и пункте 4.1.2?", (), ("2.3", "4.1.2")),
        ("подпункт 2.1.6.1 ОС", (), ("2.1.6.1",)),
        # не коды и не пункты: направление подготовки, число, номер без «п.»
        ("Объем программы 02.03.02 — 240 з.е., см. 1.9", (), ()),
    ],
)
def test_query_keys(query: str, codes: tuple, numbers: tuple) -> None:
    assert query_keys(query) == QueryKeys(codes, numbers)


def test_empty_keys_are_falsy() -> None:
    assert not query_keys("какой объем программы бакалавриата")
    assert query_keys("УК-1")


def test_rrf_rewards_agreement_between_lists() -> None:
    full_text = ["a", "b", "c"]
    vector = ["c", "d", "a"]
    # a: 1/61 + 1/63; c: 1/63 + 1/61 — поровну, выше тот, кто раньше в первом списке
    assert rrf([full_text, vector]) == ["a", "c", "b", "d"]


def test_rrf_counts_repeated_key_once_by_best_place() -> None:
    # части одного пункта дают один ключ несколько раз
    assert rrf([["x", "y", "x"], ["y"]]) == ["y", "x"]


def test_rrf_handles_empty_lists() -> None:
    assert rrf([[], ["a"]]) == ["a"]
    assert rrf([]) == []


def test_weighted_rrf_lets_stronger_list_win() -> None:
    full_text, vector = ["a", "b"], ["b", "a"]
    assert rrf([full_text, vector]) == ["a", "b"]  # поровну — первый список решает
    assert rrf([full_text, vector], [0.5, 1.0]) == ["b", "a"]  # вектор весомее
    assert rrf([full_text, vector], [0.0, 1.0]) == ["b", "a"]  # текст не учитывается
    assert rrf([["c"], vector], [0.0, 1.0]) == ["b", "a"]  # и его ключи не попадают


OS = "spbu-os"
HEADS = {(OS, "1", "2.2"), (OS, "1", "2.2.4")}


def key(number: str, doc: str = OS) -> tuple[str, str]:
    return (f"{doc}:{number}", "1")


def test_clause_of() -> None:
    assert clause_of("spbu-os:2.2.4.1") == ("spbu-os", "2.2.4.1")
    assert clause_of("fgos-02.03.02:3.2:УК-1") is None
    assert clause_of("fgos-02.03.02:app:06.001") is None


def test_list_item_collapses_to_nearest_list_head() -> None:
    assert collapse_lists([key("2.2.4.1"), key("2.2.4.3"), key("1.4")], HEADS) == [
        (key("2.2.4"), True),  # оба подпункта — один результат
        (key("1.4"), False),
    ]


def test_outer_list_replaces_nested_one_at_best_place() -> None:
    keys = [key("1.4"), key("2.2.4.1"), key("2.2.1"), key("2.7")]
    assert collapse_lists(keys, HEADS) == [
        (key("1.4"), False),
        (key("2.2"), True),  # 2.2.4 вложен в 2.2 — остаётся внешний, на месте 2.2.4
        (key("2.7"), False),
    ]


def test_competencies_and_other_documents_untouched() -> None:
    uk = ("fgos-02.03.02:3.2:УК-1", "1")
    other = key("2.2.4.1", doc="fgos-01.03.02")  # перечень есть только в ОС
    assert collapse_lists([uk, other, uk], HEADS) == [(uk, False), (other, False)]


def test_number_order() -> None:
    assert sorted(["2.2.10", "2.2.9", "2.2", "2.2.4.1"], key=number_order) == [
        "2.2",
        "2.2.4.1",
        "2.2.9",
        "2.2.10",
    ]
