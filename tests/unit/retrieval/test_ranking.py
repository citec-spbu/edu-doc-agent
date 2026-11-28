"""Разбор запроса и RRF (retrieval.ranking) — без базы и модели."""

import pytest

from edu_doc_agent.retrieval.ranking import (
    QueryKeys,
    clause_of,
    collapse_lists,
    natural_key,
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
        ("пункт 2.3 и 2.4", (), ("2.3", "2.4")),
        ("пп. 2.3, 2.4 или 3.1 ФГОС", (), ("2.3", "2.4", "3.1")),
        ("п. 1.9, 02.03.02", (), ("1.9",)),  # код направления в перечне — не пункт
        # индикатор — не код компетенции: УК-1 из «УК-1.2» точно не ищется
        ("Что такое УК-1.2?", (), ()),
        ("ук-1.2 и УК-2. Что ещё?", ("УК-2",), ()),
        # закон: часть статьи, пункт статьи, вся статья; пункт части — это часть
        ("что в ч. 5 ст. 13", (), ("ст.13:ч.5",)),
        ("часть 3.1 статьи 13 закона", (), ("ст.13:ч.3.1",)),
        ("ст. 13, ч. 5", (), ("ст.13:ч.5",)),
        ("п. 6 ст. 2", (), ("ст.2:п.6",)),
        ("п. 2 ч. 7 ст. 13", (), ("ст.13:ч.7",)),
        ("статья 13 и п. 2.3", (), ("2.3", "ст.13")),
        # пункт приказа — целый номер; «п. 2.3» — пункт ФГОС, «1)» — не пункт
        ("п. 15 Порядка", (), ("15",)),
        ("пункт 17(1).", (), ("17(1)",)),
        ("перечень: 1) п. 3)", (), ()),
        # трудовая функция — вместе с кодом профстандарта, буквы — латиницей
        ("06.001 A/01.3", ("06.001:A/01.3",), ()),
        ("ТФ А/02.3 профстандарта 06.001", ("06.001:A/02.3",), ()),
        ("06.001 ОТФ D", ("06.001:D",), ()),
        ("06.001, A.3", ("06.001:A",), ()),
        ("A/01.3 без профстандарта", (), ()),
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


def test_natural_key() -> None:
    ids = ["ст.13:ч.10", "ст.13:ч.2", "ст.13:ч.3.1", "ст.2:п.6"]
    assert sorted(ids, key=natural_key) == ["ст.2:п.6", "ст.13:ч.2", "ст.13:ч.3.1", "ст.13:ч.10"]


def test_number_order() -> None:
    assert sorted(["2.2.10", "2.2.9", "2.2", "2.2.4.1"], key=number_order) == [
        "2.2",
        "2.2.4.1",
        "2.2.9",
        "2.2.10",
    ]
