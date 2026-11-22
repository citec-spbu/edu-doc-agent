"""Разбор запроса и RRF (retrieval.ranking) — без базы и модели."""

import pytest

from edu_doc_agent.retrieval.ranking import QueryKeys, query_keys, rrf


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
