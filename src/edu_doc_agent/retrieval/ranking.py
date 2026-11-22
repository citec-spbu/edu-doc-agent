"""Разбор запроса и слияние рангов для поиска по нормативам (ADR-0005, решение 3).

Без зависимостей от базы и модели: только то, что можно проверить на строках.
"""

import re
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from edu_doc_agent.retrieval.corpus.codes import PROF_STANDARD_CODE, find_competency_codes

RRF_K = 60  # константа RRF: сглаживает разницу между верхними местами списков

# номер пункта только с явным «п.» / «пункт»: голые «02.03.02» и «2.3» — не пункты
_CLAUSE_NUMBER = re.compile(
    r"(?:\bп(?:ункт[а-я]*)?\.?|\bпп\.?|\bподпункт[а-я]*)\s*(\d+(?:\.\d+)+)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class QueryKeys:
    """Что в запросе можно найти точно: коды компетенций и профстандартов, номера пунктов."""

    codes: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.codes or self.numbers)


def query_keys(query: str) -> QueryKeys:
    """`'Что требует УК-1 и п. 2.3?'` -> коды `('УК-1',)`, номера `('2.3',)`."""
    # в чате коды пишут и строчными: «ук-1»
    codes = [*find_competency_codes(query.upper()), *PROF_STANDARD_CODE.findall(query)]
    numbers = [m.rstrip(".") for m in _CLAUSE_NUMBER.findall(query)]
    return QueryKeys(tuple(dict.fromkeys(codes)), tuple(dict.fromkeys(numbers)))


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> list[Hashable]:
    """Reciprocal Rank Fusion: сумма 1 / (k + место) по спискам, без сырых оценок.

    Ключ, встретившийся в списке несколько раз (части одного пункта), учитывается по лучшему
    месту. При равной сумме выше тот, кто раньше встретился в первом списке.
    """
    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    for ranking in rankings:
        best: dict[Hashable, int] = {}
        for place, key in enumerate(ranking, start=1):
            best.setdefault(key, place)
        for key, place in best.items():
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + place)
            first_seen.setdefault(key, len(first_seen))
    return sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
