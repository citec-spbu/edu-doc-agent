"""Разбор запроса и слияние рангов для поиска по нормативам (ADR-0005, решение 3).

Без зависимостей от базы и модели: только то, что можно проверить на строках.
"""

import re
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from edu_doc_agent.retrieval.corpus.codes import COMPETENCY_CODE, PROF_STANDARD_CODE, normalize_code

RRF_K = 60  # константа RRF: сглаживает разницу между верхними местами списков

_NUMBER_IN_QUERY = r"\d+(?:\.\d+)+"
# номер пункта только с явным «п.» / «пункт»: голые «02.03.02» и «2.3» — не пункты;
# после «п.» — один номер или перечень: «пункт 2.3 и 2.4», «пп. 2.3, 2.4»
_CLAUSE_NUMBERS = re.compile(
    rf"(?:\bп(?:ункт[а-я]*)?\.?|\bпп\.?|\bподпункт[а-я]*)\s*"
    rf"({_NUMBER_IN_QUERY}(?:\s*(?:,|\bи\b|\bили\b)\s*{_NUMBER_IN_QUERY})*)",
    re.IGNORECASE,
)
# после кода компетенции — номер индикатора: «УК-1.2»
_INDICATOR_TAIL = re.compile(r"\.\d")


@dataclass(frozen=True)
class QueryKeys:
    """Что в запросе можно найти точно: коды компетенций и профстандартов, номера пунктов."""

    codes: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.codes or self.numbers)


def query_keys(query: str) -> QueryKeys:
    """`'Что требует УК-1 и п. 2.3?'` -> коды `('УК-1',)`, номера `('2.3',)`.

    Индикатор («УК-1.2») — не код компетенции: во ФГОС индикаторов нет, и найденная
    «точно» УК-1 выдавала бы несуществующее требование за подтверждённое. Индикатор
    ищется как обычный текст.
    """
    upper = query.upper()  # в чате коды пишут и строчными: «ук-1»
    competencies = [
        normalize_code(m.group(0))
        for m in COMPETENCY_CODE.finditer(upper)
        if not _INDICATOR_TAIL.match(upper, m.end())
    ]
    codes = [*competencies, *PROF_STANDARD_CODE.findall(query)]
    numbers = [
        number
        for match in _CLAUSE_NUMBERS.findall(query)
        for number in re.findall(_NUMBER_IN_QUERY, match)
        if not number.startswith("0")  # «02.03.02» в перечне — код направления, не пункт
    ]
    return QueryKeys(tuple(dict.fromkeys(codes)), tuple(dict.fromkeys(numbers)))


def rrf(
    rankings: Sequence[Sequence[Hashable]],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[Hashable]:
    """Взвешенный Reciprocal Rank Fusion: сумма вес / (k + место) по спискам.

    Сырые оценки списков несравнимы, поэтому важны только места. Ключ, встретившийся в
    списке несколько раз (части одного пункта), учитывается по лучшему месту. При равной
    сумме выше тот, кто раньше встретился в первом списке. Вес 0 — список не учитывается.
    """
    weights = weights if weights is not None else [1.0] * len(rankings)
    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        best: dict[Hashable, int] = {}
        for place, key in enumerate(ranking, start=1):
            best.setdefault(key, place)
        for key, place in best.items():
            if weight > 0:
                scores[key] = scores.get(key, 0.0) + weight / (k + place)
                first_seen.setdefault(key, len(first_seen))
    return sorted(scores, key=lambda key: (-scores[key], first_seen[key]))


# --- перечни: вводный пункт «…, содержащие:» и его подпункты возвращаются одним результатом

_NUMBER = re.compile(r"\d+(?:\.\d+)+")


def clause_of(fragment_id: str) -> tuple[str, str] | None:
    """`'spbu-os:2.2.4.1'` -> `('spbu-os', '2.2.4.1')`; компетенция, профстандарт -> None."""
    doc_id, _, number = fragment_id.partition(":")
    return (doc_id, number) if _NUMBER.fullmatch(number) else None


def number_order(number: str) -> tuple[int, ...]:
    """Порядок пунктов в документе: 2.2.10 после 2.2.9."""
    return tuple(int(n) for n in number.split("."))


def collapse_lists(
    keys: Sequence[tuple[str, str]], list_heads: set[tuple[str, str, str]]
) -> list[tuple[tuple[str, str], bool]]:
    """Подпункты перечня -> его вводный пункт; вернуть (ключ, раскрыть ли перечень).

    `keys` — (id, edition) по рангу; `list_heads` — (doc_id, edition, номер) пунктов,
    которые вводят перечень (кончаются двоеточием). Найденный подпункт заменяется ближайшим
    таким предком (или самим собой, если он вводит перечень). Если в выдаче оказались и
    перечень, и вложенный в него перечень, остаётся внешний — на месте лучшего из них.
    """
    out: list[tuple[tuple[str, str], bool]] = []
    heads_out: dict[tuple[str, str], str] = {}  # ключ перечня в выдаче -> его номер
    for key in keys:
        fragment_id, edition = key
        clause = clause_of(fragment_id)
        head = None
        if clause is not None:
            doc_id, number = clause
            parts = number.split(".")
            for size in range(len(parts), 1, -1):  # сам пункт, затем предки снизу вверх
                candidate = ".".join(parts[:size])
                if (doc_id, edition, candidate) in list_heads:
                    head = candidate
                    break
        if head is None:
            if all(k != key for k, _ in out):
                out.append((key, False))
            continue
        head_key = (f"{doc_id}:{head}", edition)
        if any(
            k[1] == edition
            and k[0].split(":", 1)[0] == doc_id
            and (head == n or head.startswith(n + "."))
            for k, n in heads_out.items()
        ):
            continue  # уже есть этот перечень или объемлющий его
        nested = [
            k
            for k, n in heads_out.items()
            if k[1] == edition and k[0].split(":", 1)[0] == doc_id and n.startswith(head + ".")
        ]
        if nested:  # внешний перечень встаёт на место лучшего из вложенных
            place = min(i for i, (k, _) in enumerate(out) if k in nested)
            out = [item for item in out if item[0] not in nested]
            out.insert(place, (head_key, True))
            for k in nested:
                del heads_out[k]
        else:
            out.append((head_key, True))
        heads_out[head_key] = head
    return out
