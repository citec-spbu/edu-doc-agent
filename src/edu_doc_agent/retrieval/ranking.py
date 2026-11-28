"""Разбор запроса и слияние рангов для поиска по нормативам (ADR-0005, решение 3).

Без зависимостей от базы и модели: только то, что можно проверить на строках.
"""

import re
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from edu_doc_agent.retrieval.corpus.codes import (
    COMPETENCY_CODE,
    PROF_STANDARD_CODE,
    normalize_code,
)

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

# ссылки на закон: «ч. 5 ст. 13», «часть 5 статьи 13», «ст. 13 ч. 5», «п. 6 ст. 2», «ст. 13»
_ARTICLE = r"(?:ст\.|статьей|стать[яиеюй])\s*(\d+(?:\.\d+)?)"
_PART = r"(?:ч\.|частью|част[ьиейю])\s*(\d+(?:\.\d+)?)"
_ITEM = r"(?:п\.|пункт[а-я]*)\s*(\d+(?:\.\d+)?)"
_LAW_REFS: tuple[tuple[re.Pattern[str], tuple[int, ...], str], ...] = (
    # (шаблон, группы: статья, часть или пункт; вид)
    (re.compile(rf"\b{_ITEM}\s+{_PART}\s+{_ARTICLE}", re.I), (3, 2), "ч"),  # пункт части
    (re.compile(rf"\b{_PART}\s+{_ARTICLE}", re.I), (2, 1), "ч"),
    (re.compile(rf"\b{_ARTICLE}\s*,?\s*{_PART}", re.I), (1, 2), "ч"),
    (re.compile(rf"\b{_ITEM}\s+{_ARTICLE}", re.I), (2, 1), "п"),
    (re.compile(rf"\b{_ARTICLE}", re.I), (1,), ""),
)
# пункт приказа — целое число: «п. 15», «пункт 17(1)» (но не «п. 2.3» — это пункт ФГОС)
_ORDER_POINT = re.compile(
    r"(?:\bп\.|\bпункт[а-я]*)\s*(\d+(?:\(\d+\))?)(?!\d)(?!\.\d)(?!\s*\))", re.I
)
# трудовая функция профстандарта «A/01.3» и обобщённая «A.3», «ОТФ A»
_LABOR_FUNCTION = re.compile(r"(?<![\wЁё])([A-ZА-ЯЁ])\s*/\s*(\d{2}\.\d(?:\.\d)?)(?!\d)")
_GENERALIZED_FUNCTION = re.compile(
    r"(?<![\wЁё.])([A-ZА-ЯЁ])\.\d(?![\d.])|\bОТФ\s*[«\"]?([A-ZА-ЯЁ])\b"
)
_LATIN = str.maketrans("АВСЕНКМОРТХ", "ABCEHKMOPTX")
# код направления «02.03.02» — точный ключ строки перечня (приказ 1061), только в вопросе
# о перечне: иначе любой вопрос о программе 02.03.02 начинался бы со строки перечня
_DIRECTION_CODE = re.compile(r"(?<![\d.])\d{2}\.\d{2}\.\d{2}(?![\d.])")
_ABOUT_LIST = re.compile(r"перечн|1061|наименовани|квалификаци(?!и\b)|присваива", re.I)


@dataclass(frozen=True)
class QueryKeys:
    """Что в запросе можно найти точно: коды компетенций, профстандартов и трудовых функций,
    направлений в вопросе о перечне, номера пунктов ФГОС и ОС («2.3»), пунктов приказов
    («15») и частей статей («ст.13:ч.5»)."""

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
    codes = [*competencies, *_prof_standard_codes(query)]
    if _ABOUT_LIST.search(query):
        codes += _DIRECTION_CODE.findall(query)
    numbers = [
        number
        for match in _CLAUSE_NUMBERS.findall(query)
        for number in re.findall(_NUMBER_IN_QUERY, match)
        if not number.startswith("0")  # «02.03.02» в перечне — код направления, не пункт
    ]
    laws, rest = _law_numbers(query)
    numbers += laws + _ORDER_POINT.findall(rest)
    return QueryKeys(tuple(dict.fromkeys(codes)), tuple(dict.fromkeys(numbers)))


def _law_numbers(query: str) -> tuple[list[str], str]:
    """Ссылки на статьи закона -> номера фрагментов («ст.13:ч.5», «ст.2:п.6», «ст.13») и
    запрос без них (чтобы «п. 2» из «п. 2 ч. 7 ст. 13» не стал пунктом приказа)."""
    found: list[tuple[int, str]] = []
    rest = query
    for pattern, groups, kind in _LAW_REFS:
        for match in pattern.finditer(rest):
            article, *sub = (match.group(g) for g in groups)
            number = f"ст.{article}:{kind}.{sub[0]}" if sub else f"ст.{article}"
            found.append((match.start(), number))
        rest = pattern.sub(lambda m: " " * len(m.group(0)), rest)
    return [number for _, number in sorted(found)], rest


def _prof_standard_codes(query: str) -> list[str]:
    """Коды профстандартов; с кодом функции — «06.001:A/01.3», «06.001:A» вместо «06.001»."""
    standards = PROF_STANDARD_CODE.findall(query)
    functions = [
        f"{letter.translate(_LATIN)}/{number}" for letter, number in _LABOR_FUNCTION.findall(query)
    ]
    functions += [
        (dotted or named).translate(_LATIN)
        for dotted, named in _GENERALIZED_FUNCTION.findall(_LABOR_FUNCTION.sub(" ", query))
    ]
    if not standards or not functions:
        return standards
    return [f"{standard}:{function}" for standard in standards for function in functions]


def natural_key(text: str) -> tuple:
    """Ключ сортировки с числами по значению: «ст.13:ч.2» раньше «ст.13:ч.10»."""
    return tuple(int(x) if x.isdigit() else x for x in re.findall(r"\d+|\D+", text))


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
