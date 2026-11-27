"""Функции проверки для правил `method: code` (ADR-0003, «Каталог правил»).

Функция получает единицу проверки (`validation/units.py`) и параметры правила и возвращает
`CheckOutcome`.
Конкретные числа и списки функция берёт из параметров. Если нужных данных в документе
нет, результат `not_checked` с причиной.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from edu_doc_agent.extraction.source import SourceRef

CheckStatus = Literal["passed", "violation", "not_checked"]


@dataclass(frozen=True)
class CheckOutcome:
    status: CheckStatus
    message: str = ""
    sources: tuple[SourceRef, ...] = ()


def passed() -> CheckOutcome:
    return CheckOutcome("passed")


def violation(message: str, sources: list[SourceRef]) -> CheckOutcome:
    return CheckOutcome("violation", message, tuple(sources))


def not_checked(reason: str) -> CheckOutcome:
    return CheckOutcome("not_checked", reason)


CheckFunction = Callable[..., CheckOutcome]
CHECKS: dict[str, CheckFunction] = {}
# поля данных, которые читает функция: по ним «Генерация» находит правила, затронутые правкой
CHECK_FIELDS: dict[str, tuple[str, ...]] = {}


def register(name: str, fields: tuple[str, ...] = ()) -> Callable[[CheckFunction], CheckFunction]:
    def decorator(function: CheckFunction) -> CheckFunction:
        CHECKS[name] = function
        CHECK_FIELDS[name] = fields
        return function

    return decorator


def run_check(name: str, unit_data: Any, params: dict[str, Any]) -> CheckOutcome:
    function = CHECKS.get(name)
    if function is None:
        return not_checked(f"нет функции проверки {name}")
    return function(unit_data, **params)


# регистрация функций
from edu_doc_agent.validation.checks import (  # noqa: E402, F401
    competences,
    coverage,
    documents,
    practice,
    program,
    structure,
    workload,
)
