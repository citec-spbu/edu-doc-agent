"""Каталог правил проверки: модели и загрузка из `rules/` (ADR-0003, «Каталог правил»).

Файлы разделены по источнику требований:

- `fgos/<код направления>.yaml` — ФГОС направления;
- `spbu.yaml` — нормы СПбГУ и профстандарты;
- `cross_check.yaml` — сверка документов между собой, без цитаты норматива.

Каждый файл — список правил. Модель проверяет, что правило заполнено под свой способ
проверки и набор; уникальность ID и соответствие файла набору проверяет загрузка.
"""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

RuleSet = Literal["fgos", "spbu", "cross_check"]

# id фрагмента корпуса нормативов (ADR-0005): `<doc_id>:<номер>[:<код>]`, например
# `fgos-09.03.03:1.9` или `fgos-09.03.03:3.2:УК-1`
_CLAUSE_ID = re.compile(r"^[\w.-]+:\S+$")


class CatalogError(ValueError):
    """Ошибка в файле каталога правил: путь к файлу и ID правила в тексте."""


class Requirement(BaseModel):
    """Пункт норматива, из которого взято правило."""

    model_config = ConfigDict(extra="forbid")

    quote: str = Field(min_length=1)  # дословно, как в корпусе нормативов
    clause_id: str
    edition: str = Field(min_length=1)  # строка редакции из корпуса

    @model_validator(mode="after")
    def _clause_id_format(self) -> Self:
        if not _CLAUSE_ID.match(self.clause_id):
            raise ValueError(
                f"clause_id {self.clause_id!r}: нужен id фрагмента корпуса, "
                "например «fgos-09.03.03:1.9»"
            )
        return self


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    rule_set: RuleSet
    applies_to: Literal["course", "program", "documents"]
    doc_kinds: list[str] = Field(min_length=1)
    method: Literal["code", "llm"]
    check: str | None = None  # функция из validation/checks/, для method: code
    params: dict[str, Any] = {}
    criterion: str | None = None  # для method: llm
    fields: list[str] = []  # для method: llm
    severity: Literal["error", "warning"]
    fixable: bool
    requirement: Requirement | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.method == "code" and not self.check:
            raise ValueError("для method: code нужно поле check")
        if self.method == "llm" and not (self.criterion and self.fields):
            raise ValueError("для method: llm нужны поля criterion и fields")
        if self.rule_set == "cross_check" and self.requirement is not None:
            raise ValueError("у правил cross_check нет цитаты норматива (requirement)")
        if self.rule_set != "cross_check" and self.requirement is None:
            raise ValueError(f"для набора {self.rule_set} нужно поле requirement")
        return self


def load_rule_file(path: Path) -> list[Rule]:
    """Правила одного файла. Пустой файл — пустой список."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CatalogError(f"{path}: некорректный YAML: {error}") from None
    if data is None:
        return []
    if not isinstance(data, list):
        raise CatalogError(f"{path}: файл должен содержать список правил")
    rules = []
    for index, item in enumerate(data):
        rule_id = item.get("id", f"#{index}") if isinstance(item, dict) else f"#{index}"
        try:
            rules.append(Rule.model_validate(item))
        except ValidationError as error:
            raise CatalogError(f"{path}: правило {rule_id}: {error}") from None
    return rules


def _expected_rule_set(root: Path, path: Path) -> RuleSet:
    relative = path.relative_to(root)
    if relative.parent == Path("fgos"):
        return "fgos"
    if relative == Path("spbu.yaml"):
        return "spbu"
    if relative == Path("cross_check.yaml"):
        return "cross_check"
    raise CatalogError(
        f"{path}: файл вне схемы каталога (fgos/<код>.yaml, spbu.yaml, cross_check.yaml)"
    )


def _load(root: Path, paths: list[Path]) -> list[Rule]:
    rules: list[Rule] = []
    seen: dict[str, Path] = {}
    for path in paths:
        expected = _expected_rule_set(root, path)
        for rule in load_rule_file(path):
            if rule.rule_set != expected:
                raise CatalogError(
                    f"{path}: правило {rule.id}: rule_set {rule.rule_set}, а файл для {expected}"
                )
            if rule.id in seen:
                raise CatalogError(f"{path}: правило {rule.id} уже есть в {seen[rule.id]}")
            seen[rule.id] = path
            rules.append(rule)
    return rules


def load_catalog(root: Path) -> list[Rule]:
    """Все правила каталога: для проверки каталога целиком (например, в CI)."""
    return _load(root, sorted(root.rglob("*.yaml")))


def load_rules(root: Path, direction_codes: Iterable[str] = ()) -> list[Rule]:
    """Правила для программы: нормы СПбГУ, сверка документов и ФГОС направлений.

    Без кода направления правила ФГОС не загружаются: их результат `not_checked` выставляет
    граф. Отсутствующий файл означает, что правил этого набора пока нет.
    """
    paths = [root / "spbu.yaml", root / "cross_check.yaml"]
    paths += [root / "fgos" / f"{code}.yaml" for code in dict.fromkeys(direction_codes)]
    return _load(root, [p for p in paths if p.exists()])
