"""Каталог правил: модель правила и загрузка файлов rules/."""

from pathlib import Path

import pytest

from edu_doc_agent.validation.catalog import (
    CatalogError,
    Rule,
    load_catalog,
    load_rule_file,
    load_rules,
)

FGOS_RULE = """
- id: fgos-09.03.03-total-credits
  title: Объём программы 240 з.е.
  rule_set: fgos
  applies_to: program
  doc_kinds: [curriculum]
  method: code
  check: equals
  params: {field: total_credits, expected: 240}
  severity: error
  fixable: false
  requirement:
    quote: "Объем программы бакалавриата составляет 240 зачетных единиц"
    clause_id: "fgos-09.03.03:1.9"
    edition: "808 ред. от 08.02.2021"
"""

SPBU_RULES = """
- id: spbu-rpd-hours-per-credit
  title: Трудоёмкость в часах соответствует зачётным единицам
  rule_set: spbu
  applies_to: course
  doc_kinds: [discipline]
  method: code
  check: hours_match_credits
  params: {hours_per_credit: 36}
  severity: error
  fixable: true
  requirement:
    quote: "1 з.е. соответствует 36 академическим часам"
    clause_id: "spbu-os:2.8.1"
    edition: "11144/1 от 24.08.2026"

- id: spbu-rpd-outcomes-match-indicators
  title: Результаты обучения раскрывают индикаторы компетенций
  rule_set: spbu
  applies_to: course
  doc_kinds: [discipline]
  method: llm
  criterion: "Каждый результат обучения конкретизирует индикатор, к которому он отнесён."
  fields: [indicators, learning_outcomes]
  severity: warning
  fixable: true
  requirement:
    quote: "Результаты обучения соотносятся с индикаторами"
    clause_id: "spbu-os:3.5"
    edition: "11144/1 от 24.08.2026"
"""

CROSS_RULE = """
- id: cross-check-credits-plan-vs-rpd
  title: Трудоёмкость дисциплины совпадает в учебном плане и РПД
  rule_set: cross_check
  applies_to: documents
  doc_kinds: [discipline]
  method: code
  check: values_match_across_documents
  params: {field: credits}
  severity: error
  fixable: true
"""


def _catalog(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _rule(**overrides) -> dict:
    rule = {
        "id": "r",
        "title": "t",
        "rule_set": "cross_check",
        "applies_to": "course",
        "doc_kinds": ["discipline"],
        "method": "code",
        "check": "equals",
        "severity": "error",
        "fixable": False,
    }
    return rule | overrides


def test_load_rules_for_program(tmp_path: Path) -> None:
    root = _catalog(
        tmp_path,
        {
            "fgos/09.03.03.yaml": FGOS_RULE,
            "spbu.yaml": SPBU_RULES,
            "cross_check.yaml": CROSS_RULE,
        },
    )

    rules = load_rules(root, ["09.03.03"])

    assert [r.id for r in rules] == [
        "spbu-rpd-hours-per-credit",
        "spbu-rpd-outcomes-match-indicators",
        "cross-check-credits-plan-vs-rpd",
        "fgos-09.03.03-total-credits",
    ]
    fgos = rules[-1]
    assert fgos.params == {"field": "total_credits", "expected": 240}
    assert fgos.requirement is not None
    assert fgos.requirement.clause_id == "fgos-09.03.03:1.9"


def test_without_direction_code_fgos_rules_are_skipped(tmp_path: Path) -> None:
    root = _catalog(tmp_path, {"fgos/09.03.03.yaml": FGOS_RULE, "spbu.yaml": SPBU_RULES})

    assert {r.rule_set for r in load_rules(root)} == {"spbu"}


def test_missing_files_mean_no_rules(tmp_path: Path) -> None:
    assert load_rules(tmp_path, ["01.03.02"]) == []


def test_empty_file(tmp_path: Path) -> None:
    root = _catalog(tmp_path, {"spbu.yaml": ""})

    assert load_rule_file(root / "spbu.yaml") == []


def test_load_catalog_reads_all_files(tmp_path: Path) -> None:
    root = _catalog(
        tmp_path,
        {"fgos/09.03.03.yaml": FGOS_RULE, "fgos/02.03.02.yaml": "", "spbu.yaml": SPBU_RULES},
    )

    assert len(load_catalog(root)) == 3


def test_duplicate_id_across_files(tmp_path: Path) -> None:
    root = _catalog(
        tmp_path,
        {"fgos/09.03.03.yaml": FGOS_RULE, "fgos/02.03.02.yaml": FGOS_RULE},
    )

    with pytest.raises(CatalogError, match="fgos-09.03.03-total-credits уже есть"):
        load_catalog(root)


def test_rule_set_must_match_file(tmp_path: Path) -> None:
    root = _catalog(tmp_path, {"spbu.yaml": FGOS_RULE})

    with pytest.raises(CatalogError, match="rule_set fgos, а файл для spbu"):
        load_rules(root)


def test_file_outside_catalog_layout(tmp_path: Path) -> None:
    root = _catalog(tmp_path, {"other.yaml": CROSS_RULE})

    with pytest.raises(CatalogError, match="вне схемы каталога"):
        load_catalog(root)


def test_error_names_file_and_rule(tmp_path: Path) -> None:
    root = _catalog(tmp_path, {"cross_check.yaml": "- id: broken\n  title: t\n"})

    with pytest.raises(CatalogError, match=r"cross_check.yaml: правило broken"):
        load_rules(root)


def test_not_a_list(tmp_path: Path) -> None:
    root = _catalog(tmp_path, {"spbu.yaml": "id: x\n"})

    with pytest.raises(CatalogError, match="список правил"):
        load_rules(root)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"check": None}, "нужно поле check", id="code-without-check"),
        pytest.param(
            {"method": "llm", "check": None, "criterion": "c"},
            "criterion и fields",
            id="llm-without-fields",
        ),
        pytest.param(
            {"requirement": {"quote": "q", "clause_id": "spbu-os:1.1", "edition": "e"}},
            "нет цитаты норматива",
            id="cross-check-with-requirement",
        ),
        pytest.param({"rule_set": "spbu"}, "нужно поле requirement", id="spbu-without-requirement"),
        pytest.param(
            {
                "rule_set": "fgos",
                "requirement": {"quote": "q", "clause_id": "1.8", "edition": "e"},
            },
            "id фрагмента корпуса",
            id="bare-clause-number",
        ),
        pytest.param({"unknown": 1}, "Extra inputs", id="unknown-field"),
    ],
)
def test_invalid_rule(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Rule.model_validate(_rule(**overrides))
