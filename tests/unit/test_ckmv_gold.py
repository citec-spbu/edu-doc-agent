"""Эталон ЦКМВ: сам эталон, чувствительность метрики и замер на реальном документе."""

import copy
from pathlib import Path

import pytest

from edu_doc_agent.eval.ckmv_gold import (
    CKMV_GOLD_FULL,
    COMPETENCE_FIELDS,
    INDICATOR_FIELDS,
    OUTCOME_FIELDS,
    THRESHOLD,
    load_ckmv_gold,
    score_ckmv,
)
from edu_doc_agent.extraction.ckmv import (
    CkmvCompetence,
    CkmvDocument,
    CkmvIndicator,
    CkmvOutcome,
    extract_ckmv,
)
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.source import SourceRef
from tests.real_data import real_ckmv_markdown

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MANUAL = load_ckmv_gold(_FIXTURES / "ckmv_gold.json")
FULL = load_ckmv_gold(_FIXTURES / "ckmv_gold_full.json")
REAL_MD = real_ckmv_markdown()
needs_data = pytest.mark.skipif(
    REAL_MD is None or not CKMV_GOLD_FULL.exists(), reason="нет data/ЦКМВ.zip"
)


def _doc_from(gold: dict) -> CkmvDocument:
    """Результат «идеального» экстрактора: ровно то, что записано в эталоне."""
    ref = SourceRef(file="gold")
    doc = CkmvDocument(file="gold")
    for g in gold["competences"]:
        comp = CkmvCompetence(
            code=g["code"],
            name=g["name"],
            group=g["group"],
            role_section=g["role_section"],
            role=g["role"],
            source=ref,
        )
        for gi in g["indicators"]:
            ind = CkmvIndicator(code=gi["code"], name=gi["name"], level=gi["level"], source=ref)
            for go in gi["outcomes"]:
                ind.outcomes.append(
                    CkmvOutcome(
                        code=go["code"],
                        letter=go["letter"],
                        number=go["number"],
                        kind=go["kind"],
                        text=go["text"],
                        source=ref,
                    )
                )
            comp.indicators.append(ind)
        doc.competences.append(comp)
    return doc


# --- сам эталон ------------------------------------------------------------------------------


def test_full_gold_totals_and_shape() -> None:
    comps = FULL["competences"]
    assert len(comps) == FULL["totals"]["competences"] == 4
    assert sum(len(c["indicators"]) for c in comps) == FULL["totals"]["indicators"] == 8
    outs = [o for c in comps for i in c["indicators"] for o in i["outcomes"]]
    assert len(outs) == FULL["totals"]["outcomes"] == 24
    assert {o["letter"] for o in outs} == {"З", "У", "В"}
    assert FULL["complete"] and not MANUAL["complete"]


def test_gold_covers_roles_groups_and_duplicate_codes() -> None:
    comps = FULL["competences"]
    assert {c["role_section"] for c in comps} >= {
        "Основная профессиональная роль",
        "Профессиональная опережающая роль",
    }
    assert len({c["group"] for c in comps}) == 4
    # один код в двух группах (базовый и топ-уровень): сопоставление идёт по порядку
    codes = [c["code"] for c in comps]
    assert codes.count("ПКП-9-ППК-ДА2") == 2
    assert {i["level"] for c in comps for i in c["indicators"]} == {
        "Базовый",
        "Продвинутый",
    }


def test_synthetic_subsets_agree() -> None:
    full = {(c["code"], c["line"]): c for c in FULL["competences"]}
    for c in MANUAL["competences"]:
        e = full[(c["code"], c["line"])]
        assert {k: c[k] for k in ("name", "group", "role_section", "role")} == {
            k: e[k] for k in ("name", "group", "role_section", "role")
        }
        assert len(c["indicators"]) == len(e["indicators"])
        for i, j in zip(c["indicators"], e["indicators"], strict=True):
            assert (i["code"], i["name"], i["level"]) == (j["code"], j["name"], j["level"])
            assert len(i["outcomes"]) == len(j["outcomes"])
            for a, b in zip(i["outcomes"], j["outcomes"], strict=True):
                assert {k: a[k] for k in ("code", "letter", "number", "kind")} == {
                    k: b[k] for k in ("code", "letter", "number", "kind")
                }
                assert normalize(a["text"]) == normalize(b["text"])


# --- чувствительность метрики ----------------------------------------------------------------


def test_perfect_result_scores_one() -> None:
    for gold in (MANUAL, FULL):
        report = score_ckmv(_doc_from(gold), gold)
        assert report.passed() and report.overall == 1.0, report.summary()


def _first_outcome(doc: CkmvDocument) -> CkmvOutcome:
    return doc.competences[1].indicators[0].outcomes[0]


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("competence.name", lambda d: setattr(d.competences[1], "name", "другое")),
        ("competence.group", lambda d: setattr(d.competences[1], "group", None)),
        ("competence.role", lambda d: setattr(d.competences[1], "role", "Другая роль")),
        ("competence.n_indicators", lambda d: d.competences[1].indicators.pop()),
        ("indicator.level", lambda d: setattr(d.competences[1].indicators[0], "level", "Средний")),
        ("indicator.name", lambda d: setattr(d.competences[1].indicators[0], "name", "иное")),
        ("indicator.n_outcomes", lambda d: d.competences[1].indicators[0].outcomes.pop()),
        (
            "outcome.text",
            lambda d: setattr(d.competences[1].indicators[0].outcomes[0], "text", "x"),
        ),
        (
            "outcome.letter",
            lambda d: setattr(d.competences[1].indicators[0].outcomes[0], "letter", "В"),
        ),
        (
            "outcome.kind",
            lambda d: setattr(d.competences[1].indicators[0].outcomes[0], "kind", "умеет"),
        ),
        (
            "outcome.number",
            lambda d: setattr(d.competences[1].indicators[0].outcomes[0], "number", 9),
        ),
    ],
)
def test_corrupting_one_field_is_detected_in_that_field(field: str, mutate) -> None:
    doc = _doc_from(MANUAL)
    mutate(doc)
    report = score_ckmv(doc, MANUAL)
    assert report.accuracy(field) < 1.0
    assert not report.passed(1.0)
    if field.endswith(("n_indicators", "n_outcomes")):
        return  # удалённая сущность закономерно становится «не найденной» и на нижнем уровне
    all_fields = (
        [f"competence.{n}" for n in COMPETENCE_FIELDS]
        + [f"indicator.{n}" for n in INDICATOR_FIELDS]
        + [f"outcome.{n}" for n in OUTCOME_FIELDS]
    )
    untouched = [f for f in all_fields if f != field and f != "competence.n_indicators"]
    assert all(report.accuracy(f) == 1.0 for f in untouched)


def test_missing_outcome_and_wrong_totals_fail() -> None:
    doc = _doc_from(FULL)
    del doc.competences[3].indicators[0].outcomes[0]
    report = score_ckmv(doc, FULL)
    assert report.missing and report.count_errors and not report.passed()


def test_duplicate_competence_codes_are_matched_in_document_order() -> None:
    doc = _doc_from(FULL)
    second = [c for c in doc.competences if c.code == "ПКП-9-ППК-ДА2"][1]
    second.group = "Другая группа"
    report = score_ckmv(doc, FULL)
    assert report.correct["competence.group"] == report.total["competence.group"] - 1


def test_threshold_matches_claude_md_target() -> None:
    assert THRESHOLD == 0.95
    assert copy.deepcopy(MANUAL) == MANUAL


# --- реальный документ -----------------------------------------------------------------------


@needs_data
def test_real_ckmv_accuracy_on_manual_gold() -> None:
    report = score_ckmv(extract_ckmv(REAL_MD or "", "ckmv.docx"), load_ckmv_gold())
    assert report.passed(), report.summary()


@needs_data
def test_real_ckmv_accuracy_on_full_gold() -> None:
    report = score_ckmv(extract_ckmv(REAL_MD or "", "ckmv.docx"), load_ckmv_gold(CKMV_GOLD_FULL))
    assert report.total["outcome.text"] == 409
    assert report.passed(), report.summary()
