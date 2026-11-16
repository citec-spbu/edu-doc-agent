"""Эталон учебного плана: сам эталон, чувствительность метрики и замер на реальном плане."""

import re
from pathlib import Path

import pytest

from edu_doc_agent.eval.plan_gold import (
    ENTRY_FIELDS,
    PLAN_GOLD_FULL,
    THRESHOLD,
    load_plan_gold,
    score_plan,
)
from edu_doc_agent.extraction.curriculum import CurriculumPlan, extract_plan
from edu_doc_agent.extraction.llm_extract import normalize
from tests.real_data import real_plan_markdown

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOLD = load_plan_gold(_FIXTURES / "plan_gold.json")
FULL = load_plan_gold(_FIXTURES / "plan_gold_full.json")
REAL_MD = real_plan_markdown()
needs_data = pytest.mark.skipif(
    REAL_MD is None or not PLAN_GOLD_FULL.exists(), reason="нет data/Учебный план.zip"
)


def _gold_plan() -> CurriculumPlan:
    """Результат «идеального» экстрактора: ровно то, что записано в эталоне."""
    from edu_doc_agent.extraction.curriculum import (
        FinalAttestation,
        PlanCompetence,
        PlanEntry,
    )
    from edu_doc_agent.extraction.source import SourceRef

    ref = SourceRef(file="gold")
    keys = ("contact_hours", "self_study_hours")
    plan = CurriculumPlan(file="gold", **GOLD["title"])
    plan.competences = [PlanCompetence(source=ref, **c) for c in GOLD["competences"]]
    for g in GOLD["entries"]:
        contact_names = [f"c{i}" for i in range(len(g["contact"]))]
        self_names = [f"s{i}" for i in range(len(g["self"]))]
        fields = {n: g[n] for n in ("block", "code", "name", "name_en", "credits", "semester")}
        fields |= {n: g[n] for n in ("year", "part", "group", "profile", "trajectory")}
        plan.entries.append(
            PlanEntry(
                competences=g["competences"],
                control=g["control"],
                active_hours=g["active"],
                **{keys[0]: dict(zip(contact_names, g["contact"], strict=True))},
                **{keys[1]: dict(zip(self_names, g["self"], strict=True))},
                source=ref,
                **fields,
            )
        )
    plan.final_attestation = [FinalAttestation(source=ref, **a) for a in GOLD["final_attestation"]]
    # При неполном эталоне дополняем результат до ожидаемых счётчиков.
    plan.entries += [plan.entries[0]] * (GOLD["totals"]["entries"] - len(plan.entries))
    plan.competences += [plan.competences[0]] * (
        GOLD["totals"]["competences"] - len(plan.competences)
    )
    return plan


# --- сам эталон ------------------------------------------------------------------------------


def test_gold_entries_have_unique_keys_and_valid_shape() -> None:
    keys = [(g["code"], normalize(g["name"]), g["semester"]) for g in GOLD["entries"]]
    assert len(keys) == len(set(keys)), "ключ (код, название, семестр) должен быть уникален"
    for g in GOLD["entries"]:
        assert len(g["contact"]) == 9 and len(g["self"]) == 5, g["line"]
        assert re.fullmatch(r"\d{6}", g["code"])
        assert g["year"] == (g["semester"] + 1) // 2


def test_gold_covers_the_hard_cases() -> None:
    entries = GOLD["entries"]
    assert any(g["trajectory"] for g in entries), "траектории обучения"
    assert {g["profile"] for g in entries} >= {None, "Профиль СТАНДАРТ", "Профиль ТОП ИТ ПРО"}
    assert any(g["part"] == "Факультативные занятия" for g in entries)
    assert any(g["block"] is None for g in entries) and any(g["block"] == "-" for g in entries)
    assert any(g["block"] == "Блок.2..прки" for g in entries)
    assert {g["semester"] for g in entries} == set(range(1, 5))


# --- чувствительность метрики ----------------------------------------------------------------


def test_perfect_result_scores_one() -> None:
    report = score_plan(_gold_plan(), GOLD)
    assert report.passed() and report.overall == 1.0 and report.hours_cell_accuracy == 1.0


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("credits", lambda e: setattr(e, "credits", e.credits + 1)),
        ("profile", lambda e: setattr(e, "profile", None)),
        ("trajectory", lambda e: setattr(e, "trajectory", "Траектория 9")),
        ("competences", lambda e: e.competences.pop()),
        ("control", lambda e: setattr(e, "control", ["зачёт"])),
        ("active_hours", lambda e: setattr(e, "active_hours", 0.0)),
        ("part", lambda e: setattr(e, "part", None)),
    ],
)
def test_corrupting_one_field_is_detected_in_that_field(field: str, mutate) -> None:
    plan = _gold_plan()
    # портим одну запись, у которой поле заполнено (иначе порча ничего не меняет)
    target = next(e for e in plan.entries if getattr(e, field))
    mutate(target)
    report = score_plan(plan, GOLD)
    assert report.accuracy(field) < 1.0
    assert not report.passed(1.0)
    assert all(
        report.accuracy(f) == 1.0 for f in ENTRY_FIELDS if f not in (field, "semester", "year")
    )


def test_one_wrong_hour_cell_is_detected() -> None:
    plan = _gold_plan()
    first = next(iter(plan.entries[2].contact_hours))
    plan.entries[2].contact_hours[first] += 2
    report = score_plan(plan, GOLD)
    assert report.hour_cells_ok == report.hour_cells - 1
    assert report.accuracy("contact_hours") < 1.0


def test_missing_entry_and_wrong_counts_fail() -> None:
    plan = _gold_plan()
    del plan.entries[3]
    report = score_plan(plan, GOLD)
    assert report.missing and report.count_errors and not report.passed()


def test_threshold_matches_claude_md_target() -> None:
    assert THRESHOLD == 0.95


# --- реальный план ---------------------------------------------------------------------------


@needs_data
def test_independent_row_counts_in_raw_markdown() -> None:
    """Без экстрактора: строки с кодом `[NNNNNN]` и строки таблицы 1.1 прямо в сыром Markdown."""
    text = REAL_MD or ""
    coded = [
        ln for ln in text.splitlines() if ln.startswith("|") and re.search(r"\|\s*\[\d{6}\]", ln)
    ]
    assert len(coded) == 226
    section = text.split("# Раздел 2")[0].split("## 1.1")[1]
    codes = [
        ln for ln in section.splitlines() if re.match(r"\|\s*[А-ЯЁ]+(?:-[А-ЯЁ0-9]+)+\s*\|", ln)
    ]
    assert len(codes) == 49


@needs_data
def test_real_plan_accuracy_on_gold() -> None:
    plan = extract_plan(REAL_MD or "", "plan.docx")
    report = score_plan(plan, load_plan_gold())
    assert report.passed(), report.summary()


@needs_data
def test_real_plan_accuracy_on_full_gold() -> None:
    """Эталон на все 226 строк и 49 компетенций."""
    report = score_plan(extract_plan(REAL_MD or "", "plan.docx"), load_plan_gold(PLAN_GOLD_FULL))
    assert report.total["code"] == 226
    assert report.passed(), report.summary()


def test_synthetic_subsets_agree() -> None:
    """Синтетические записи обоих наборов согласованы по каждому полю."""
    full = {e["line"]: e for e in FULL["entries"]}
    for g in GOLD["entries"]:
        e = full[g["line"]]
        for key in g:
            if key in ("name", "name_en"):
                assert normalize(g[key] or "") == normalize(e[key] or ""), (g["line"], key)
            else:
                assert g[key] == e[key], (g["line"], key)


def test_full_gold_shape() -> None:
    assert len(FULL["entries"]) == FULL["totals"]["entries"] == 4
    assert len(FULL["competences"]) == FULL["totals"]["competences"] == 1
    assert all(len(g["contact"]) == 9 and len(g["self"]) == 5 for g in FULL["entries"])
    assert {g["semester"] for g in FULL["entries"]} == set(range(1, 5))
