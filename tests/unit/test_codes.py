"""Сопоставление кодов между документами: каноническая форма, индекс, реальные документы."""

# ruff: noqa: E501  (строки таблиц Markdown)

import difflib

import pytest

from edu_doc_agent.extraction.ckmv import extract_ckmv
from edu_doc_agent.extraction.codes import (
    build_index,
    canonical_code,
    check_codes,
    check_competence_codes,
    code_dialect,
)
from edu_doc_agent.extraction.curriculum import extract_plan
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.rpd import extract_rpd_file
from tests.real_data import REAL_MD, real_ckmv_markdown, real_plan_markdown

# (как записано, канонический вид): таблица составлена вручную по реальным кодам документов
CASES = [
    # нумерованный диалект ЦКМВ и матрицы → канонический
    ("ПКП-2-ППК-Р2", "ПКП-ППК-Р2"),
    ("ПКП-8-ППК-ДА1", "ПКП-ППК-ДА1"),
    ("ПКП-10-ППК-ДА3", "ПКП-ППК-ДА3"),
    ("ПКП-15-ППК-ОС2", "ПКП-ППК-ОС2"),
    ("ПКП-21-ППК-У2", "ПКП-ППК-У2"),
    ("ПКП-19-МО-1", "ПКП-МО-1"),
    ("ПКП-20-МО-2", "ПКП-МО-2"),
    ("ПКП-22-РГ-2", "ПКП-РГ-2"),
    ("ПКП-1-ППК-Р1", "ПКП-ППК-Р1"),  # матрица; в ЦКМВ тот же код без номера
    # канонические не меняются
    ("ПКП-ППК-Р1", "ПКП-ППК-Р1"),
    ("ПКП-МО-1", "ПКП-МО-1"),
    ("УК-1", "УК-1"),
    ("УКБ-4", "УКБ-4"),
    ("ОПК-6", "ОПК-6"),
    ("ПКА-2", "ПКА-2"),
    # УНК: нет префикса ПКП- и нули
    ("УНК-01", "ПКП-УНК-1"),
    ("ПКП-УНК-01", "ПКП-УНК-1"),
    ("ПКП-УНК-3", "ПКП-УНК-3"),
    ("ПКП-УНК-03", "ПКП-УНК-3"),
    # индикаторы
    ("ПКП-2-ППК-Р2.3", "ПКП-ППК-Р2.3"),
    ("ПКП-2-ППК-Р2.3.", "ПКП-ППК-Р2.3"),
    ("УНК-01.01", "ПКП-УНК-1.1"),
    ("ПКП-УНК-03.01", "ПКП-УНК-3.1"),
    ("ПКП-МО-2.1.", "ПКП-МО-2.1"),
    # результаты обучения, в том числе с опечаткой «3-1» и разной пунктуацией
    ("ПКП-МО-1.1 З-1", "ПКП-МО-1.1 З-1"),
    ("ПКП-19-МО-1.2 У-3", "ПКП-МО-1.2 У-3"),
    ("ПКП-2-ППК-Р2.1 3-1", "ПКП-ППК-Р2.1 З-1"),
    ("УНК-01.01 У-1.", "ПКП-УНК-1.1 У-1"),
    # тире и пробелы
    ("УК–1", "УК-1"),
    ("  ПКП–2–ППК–Р2  ", "ПКП-ППК-Р2"),
    ("пкп-2-ппк-р2", "ПКП-ППК-Р2"),
]


@pytest.mark.parametrize(("raw", "expected"), CASES)
def test_canonical_code(raw: str, expected: str) -> None:
    assert canonical_code(raw) == expected


def test_canonical_code_is_idempotent_and_handles_empty() -> None:
    for raw, _ in CASES:
        once = canonical_code(raw)
        assert canonical_code(once) == once
    assert canonical_code("") == "" and canonical_code("   ") == ""


def test_distinct_competences_do_not_collide() -> None:
    codes = ["ПКП-ППК-Р1", "ПКП-ППК-Р2", "ПКП-ППК-ДА1", "ПКП-МО-1", "ПКП-МО-2", "ПКП-РГ-2", "УК-1"]
    assert len({canonical_code(c) for c in codes}) == len(codes)
    # сквозной номер не отличает компетенции: ПКП-19-МО-1 и ПКП-МО-1 — одна и та же
    assert canonical_code("ПКП-19-МО-1") == canonical_code("ПКП-МО-1")


def test_code_dialect() -> None:
    assert code_dialect("ПКП-2-ППК-Р2") == "numbered" and code_dialect("УНК-01") == "numbered"
    assert code_dialect("ПКП-ППК-Р2") == "canonical" and code_dialect("УК-1") == "canonical"
    assert code_dialect("ПКП-УНК-01") == "canonical" and code_dialect("") == "canonical"


# --- индекс ----------------------------------------------------------------------------------


def test_index_links_spellings_and_reports_unknown_codes() -> None:
    index = build_index(
        {
            "план": [("ПКП-ППК-Р2", "Способен проверять код"), ("УК-1", "Способен искать")],
            "ЦКМВ": [("ПКП-2-ППК-Р2", "Способен проверять код"), ("ПКП-9-ППК-Х9", "Другое")],
        }
    )
    entry = index.entries["ПКП-ППК-Р2"]
    assert entry.sources == {"план", "ЦКМВ"}
    assert entry.spellings["ЦКМВ"] == ["ПКП-2-ППК-Р2"]
    assert index.unknown_in("ЦКМВ", "план") == ["ПКП-ППК-Х9"]
    assert index.codes_of("план") == {"ПКП-ППК-Р2", "УК-1"}
    warnings = check_codes(index, reference="план", subsets=["ЦКМВ"])
    assert len(warnings) == 1 and "ПКП-ППК-Х9" in warnings[0]


def test_name_differences_are_deduplicated_and_ignore_case_dash_and_dot() -> None:
    index = build_index(
        {
            "план": [("ПКП-МО-1", "Способен разрабатывает модели")],
            "РПД": [
                ("ПКП-МО-1", "способен разрабатывает модели."),  # только регистр и точка
                ("ПКП-МО-1", "Способен разрабатывать модели"),
                ("ПКП-МО-1", "Способен разрабатывать модели"),  # такой же из другого файла
            ],
        }
    )
    diffs = index.name_differences("план")
    assert len(diffs) == 1 and diffs[0].source == "РПД" and 0.9 < diffs[0].similarity < 1
    assert "сходство" in check_codes(index, reference="план", subsets=[])[0]


def test_name_differences_need_a_reference_name() -> None:
    index = build_index({"ЦКМВ": [("УНК-01", "А")], "РПД": [("ПКП-УНК-01", "Б")]})
    assert index.name_differences("план") == []


# --- реальные документы ----------------------------------------------------------------------

needs_data = pytest.mark.skipif(
    real_plan_markdown() is None or real_ckmv_markdown() is None or not REAL_MD,
    reason="нет data/ (план, ЦКМВ, РПД)",
)


def _plan_and_ckmv():
    return (
        extract_plan(real_plan_markdown() or "", "plan.docx"),
        extract_ckmv(real_ckmv_markdown() or "", "ckmv.docx"),
    )


@needs_data
def test_real_plan_codes_are_unique_and_ckmv_covers_all_professional_ones() -> None:
    plan, ckmv = _plan_and_ckmv()
    plan_codes = [canonical_code(c.code) for c in plan.competences]
    assert len(set(plan_codes)) == len(plan_codes) == 49
    index = build_index(
        {
            "план": [(c.code, c.name) for c in plan.competences],
            "ЦКМВ": [(c.code, c.name) for c in ckmv.competences],
        }
    )
    assert index.unknown_in("ЦКМВ", "план") == []
    professional = {c for c in plan_codes if c.startswith("ПКП")}
    assert professional == index.codes_of("ЦКМВ") and len(professional) == 27


@needs_data
def test_real_mapping_agrees_with_name_oracle() -> None:
    """Независимая проверка: код сопоставлен с тем же кодом плана, что и по тексту названия."""
    plan, ckmv = _plan_and_ckmv()
    plan_names = [(c.code, normalize(c.name)) for c in plan.competences]

    def by_name(name: str) -> str | None:
        n = normalize(name)
        best = max(plan_names, key=lambda p: difflib.SequenceMatcher(None, n, p[1]).ratio())
        return best[0] if difflib.SequenceMatcher(None, n, best[1]).ratio() >= 0.9 else None

    checked = 0
    for comp in ckmv.competences:
        expected = by_name(comp.name)
        assert expected is not None, comp.code
        assert canonical_code(comp.code) == canonical_code(expected), comp.code
        checked += 1
    assert checked == 31


@needs_data
def test_real_rpd_codes_map_into_plan_and_name_checks_flag_known_issues() -> None:
    plan, ckmv = _plan_and_ckmv()
    rpd = [extract_rpd_file(p) for p in REAL_MD]
    warnings = check_competence_codes(plan, ckmv, rpd)
    assert not any("коды, которых нет" in w for w in warnings)
    # опечатка плана «разрабатывает» вместо «разрабатывать» видна и в ЦКМВ, и в РПД
    typo = [w for w in warnings if w.startswith("ПКП-МО-1:") and "разрабатывает" in w]
    assert {"ЦКМВ" in w for w in typo} == {True, False}  # есть предупреждения по обоим документам
    assert all(float(w.split("сходство ")[1][:4]) >= 0.98 for w in typo if "ЦКМВ" in w)


def test_check_competence_codes_on_synthetic_documents() -> None:
    plan = extract_plan(
        "| Код компетенции | Наименование и (или) описание компетенции |\n| --- | --- |\n"
        "| ПКП-ППК-Р2 | Способен проверять код |\n| УК-1 | Способен искать |\n",
        "plan.md",
    )
    head = "| Роль: | Группа | Индикаторы группы | Результаты обучения группы |\n| --- | --- | --- | --- |\n"
    ckmv = extract_ckmv(
        head + "| Р | ПКП-2-ППК-Р2. Способен проверять код | ПКП-2-ППК-Р2.1. Тесты (Базовый) | "
        "ПКП-2-ППК-Р2.1 З-1. Знает тесты. |\n"
        "| | ПКП-9-ППК-Х9. Лишнее | ПКП-9-ППК-Х9.1. Иное (Базовый) | "
        "ПКП-9-ППК-Х9.1 З-1. Знает. |\n",
        "ckmv.md",
    )
    warnings = check_competence_codes(plan, ckmv)
    assert len(warnings) == 1 and "ПКП-ППК-Х9" in warnings[0] and "ЦКМВ" in warnings[0]
