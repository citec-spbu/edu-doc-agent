"""Эталон характеристики: сам эталон, чувствительность метрики и замер на реальных документах."""

from pathlib import Path

import pytest

from edu_doc_agent.eval.characteristic_gold import (
    CHAR_GOLD_FULL,
    THRESHOLD,
    load_char_gold,
    score_characteristic,
)
from edu_doc_agent.extraction.characteristic import (
    ActivityKind,
    CharCompetence,
    CompetenceMapping,
    LaborFunction,
    OkvedCode,
    ProfessionalArea,
    ProfessionalStandard,
    ProgramCharacteristic,
    TextItem,
    extract_characteristic,
)
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.source import SourceRef
from tests.real_data import real_characteristic_markdown

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MANUAL = load_char_gold(_FIXTURES / "characteristic_gold.json")
FULL = load_char_gold(_FIXTURES / "characteristic_gold_full.json")
REF = SourceRef(file="gold")
needs_data = pytest.mark.skipif(
    real_characteristic_markdown() is None or not CHAR_GOLD_FULL.exists(),
    reason="нет data/Характеристика.zip",
)


def _doc_from(gold: dict) -> ProgramCharacteristic:
    """Результат «идеального» экстрактора: ровно то, что записано в эталоне."""
    doc = ProgramCharacteristic(file="gold", **gold["title"])
    doc.qualification = list(gold["qualification"])
    doc.areas = [ProfessionalArea(source=REF, **a) for a in _clean(gold["areas"], ("code", "name"))]
    doc.standards = [
        ProfessionalStandard(source=REF, **s) for s in _clean(gold["standards"], ("code", "name"))
    ]
    doc.employers = [TextItem(source=REF, text=e["text"]) for e in gold["employers"]]
    doc.objects = [TextItem(source=REF, text=o["text"]) for o in gold.get("objects", [])]
    counts = gold["totals"].get("task_counts")
    for i, a in enumerate(gold["activities"]):
        tasks = [TextItem(source=REF, text=t["text"]) for t in a.get("tasks", [])]
        if counts:
            tasks = [TextItem(source=REF, text=f"задача {j}") for j in range(counts[i])]
        doc.activities.append(
            ActivityKind(
                name=a["name"],
                okved=[OkvedCode(source=REF, **o) for o in _clean(a["okved"], ("code", "name"))],
                tasks=tasks,
                source=REF,
            )
        )
    for c in gold["competences"]:
        doc.competences.append(
            CharCompetence(
                code=c["code"],
                name=c["name"],
                group=c["group"],
                section=c.get("section"),
                source=REF,
            )
        )
    for m in gold["mappings"]:
        funcs = [LaborFunction(source=REF, **f) for f in m["functions"]]
        doc.mappings.append(
            CompetenceMapping(competence_code=m["competence_code"], functions=funcs, source=REF)
        )
    if "group_counts" in gold:  # у ручного набора проверяются счётчики групп: добиваем до них
        for group, n in gold["group_counts"].items():
            current = sum(c.group == group for c in doc.competences)
            for k in range(n - current):
                doc.competences.append(
                    CharCompetence(code=f"ЗАП-{group[:3]}{k}", name="n", group=group, source=REF)
                )
    return doc


def _clean(items: list[dict], keys: tuple[str, ...]) -> list[dict]:
    return [{k: i[k] for k in keys} for i in items]


# --- сам эталон ------------------------------------------------------------------------------


def test_full_gold_totals_and_shape() -> None:
    t = FULL["totals"]
    assert (len(FULL["competences"]), len(FULL["mappings"])) == (t["competences"], t["mappings"])
    assert sum(len(m["functions"]) for m in FULL["mappings"]) == t["functions"] == 9
    assert len(FULL["objects"]) == t["objects"] == 1 and len(FULL["standards"]) == 3
    assert FULL["complete"] and not MANUAL["complete"]


def test_gold_covers_the_hard_cases() -> None:
    groups = {c["group"] for c in FULL["competences"]}
    assert len(groups) == 2 and "общепрофессиональные" in groups
    letters = {f["code"] for m in MANUAL["mappings"] for f in m["functions"] if "А" in f["code"]}
    assert letters, "ручной эталон должен включать код с кириллической «А»"
    assert any(f["name"].startswith("Cоз") for m in MANUAL["mappings"] for f in m["functions"])
    assert any(len(m["functions"]) >= 3 for m in MANUAL["mappings"])


def test_synthetic_subsets_agree() -> None:
    assert MANUAL["title"] == FULL["title"]
    assert MANUAL["qualification"] == FULL["qualification"]
    key = lambda items, *ks: [tuple(i[k] for k in ks) for i in items]  # noqa: E731
    assert key(MANUAL["areas"], "code", "name") == key(FULL["areas"], "code", "name")
    assert key(MANUAL["standards"], "code", "name") == key(FULL["standards"], "code", "name")
    assert key(MANUAL["employers"], "text") == key(FULL["employers"], "text")
    for a, b in zip(MANUAL["activities"], FULL["activities"], strict=True):
        assert a["name"] == b["name"]
        assert key(a["okved"], "code", "name") == key(b["okved"], "code", "name")
    assert MANUAL["totals"]["task_counts"] == [len(a["tasks"]) for a in FULL["activities"]]
    full = {c["code"]: c for c in FULL["competences"]}
    for c in MANUAL["competences"]:
        assert (c["name"], c["group"]) == (full[c["code"]]["name"], full[c["code"]]["group"])
    mapped = {m["competence_code"]: m["functions"] for m in FULL["mappings"]}
    for m in MANUAL["mappings"]:
        assert m["functions"] == mapped[m["competence_code"]], m["competence_code"]
    assert normalize("x") == "x"


# --- чувствительность метрики ----------------------------------------------------------------


def test_perfect_result_scores_one() -> None:
    for gold in (MANUAL, FULL):
        report = score_characteristic(_doc_from(gold), gold)
        assert report.passed() and report.overall == 1.0, report.summary()


def _first_function(doc: ProgramCharacteristic) -> LaborFunction:
    return next(m for m in doc.mappings if m.competence_code == "ПКП-ППК-Р1").functions[1]


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("title.direction_code", lambda d: setattr(d, "direction_code", "09.03.01")),
        ("title.duration", lambda d: setattr(d, "duration", "5 лет")),
        ("area.name", lambda d: setattr(d.areas[0], "name", "другое")),
        ("standard.code", lambda d: setattr(d.standards[2], "code", "06.999")),
        ("employer.text", lambda d: setattr(d.employers[4], "text", "ООО «Другая»")),
        ("activity.okved", lambda d: d.activities[0].okved.pop()),
        ("activity.task_counts", lambda d: d.activities[1].tasks.pop()),
        ("competence.group", lambda d: setattr(d.competences[3], "group", "иное")),
        ("competence.name", lambda d: setattr(d.competences[3], "name", "иное")),
        ("function.level", lambda d: setattr(_first_function(d), "level", 9)),
        ("function.function_letter", lambda d: setattr(_first_function(d), "function_letter", "Z")),
        ("function.name", lambda d: setattr(_first_function(d), "name", "иное")),
        (
            "function.standard_code",
            lambda d: setattr(_first_function(d), "standard_code", "99.999"),
        ),
    ],
)
def test_corrupting_one_field_is_detected(field: str, mutate) -> None:
    doc = _doc_from(MANUAL)
    mutate(doc)
    report = score_characteristic(doc, MANUAL)
    assert report.accuracy(field) < 1.0, report.summary()
    assert not report.passed(1.0)


def test_cyrillic_and_latin_function_codes_are_equivalent() -> None:
    doc = _doc_from(MANUAL)
    fn = _first_function(doc)
    fn.code = fn.code.replace("А", "A")  # «06.015.А.4» → «06.015.A.4»
    assert score_characteristic(doc, MANUAL).accuracy("function.code") == 1.0


def test_missing_competence_and_wrong_totals_fail() -> None:
    doc = _doc_from(FULL)
    del doc.competences[5]
    del doc.mappings[2]
    report = score_characteristic(doc, FULL)
    assert report.missing and report.count_errors and not report.passed()


def test_threshold_matches_claude_md_target() -> None:
    assert THRESHOLD == 0.95


# --- реальные документы ----------------------------------------------------------------------


@needs_data
def test_real_characteristic_accuracy_on_manual_gold() -> None:
    doc = extract_characteristic(real_characteristic_markdown() or "", "char.docx")
    report = score_characteristic(doc, load_char_gold())
    assert report.passed(), report.summary()


@needs_data
def test_real_characteristic_accuracy_on_full_gold() -> None:
    doc = extract_characteristic(real_characteristic_markdown() or "", "char.docx")
    report = score_characteristic(doc, load_char_gold(CHAR_GOLD_FULL))
    assert report.total["competence.name"] == 49 and report.total["function.name"] == 56
    assert report.passed(), report.summary()
