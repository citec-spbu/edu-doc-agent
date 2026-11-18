"""Метрика качества извлечения учебного плана на размеченных эталонах.

Эталоны (`data/evaluation/plan_gold*.json`) составлены по сырому Markdown реального плана, а не
по выводу экстрактора. Элемент плана ищется в результате по (код, название, семестр), при
совпадении ключа — по порядку в документе; для найденного сравнивается каждое поле отдельно,
поэтому видно, что именно извлекается хуже. Ненайденный элемент считается ошибкой по всем полям.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from edu_doc_agent.eval.datasets import local_gold
from edu_doc_agent.extraction.curriculum import CurriculumPlan, PlanEntry
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.patterns import code_key

PLAN_GOLD = local_gold("plan_gold.json")  # 40 записей, перенесены вручную
PLAN_GOLD_FULL = local_gold("plan_gold_full.json")  # все строки плана, разбор сырого md

# ориентир из CLAUDE.md: числа и коды из таблиц детерминированно, точность >= 0.95
THRESHOLD = 0.95

ENTRY_FIELDS = (
    "block",
    "code",
    "name",
    "name_en",
    "credits",
    "competences",
    "control",
    "contact_hours",
    "self_study_hours",
    "active_hours",
    "semester",
    "year",
    "part",
    "group",
    "profile",
    "trajectory",
)


@dataclass
class PlanReport:
    correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hour_cells_ok: int = 0
    hour_cells: int = 0
    missing: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)
    count_errors: list[str] = field(default_factory=list)

    def _add(self, name: str, ok: bool, detail: str = "") -> None:
        self.total[name] += 1
        self.correct[name] += ok
        if not ok and detail:
            self.mismatches.append(f"{name}: {detail}")

    def accuracy(self, name: str) -> float:
        return self.correct[name] / self.total[name] if self.total[name] else 1.0

    @property
    def overall(self) -> float:
        total = sum(self.total.values())
        return sum(self.correct.values()) / total if total else 1.0

    @property
    def hours_cell_accuracy(self) -> float:
        return self.hour_cells_ok / self.hour_cells if self.hour_cells else 1.0

    def passed(self, threshold: float = THRESHOLD) -> bool:
        return (
            not self.missing
            and not self.count_errors
            and all(self.accuracy(n) >= threshold for n in self.total)
        )

    def summary(self) -> str:
        lines = [f"{'поле':<22}{'верно':>8}{'всего':>8}{'доля':>8}"]
        for name in sorted(self.total):
            lines.append(
                f"{name:<22}{self.correct[name]:>8}{self.total[name]:>8}{self.accuracy(name):>8.3f}"
            )
        lines.append(f"ячейки часов: {self.hour_cells_ok}/{self.hour_cells}")
        lines += [f"НЕ НАЙДЕН {m}" for m in self.missing]
        lines += [f"СЧЁТ {m}" for m in self.count_errors]
        lines += [f"  {m}" for m in self.mismatches]
        return "\n".join(lines)


def load_plan_gold(path: Path = PLAN_GOLD) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_list(values: list[str]) -> list[str]:
    return [code_key(v) for v in values]


def _entry_key(code: str | None, name: str, semester: int | None) -> tuple:
    return (code, normalize(name), semester)


def _same(field_name: str, got: object, want: object) -> bool:
    if field_name in ("name", "name_en", "part", "group", "profile", "trajectory"):
        return (normalize(got) if got else None) == (normalize(want) if want else None)
    if field_name == "competences":
        return sorted(_norm_list(got)) == sorted(_norm_list(want))  # type: ignore[arg-type]
    if field_name == "control":
        return [normalize(c) for c in got] == [normalize(c) for c in want]  # type: ignore[union-attr]
    if field_name in ("credits", "active_hours"):
        return (float(got) if got is not None else None) == (
            float(want) if want is not None else None
        )
    return got == want


_HOUR_DICTS = ("contact_hours", "self_study_hours")


def _got(entry: PlanEntry, field_name: str) -> object:
    value = getattr(entry, field_name)
    return value


def _gold_value(gold: dict, field_name: str) -> object:
    renamed = {"contact_hours": "contact", "self_study_hours": "self", "active_hours": "active"}
    value = gold[renamed.get(field_name, field_name)]
    if field_name in _HOUR_DICTS and not isinstance(value, dict):
        raise ValueError("эталон часов должен хранить названия видов занятий, а не массив значений")
    return value


def score_plan(plan: CurriculumPlan, gold: dict | None = None) -> PlanReport:
    gold = gold or load_plan_gold()
    report = PlanReport()

    index: dict[tuple, list[PlanEntry]] = defaultdict(list)
    for e in plan.entries:
        index[_entry_key(e.code, e.name, e.semester)].append(e)
    seen: dict[tuple, int] = defaultdict(int)  # n-я запись эталона с ключом ↔ n-я запись результата
    for g in gold["entries"]:
        label = f"{g['code']} «{g['name']}» сем. {g['semester']} (строка {g['line']})"
        key = _entry_key(g["code"], g["name"], g["semester"])
        found = index.get(key, [])
        nth = seen[key]
        seen[key] += 1
        if nth >= len(found):
            report.missing.append(label)
            for name in ENTRY_FIELDS:
                report._add(name, False)
            report.hour_cells += len(g["contact"]) + len(g["self"])
            continue
        entry = found[nth]
        for name in ENTRY_FIELDS:
            got, want = _got(entry, name), _gold_value(g, name)
            report._add(
                name, _same(name, got, want), f"{label}: получено {got!r}, ожидалось {want!r}"
            )
        for name, want in (("contact_hours", g["contact"]), ("self_study_hours", g["self"])):
            got = _got(entry, name)
            report.hour_cells += len(want)
            report.hour_cells_ok += sum(got.get(kind) == hours for kind, hours in want.items())

    by_code = {code_key(c.code): c for c in plan.competences}
    for g in gold["competences"]:
        c = by_code.get(code_key(g["code"]))
        report._add(
            "competence_name",
            c is not None and normalize(c.name) == normalize(g["name"]),
            f"{g['code']}: получено {c.name if c else None!r}",
        )

    for name, want in gold["title"].items():
        got = getattr(plan, name)
        report._add(
            f"title.{name}", normalize(got or "") == normalize(want), f"{got!r} != {want!r}"
        )

    by_name = {normalize(a.name): a for a in plan.final_attestation}
    for g in gold["final_attestation"]:
        a = by_name.get(normalize(g["name"]))
        ok = a is not None
        report._add("gia.found", ok, g["name"])
        if a is not None:
            report._add("gia.credits", a.credits == g["credits"])
            report._add("gia.block", a.block == g["block"])
            report._add("gia.name_en", normalize(a.name_en or "") == normalize(g["name_en"]))
            report._add(
                "gia.competences",
                sorted(_norm_list(a.competences)) == sorted(_norm_list(g["competences"])),
            )

    for name, got in (
        ("entries", len(plan.entries)),
        ("competences", len(plan.competences)),
        ("final_attestation", len(plan.final_attestation)),
    ):
        if got != gold["totals"][name]:
            report.count_errors.append(f"{name}: {got}, ожидалось {gold['totals'][name]}")
    return report
