"""Метрика качества извлечения ЦКМВ на размеченных эталонах.

Эталоны (`data/evaluation/ckmv_gold*.json`) получены по сырому Markdown, а не по выводу
экстрактора. Компетенции сопоставляются по коду и порядку появления (один код может стоять в
нескольких группах), индикаторы и результаты — по коду внутри родителя; для найденных сравнивается
каждое поле отдельно. Ненайденная сущность считается ошибкой по всем своим полям.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from edu_doc_agent.eval.datasets import local_gold
from edu_doc_agent.extraction.ckmv import CkmvDocument
from edu_doc_agent.extraction.llm_extract import normalize

CKMV_GOLD = local_gold("ckmv_gold.json")  # 3 компетенции, перенесены вручную
CKMV_GOLD_FULL = local_gold("ckmv_gold_full.json")  # вся ЦКМВ, разбор сырых строк

THRESHOLD = 0.95  # ориентир из CLAUDE.md для кодов и чисел из таблиц

COMPETENCE_FIELDS = ("name", "group", "role_section", "role", "n_indicators")
INDICATOR_FIELDS = ("name", "level", "n_outcomes")
OUTCOME_FIELDS = ("letter", "number", "kind", "text")


@dataclass
class CkmvReport:
    correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
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

    def passed(self, threshold: float = THRESHOLD) -> bool:
        return (
            not self.missing
            and not self.count_errors
            and all(self.accuracy(n) >= threshold for n in self.total)
        )

    def summary(self) -> str:
        lines = [f"{'поле':<28}{'верно':>8}{'всего':>8}{'доля':>8}"]
        for name in sorted(self.total):
            lines.append(
                f"{name:<28}{self.correct[name]:>8}{self.total[name]:>8}{self.accuracy(name):>8.3f}"
            )
        lines += [f"НЕ НАЙДЕН {m}" for m in self.missing]
        lines += [f"СЧЁТ {m}" for m in self.count_errors]
        lines += [f"  {m}" for m in self.mismatches]
        return "\n".join(lines)


def load_ckmv_gold(path: Path = CKMV_GOLD) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_TEXT_FIELDS = {"name", "group", "role_section", "role", "text"}


def _same(name: str, got: object, want: object) -> bool:
    if name in _TEXT_FIELDS:
        return (normalize(str(got)) if got else None) == (normalize(str(want)) if want else None)
    return got == want


def _nth(seen: dict[str, int], key: str) -> int:
    n = seen[key]
    seen[key] += 1
    return n


def score_ckmv(doc: CkmvDocument, gold: dict | None = None) -> CkmvReport:
    gold = gold or load_ckmv_gold()
    report = CkmvReport()
    by_code: dict[str, list] = defaultdict(list)
    for c in doc.competences:
        by_code[c.code].append(c)
    seen: dict[str, int] = defaultdict(int)

    for g in gold["competences"]:
        label = f"{g['code']} (строка {g['line']})"
        found = by_code.get(g["code"], [])
        nth = _nth(seen, g["code"])
        if nth >= len(found):
            report.missing.append(label)
            for n in COMPETENCE_FIELDS:
                report._add("competence." + n, False)
            continue
        comp = found[nth]
        got = {
            "name": comp.name,
            "group": comp.group,
            "role_section": comp.role_section,
            "role": comp.role,
            "n_indicators": len(comp.indicators),
        }
        want = {**g, "n_indicators": len(g["indicators"])}
        for n in COMPETENCE_FIELDS:
            report._add(
                "competence." + n,
                _same(n, got[n], want[n]),
                f"{label}: получено {got[n]!r}, ожидалось {want[n]!r}",
            )
        _score_indicators(comp, g, report)

    for name, got in (
        ("competences", len(doc.competences)),
        ("indicators", len(doc.indicators)),
        ("outcomes", len(doc.outcomes)),
    ):
        if gold.get("complete") and got != gold["totals"][name]:
            report.count_errors.append(f"{name}: {got}, ожидалось {gold['totals'][name]}")
    return report


def _score_indicators(comp, g: dict, report: CkmvReport) -> None:
    by_code = defaultdict(list)
    for i in comp.indicators:
        by_code[i.code].append(i)
    seen: dict[str, int] = defaultdict(int)
    for gi in g["indicators"]:
        label = f"{gi['code']} (строка {gi['line']})"
        found = by_code.get(gi["code"], [])
        nth = _nth(seen, gi["code"])
        if nth >= len(found):
            report.missing.append(label)
            for n in INDICATOR_FIELDS:
                report._add("indicator." + n, False)
            continue
        ind = found[nth]
        got = {"name": ind.name, "level": ind.level, "n_outcomes": len(ind.outcomes)}
        want = {**gi, "n_outcomes": len(gi["outcomes"])}
        for n in INDICATOR_FIELDS:
            report._add(
                "indicator." + n,
                _same(n, got[n], want[n]),
                f"{label}: получено {got[n]!r}, ожидалось {want[n]!r}",
            )
        _score_outcomes(ind, gi, report)


def _score_outcomes(ind, gi: dict, report: CkmvReport) -> None:
    by_code = defaultdict(list)
    for o in ind.outcomes:
        by_code[o.code].append(o)
    seen: dict[str, int] = defaultdict(int)
    for go in gi["outcomes"]:
        label = f"{go['code']} (строка {go['line']})"
        found = by_code.get(go["code"], [])
        nth = _nth(seen, go["code"])
        if nth >= len(found):
            report.missing.append(label)
            for n in OUTCOME_FIELDS:
                report._add("outcome." + n, False)
            continue
        out = found[nth]
        for n in OUTCOME_FIELDS:
            got = getattr(out, n)
            report._add(
                "outcome." + n,
                _same(n, got, go[n]),
                f"{label}: получено {got!r}, ожидалось {go[n]!r}",
            )
