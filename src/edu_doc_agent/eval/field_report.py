"""Отчёт «верно/всего по каждому полю» для метрик качества на размеченных эталонах."""

from collections import defaultdict
from dataclasses import dataclass, field

THRESHOLD = 0.95  # ориентир из CLAUDE.md для кодов и чисел из таблиц


@dataclass
class FieldReport:
    correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    missing: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)
    count_errors: list[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
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
        lines = [f"{'поле':<30}{'верно':>8}{'всего':>8}{'доля':>8}"]
        for name in sorted(self.total):
            lines.append(
                f"{name:<30}{self.correct[name]:>8}{self.total[name]:>8}{self.accuracy(name):>8.3f}"
            )
        lines += [f"НЕ НАЙДЕН {m}" for m in self.missing]
        lines += [f"СЧЁТ {m}" for m in self.count_errors]
        lines += [f"  {m}" for m in self.mismatches]
        return "\n".join(lines)
