"""Замер качества LLM-извлечения на эталонных кейсах (`tests/fixtures/llm_gold.json`).

Элемент считается найденным, если совпал ключ: нормализованное название (пререквизиты),
вид занятий и часы (активные формы) или цитата (правила оценивания). Для найденных элементов
отдельно считается точность атрибутов (`relation`/`kind`, `mandatory`). Пустой эталон
(негативный кейс) даёт ложные срабатывания на любой элемент ответа.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from edu_doc_agent.extraction.llm_extract import (
    TASKS_BY_NAME,
    normalize,
    postprocess,
    quote_in_text,
    run_task,
)
from edu_doc_agent.llm import LLMClient

_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
DEFAULT_GOLD = _FIXTURES / "llm_gold.json"  # на ней настраивались промпты
HOLDOUT = _FIXTURES / "llm_holdout.json"  # вторая порция: по ней уже правили промпты и правила
BLIND = _FIXTURES / "llm_blind.json"  # третья порция: размечена до запуска, не настраивается
BLIND2 = (
    _FIXTURES / "llm_blind2.json"
)  # четвёртая: после правок разделов и таблиц, ничего не правим

# ориентиры из ADR-0002: текстовые поля >= 0.8
THRESHOLDS = {"prerequisites": 0.8, "active_forms": 0.8, "grading_rules": 0.8}

_ITEMS = {"prerequisites": "items", "active_forms": "items", "grading_rules": "rules"}
_ATTRS = {
    "prerequisites": ("kind", "relation"),
    "active_forms": ("mandatory",),
    "grading_rules": (),
}


@dataclass
class GoldCase:
    id: str
    task: str
    source: str
    section: str
    text: str
    expected: dict
    note: str = ""


def load_gold(path: Path = DEFAULT_GOLD) -> list[GoldCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GoldCase(**c) for c in data["cases"]]


def _stem(text: str) -> str:
    """Слова усечены до 5 букв: «лабораторных работ» и «лабораторные работы» совпадают."""
    return " ".join(w[:5] for w in normalize(text).split())


def _key(task: str, item: dict) -> tuple:
    if task == "prerequisites":
        return (_stem(item["name"]),)
    if task == "active_forms":
        return (_stem(item["kind"]), item["hours"])
    return (normalize(item["quote"]).strip(" .,;:"),)


_STOP = {"и", "в", "на", "с", "по", "о", "об", "из", "при", "для", "к", "а"}


def _words(stem: str) -> set[str]:
    return {w for w in stem.split() if w not in _STOP}


def _match(task: str, pred: dict[tuple, dict], gold: dict[tuple, dict]) -> list[tuple]:
    """Пары ключей (предсказание, эталон), каждый ключ участвует не более чем в одной паре.

    Сначала точные совпадения. Правила оценивания дальше сопоставляются по цитате, одна из
    которых содержит другую: модель то захватывает точку, то нет. Пререквизиты — по словам:
    название эталона целиком входит в название ответа или наоборот («разработка простых
    программ» и «разработка простых программ на распространённых языках»). Так формулировка
    не считается ошибкой, а склейка двух названий в одно всё равно оставляет второе пропущенным.
    """
    pairs = [(k, k) for k in pred.keys() & gold.keys()]
    free_pred = [k for k in pred if k not in gold]
    free_gold = [k for k in gold if k not in pred]
    for pk in free_pred:
        gk = next((g for g in free_gold if _similar(task, pk, g, pred[pk], gold[g])), None)
        if gk is not None:
            pairs.append((pk, gk))
            free_gold.remove(gk)
    return pairs


def _similar(task: str, a: tuple, b: tuple, item_a: dict, item_b: dict) -> bool:
    if task == "grading_rules":
        return a[0] in b[0] or b[0] in a[0]
    if task == "prerequisites":
        wa, wb = _words(a[0]), _words(b[0])
        return bool(wa and wb) and (wa <= wb or wb <= wa)
    # активные формы: часы те же, а название или цитата одной записи содержится в другой
    # («консультации по проекту» и «консультации»; группа «интерактивные методы» и её части)
    if a[1] != b[1]:
        return False
    wa, wb = _words(a[0]), _words(b[0])
    qa, qb = normalize(item_a["quote"]), normalize(item_b["quote"])
    return bool(wa and wb and (wa <= wb or wb <= wa)) or (a[1] is None and (qa in qb or qb in qa))


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    attr_ok: int = 0
    attr_total: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def attribute_accuracy(self) -> float:
        return self.attr_ok / self.attr_total if self.attr_total else 1.0

    def add(self, other: "Score") -> None:
        for f in ("tp", "fp", "fn", "attr_ok", "attr_total"):
            setattr(self, f, getattr(self, f) + getattr(other, f))


def score_case(task: str, predicted: dict, expected: dict) -> Score:
    field_name = _ITEMS[task]
    pred = {_key(task, i): i for i in predicted.get(field_name, [])}
    gold = {_key(task, i): i for i in expected.get(field_name, [])}
    pairs = _match(task, pred, gold)
    unmatched = [k for k in pred if k not in {p for p, _ in pairs}]
    if task == "grading_rules":
        # правило, вырезанное из предложения эталона, — тот же факт в более мелком виде, не ложный
        unmatched = [k for k in unmatched if not any(k[0] in g[0] for g in gold)]
    score = Score(
        tp=len({g for _, g in pairs}),
        fp=len(unmatched),
        fn=len(gold) - len({g for _, g in pairs}),
    )
    for pk, gk in pairs:
        for attr in _ATTRS[task]:
            score.attr_total += 1
            score.attr_ok += pred[pk][attr] == gold[gk][attr]
    if task == "active_forms" and "total_hours" in expected:
        score.attr_total += 1
        score.attr_ok += predicted.get("total_hours") == expected["total_hours"]
    return score


@dataclass
class Report:
    by_task: dict[str, Score] = field(default_factory=lambda: defaultdict(Score))
    by_case: dict[str, Score] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)  # кейс -> почему нет ответа

    def passed(self) -> bool:
        return not self.failures and all(
            self.by_task[t].f1 >= th for t, th in THRESHOLDS.items() if t in self.by_task
        )

    def summary(self) -> str:
        lines = [f"{'задача':<16}{'P':>6}{'R':>6}{'F1':>6}{'атрибуты':>10}"]
        for task, s in sorted(self.by_task.items()):
            lines.append(
                f"{task:<16}{s.precision:>6.2f}{s.recall:>6.2f}{s.f1:>6.2f}"
                f"{s.attribute_accuracy:>10.2f}"
            )
        lines += [f"НЕТ ОТВЕТА {c}: {why}" for c, why in self.failures.items()]
        return "\n".join(lines)


def evaluate(client: LLMClient, cases: list[GoldCase] | None = None) -> Report:
    """Прогоняет кейсы через клиента (с повторами и проверкой схемы, как в проде)."""
    report = Report()
    for case in cases or load_gold():
        output, problem = run_task(TASKS_BY_NAME[case.task], case.text, client)
        if output is not None:
            output = postprocess(output)
        if output is None:
            report.failures[case.id] = problem or "нет ответа"
            report.by_task[case.task].fn += len(case.expected[_ITEMS[case.task]])
            continue
        score = score_case(case.task, _dump(output, case), case.expected)
        report.by_case[case.id] = score
        report.by_task[case.task].add(score)
    return report


def _dump(model: BaseModel, case: GoldCase) -> dict:
    """Ответ как словарь; элементы с цитатой не из текста отбрасываются, как и в проде."""
    data = model.model_dump()
    name = _ITEMS[case.task]
    data[name] = [i for i in data[name] if quote_in_text(i["quote"], case.text)]
    return data
