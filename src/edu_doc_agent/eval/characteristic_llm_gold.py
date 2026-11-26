"""P/R/F1 смысловых полей характеристики; эталон фиксируется до вызова модели.

Сопоставление один-к-одному: должности — по полному набору основ слов в исходном порядке,
остальные названия — по основам слов (первые пять букв) и вхождению, принципы — по цитате.
Дубли ответа считаются FP. Атрибут kind оценивается отдельно.
Это мягкая метрика сходства, а не автоматическая проверка смысловой эквивалентности.

Офлайн: python -m edu_doc_agent.eval.characteristic_llm_gold --gold GOLD --prediction JSON
Онлайн: python -m edu_doc_agent.eval.characteristic_llm_gold --gold GOLD --input MD --llm
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from edu_doc_agent.eval.llm_gold import Score
from edu_doc_agent.extraction.characteristic_llm import (
    CharacteristicLLM,
    extract_characteristic_llm,
)
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.llm import LLMError, OpenAICompatibleClient

THRESHOLD = 0.8
FIELDS = (
    "roles",
    "positions",
    "tracks",
    "projects",
    "legal_acts",
    "principles",
    "profiles",
    "individualization_means",
)


def load_gold(path: Path) -> dict:
    """Читает полный эталон; отсутствие группы не превращает её в пустой негативный пример."""
    gold = json.loads(path.read_text(encoding="utf-8"))
    expected_items(gold)
    return gold


def expected_items(gold: dict) -> dict[str, list[dict]]:
    return {
        **{name: gold["annotation"][name] for name in FIELDS[:5]},
        "principles": [{"quote": text} for text in gold["mission_principles"]],
        "profiles": gold["individualization"]["profiles"],
        "individualization_means": gold["individualization"]["means"],
    }


def _key(text: str) -> str:
    return " ".join(word[:5] for word in re.findall(r"\w+", normalize(text)))


def _similar(a: str, b: str) -> bool:
    # Границы слов не дают «проект» совпасть с частью другого слова.
    a, b = _key(a), _key(b)
    return bool(a and b) and (f" {a} " in f" {b} " or f" {b} " in f" {a} ")


def _pairs(
    predicted: list[str],
    expected: list[str],
    predicted_kinds: list[str | None] | None = None,
    expected_kinds: list[str | None] | None = None,
    *,
    require_head: bool = True,
    require_complete: bool = False,
) -> list[tuple[int, int]]:
    """Максимальное паросочетание с минимальной ценой (остаточная сеть).

    При равной полноте предпочитаем точное имя и правильный kind. Ошибочный kind
    не запрещает пару: он по-прежнему снижает точность атрибутов.
    """
    n, m = len(predicted), len(expected)
    source, sink = n + m, n + m + 1
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def edge(a: int, b: int, cost: int) -> list[int]:
        forward = [b, len(graph[b]), 1, cost]
        graph[a].append(forward)
        graph[b].append([a, len(graph[a]) - 1, 0, -cost])
        return forward

    links = []
    for p, got in enumerate(predicted):
        edge(source, p, 0)
        for g, want in enumerate(expected):
            if require_complete and _key(got) != _key(want):
                continue
            if not _similar(got, want):
                continue
            if require_head and _key(want).split()[0] not in _key(got).split():
                continue
            wrong_kind = bool(
                expected_kinds and predicted_kinds and expected_kinds[g] != predicted_kinds[p]
            )
            cost = int(_key(got) != _key(want)) * (min(n, m) + 1) + wrong_kind
            links.append((p, g, edge(p, n + g, cost)))
    for g in range(m):
        edge(n + g, sink, 0)
    while True:
        distance = [float("inf")] * len(graph)
        distance[source] = 0
        previous = {}
        for _ in range(len(graph) - 1):
            changed = False
            for a, edges in enumerate(graph):
                for i, (b, _, capacity, cost) in enumerate(edges):
                    if capacity and distance[a] + cost < distance[b]:
                        distance[b] = distance[a] + cost
                        previous[b] = (a, i)
                        changed = True
            if not changed:
                break
        if sink not in previous:
            break
        b = sink
        while b != source:
            a, i = previous[b]
            e = graph[a][i]
            e[2] -= 1
            graph[b][e[1]][2] += 1
            b = a
    return [(p, g) for p, g, e in links if e[2] == 0]


@dataclass
class CharacteristicLLMReport:
    scores: dict[str, Score] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def passed(self, threshold: float = THRESHOLD) -> bool:
        return not self.warnings and all(
            s.f1 >= threshold and s.attribute_accuracy >= threshold for s in self.scores.values()
        )

    def summary(self) -> str:
        lines = [f"{'поле':<26} {'TP':>3} {'FP':>3} {'FN':>3} {'P':>5} {'R':>5} {'F1':>5} kind"]
        for name, s in self.scores.items():
            lines.append(
                f"{name:<26} {s.tp:>3} {s.fp:>3} {s.fn:>3} {s.precision:.3f} "
                f"{s.recall:.3f} {s.f1:.3f} {s.attribute_accuracy:.3f}"
            )
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


def score_characteristic_llm(doc: CharacteristicLLM, gold: dict) -> CharacteristicLLMReport:
    report = CharacteristicLLMReport(warnings=list(doc.warnings))
    for name, expected in expected_items(gold).items():
        predicted = getattr(doc, name)
        key = "quote" if name == "principles" else "name"
        pairs = _pairs(
            [getattr(p, key) for p in predicted],
            [g[key] for g in expected],
            [p.kind for p in predicted],
            [g.get("kind") for g in expected],
            require_head=name != "principles",
            require_complete=name == "positions",
        )
        score = Score(tp=len(pairs), fp=len(predicted) - len(pairs), fn=len(expected) - len(pairs))
        for p, g in pairs:
            if "kind" in expected[g]:
                score.attr_total += 1
                score.attr_ok += predicted[p].kind == expected[g]["kind"]
        report.scores[name] = score
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction", type=Path, help="JSON результата CharacteristicLLM")
    source.add_argument("--input", type=Path, help="Markdown характеристики")
    parser.add_argument("--llm", action="store_true", help="Разрешить вызовы настоящей модели")
    parser.add_argument("--output", type=Path, help="Сохранить результат для офлайн-пересчёта")
    args = parser.parse_args(argv)
    if args.input and not args.llm:
        parser.error("для --input нужен явный флаг --llm")
    gold = load_gold(args.gold)
    if args.prediction:
        result = CharacteristicLLM.model_validate_json(args.prediction.read_text(encoding="utf-8"))
    else:
        try:
            client = OpenAICompatibleClient.from_settings()
        except LLMError:
            parser.error("задайте LLM_BASE_URL, LLM_API_KEY и LLM_MODEL")
        result = extract_characteristic_llm(
            args.input.read_text(encoding="utf-8"), args.input.name, client
        )
    if args.output:
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    report = score_characteristic_llm(result, gold)
    print(report.summary())
    return 0 if report.passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
