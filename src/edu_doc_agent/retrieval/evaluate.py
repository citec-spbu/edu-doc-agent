"""Оценка поиска по нормативам на золотом наборе (этап 4 #32).

    uv run python -m edu_doc_agent.retrieval.evaluate data/regulations/golden.yaml

Две метрики, обе — доля вопросов из золотого набора, на которые в нормативах есть ответ:

- «первым» — правильный пункт на первом месте выдачи;
- «в первых 5» — правильный пункт среди первых пяти (столько по умолчанию видит агент).

Сравниваются способы: только векторный, только полнотекстовый (слова запроса через ИЛИ и
через И) и гибрид с разным сочетанием слов и весом полнотекстового списка в RRF.
Вопросы без ответа (`relevant: []`) в метрики не входят: по ним видно, какое сходство с
запросом у первого результата, когда ответа нет. Нужны загруженный корпус и Ollama с моделью
эмбеддингов.
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import yaml

from edu_doc_agent.dao.regulations import TextMatch
from edu_doc_agent.retrieval.search import (
    FTS_MATCH,
    FTS_WEIGHT,
    Mode,
    RegulationHit,
    RegulationSearch,
)

TOP = 5  # «в первых 5»
DEPTH = 10  # сколько результатов смотрим, чтобы показать место правильного пункта
FTS_WEIGHTS = (1.0, 0.25)
MATCH_NAMES: dict[TextMatch, str] = {"any": "ИЛИ", "all": "И"}


@dataclass
class Question:
    question: str
    direction: str
    relevant: list[str]  # id пунктов; «префикс*» — любой id с этим началом; [] — ответа нет


@dataclass
class Outcome:
    question: Question
    rank: int | None  # место первого правильного пункта (с 1) или None
    found: list[str]  # id первых результатов
    similarity: float | None = None  # сходство первого результата с вопросом


@dataclass(frozen=True)
class Variant:
    """Способ поиска для сравнения."""

    name: str
    mode: Mode
    fts_match: TextMatch | None = None
    fts_weight: float | None = None


def hybrid_name(match: TextMatch, weight: float) -> str:
    return f"гибрид, {MATCH_NAMES[match]}, вес текста {weight:g}"


def variants(weights: Sequence[float] = FTS_WEIGHTS) -> list[Variant]:
    out = [
        Variant("вектор", "vector"),
        Variant("текст, ИЛИ", "fts", "any"),
        Variant("текст, И", "fts", "all"),
    ]
    for match in MATCH_NAMES:
        out += [Variant(hybrid_name(match, w), "auto", match, w) for w in weights]
    return out


def load_golden(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [Question(q["question"], str(q["direction"]), list(q["relevant"])) for q in data]


def is_relevant(hit_id: str, relevant: Sequence[str]) -> bool:
    return any(hit_id.startswith(r[:-1]) if r.endswith("*") else hit_id == r for r in relevant)


def first_relevant(hits: Sequence[RegulationHit], relevant: Sequence[str]) -> int | None:
    """Место первого правильного результата; перечень засчитывается, если правильный пункт
    вошёл в него (`covers`)."""
    return next(
        (h.rank for h in hits if any(is_relevant(i, relevant) for i in [h.id, *h.covers])),
        None,
    )


def metrics(outcomes: Sequence[Outcome]) -> dict[str, float]:
    answerable = [o for o in outcomes if o.question.relevant]
    n = len(answerable) or 1
    return {
        "первым": sum(1 for o in answerable if o.rank == 1) / n,
        f"в первых {TOP}": sum(1 for o in answerable if o.rank is not None and o.rank <= TOP) / n,
    }


async def evaluate(
    search: RegulationSearch, questions: Sequence[Question], variant: Variant
) -> list[Outcome]:
    outcomes = []
    for q in questions:
        doc_ids = await search.documents_for_direction(q.direction)
        hits = await search.search(
            q.question,
            doc_ids=doc_ids,
            top_k=DEPTH,
            mode=variant.mode,
            fts_match=variant.fts_match,
            fts_weight=variant.fts_weight,
        )
        similarity = hits[0].similarity if hits else None
        outcomes.append(
            Outcome(q, first_relevant(hits, q.relevant), [h.id for h in hits], similarity)
        )
    return outcomes


def report(results: dict[str, list[Outcome]]) -> str:
    modes = list(results)
    names = list(metrics([]))
    answerable = [o for o in results[modes[0]] if o.question.relevant]
    n = len(answerable)
    lines = [
        f"Вопросов с ответом: {n}. Доля вопросов, где правильный пункт…",
        "",
        "| Способ | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for mode in modes:
        values = metrics(results[mode])
        cells = " | ".join(f"{values[name]:.0%} ({round(values[name] * n)})" for name in names)
        lines.append(f"| {mode} | {cells} |")
    lines += ["", "Место правильного пункта (— — нет в первых 10):", ""]
    lines += ["| # | Вопрос | " + " | ".join(modes) + " |", "|---" * (len(modes) + 2) + "|"]
    for i, o in enumerate(results[modes[0]]):
        if not o.question.relevant:
            continue
        cells = " | ".join(str(results[m][i].rank or "—") for m in modes)
        lines.append(f"| {i + 1} | {o.question.question} | {cells} |")
    # разбор — для способа, которым ищут агенты (гибрид с настройками по умолчанию)
    default = hybrid_name(FTS_MATCH, FTS_WEIGHT)
    main_mode = default if default in results else modes[-1]
    misses = [o for o in results[main_mode] if o.question.relevant and o.rank != 1]
    if misses:
        lines += ["", f"Правильный пункт не первым ({main_mode}):", ""]
        for o in misses:
            lines.append(f"- {o.question.question}")
            lines.append(f"  - ожидалось: {', '.join(o.question.relevant)}")
            lines.append(f"  - найдено: {', '.join(o.found[:5]) or '—'}")
    lines += _similarity_lines(results[main_mode], main_mode)
    return "\n".join(lines)


def _similarity_lines(outcomes: Sequence[Outcome], mode: str) -> list[str]:
    """Сходство первого результата: с ответом и без — чтобы выбрать порог «не нашлось»."""

    def values(answerable: bool) -> list[float]:
        return sorted(
            o.similarity
            for o in outcomes
            if o.similarity is not None and bool(o.question.relevant) == answerable
        )

    found, missing = values(True), values(False)
    if not found:
        return []
    lines = [
        "",
        f"Сходство первого результата с вопросом ({mode}):",
        "",
        f"- есть ответ: от {found[0]:.2f} до {found[-1]:.2f}, медиана {median(found):.2f}",
    ]
    if missing:
        lines.append(f"- ответа нет: от {missing[0]:.2f} до {missing[-1]:.2f}")
        for o in outcomes:
            if not o.question.relevant and o.similarity is not None and o.found:
                lines.append(f"  - {o.question.question} — {o.similarity:.2f}, {o.found[0]}")
    return lines


async def _run(args: argparse.Namespace) -> int:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from edu_doc_agent.config import get_settings
    from edu_doc_agent.dao.regulations import RegulationDAO
    from edu_doc_agent.llm.embeddings import OllamaEmbedder

    questions = load_golden(args.golden)
    settings = get_settings()
    engine = create_async_engine(settings.database_connection_url, poolclass=NullPool)
    try:
        dao = RegulationDAO(async_sessionmaker(engine, expire_on_commit=False))
        search = RegulationSearch(dao, OllamaEmbedder.from_settings(settings))
        results = {v.name: await evaluate(search, questions, v) for v in variants(args.fts_weights)}
    finally:
        await engine.dispose()
    text = report(results)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Оценка поиска по нормативам")
    parser.add_argument("golden", type=Path)
    parser.add_argument("--fts-weights", nargs="+", type=float, default=list(FTS_WEIGHTS))
    parser.add_argument("--out", type=Path, help="сохранить отчёт в Markdown")
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
