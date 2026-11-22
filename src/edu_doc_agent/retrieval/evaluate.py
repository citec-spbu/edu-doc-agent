"""Оценка поиска по нормативам на золотом наборе (этап 4 #32).

    uv run python -m edu_doc_agent.retrieval.evaluate data/regulations/golden.yaml

Для каждого способа (auto — как у агентов: точный, затем гибрид; fts и vector — по одному)
считаются Hit@1/3/5 — доля вопросов, где правильный пункт в первых k, — и MRR@10.
Нужны загруженный корпус и, для auto и vector, Ollama с моделью эмбеддингов.
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from edu_doc_agent.retrieval.search import Mode, RegulationHit, RegulationSearch

K_VALUES = (1, 3, 5)
DEPTH = 10


@dataclass
class Question:
    question: str
    direction: str
    relevant: list[str]  # id пунктов; «префикс*» — любой id с этим началом


@dataclass
class Outcome:
    question: Question
    rank: int | None  # место первого правильного пункта (с 1) или None
    found: list[str]  # id первых результатов


def load_golden(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [Question(q["question"], str(q["direction"]), list(q["relevant"])) for q in data]


def is_relevant(hit_id: str, relevant: Sequence[str]) -> bool:
    return any(hit_id.startswith(r[:-1]) if r.endswith("*") else hit_id == r for r in relevant)


def first_relevant(hits: Sequence[RegulationHit], relevant: Sequence[str]) -> int | None:
    return next((h.rank for h in hits if is_relevant(h.id, relevant)), None)


def metrics(outcomes: Sequence[Outcome]) -> dict[str, float]:
    n = len(outcomes) or 1
    result = {
        f"Hit@{k}": sum(1 for o in outcomes if o.rank is not None and o.rank <= k) / n
        for k in K_VALUES
    }
    result[f"MRR@{DEPTH}"] = sum(1 / o.rank for o in outcomes if o.rank is not None) / n
    return result


async def evaluate(
    search: RegulationSearch, questions: Sequence[Question], mode: Mode
) -> list[Outcome]:
    outcomes = []
    for q in questions:
        doc_ids = await search.documents_for_direction(q.direction)
        hits = await search.search(q.question, doc_ids=doc_ids, top_k=DEPTH, mode=mode)
        outcomes.append(Outcome(q, first_relevant(hits, q.relevant), [h.id for h in hits]))
    return outcomes


def report(results: dict[str, list[Outcome]]) -> str:
    modes = list(results)
    names = list(metrics([]))
    lines = ["| Способ | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for mode in modes:
        values = metrics(results[mode])
        lines.append(f"| {mode} | " + " | ".join(f"{values[n]:.2f}" for n in names) + " |")
    lines += ["", "| # | Вопрос | " + " | ".join(modes) + " |", "|---" * (len(modes) + 2) + "|"]
    for i, q in enumerate(results[modes[0]], start=1):
        ranks = [results[m][i - 1].rank for m in modes]
        cells = " | ".join(str(r) if r else "—" for r in ranks)
        lines.append(f"| {i} | {q.question.question} | {cells} |")
    misses = [o for o in results[modes[0]] if o.rank is None or o.rank > 3]
    if misses:
        lines += ["", f"Не в первой тройке ({modes[0]}):", ""]
        for o in misses:
            lines.append(f"- {o.question.question}")
            lines.append(f"  - ожидалось: {', '.join(o.question.relevant)}")
            lines.append(f"  - найдено: {', '.join(o.found[:5]) or '—'}")
    return "\n".join(lines)


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
        results = {mode: await evaluate(search, questions, mode) for mode in args.modes}
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
    parser.add_argument("--modes", nargs="+", default=["auto", "fts", "vector"])
    parser.add_argument("--out", type=Path, help="сохранить отчёт в Markdown")
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
