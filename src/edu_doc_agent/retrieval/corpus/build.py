"""Сборка корпуса нормативов: манифест -> парсинг -> чанкер -> YAML-снимки и отчёт.

    uv run python -m edu_doc_agent.retrieval.corpus.build data/regulations/manifest.yaml \\
        --overrides data/regulations/overrides.yaml --out data/corpus

Манифест (пути — относительно файла манифеста):

    sources:
      - files: "fgos/*.pdf"            # каждый файл — ФГОС, метаданные определяются по тексту
        type: fgos
      - file: fgos/020302_2023.pdf      # можно указать файл и переопределить найденное
        type: fgos
        edition: "808 ред. от 27.02.2023"
      - file: spbu/os_2026.pdf
        type: spbu_standard              # для не-ФГОС doc_id, title, edition — обязательны
        doc_id: spbu-os
        title: Образовательный стандарт высшего образования СПбГУ
        edition: "11144/1 от 24.08.2026"

Новый ФГОС добавляется без кода: положить файл в папку из `files` и пересобрать.
YAML-снимки нужны, чтобы проверить нарезку глазами; руками их не правят — правки идут
в overrides (по `doc_id`): `replace` (id -> текст), `add` (новые фрагменты), `drop` (id).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

from edu_doc_agent.parsing import parse_document
from edu_doc_agent.retrieval.corpus.chunker import chunk_document, detect_info
from edu_doc_agent.retrieval.corpus.models import CorpusDocument, Overrides

_META_KEYS = (
    "doc_id",
    "title",
    "edition",
    "approval_act",
    "direction_code",
    "direction_name",
    "level",
)


def build(manifest: Path, overrides: Path | None = None) -> tuple[list[CorpusDocument], list[str]]:
    """Нарезать все документы манифеста -> (документы, ошибки по файлам).

    Ошибка разбора одного файла (скан без текста, не найдены метаданные) не останавливает сборку.
    """
    config = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    patches = _load_overrides(overrides)
    documents, failures = [], []
    for path, entry in _sources(manifest, config):
        given = {k: str(entry[k]) for k in _META_KEYS if entry.get(k)}
        try:
            parsed = parse_document(path)
            info = detect_info(parsed, entry["type"], path.name, **given)
        except ValueError as error:  # UnsupportedFormatError, NoTextLayerError, метаданные
            failures.append(f"{path.name}: {error}")
            continue
        documents.append(chunk_document(parsed, info, patches.get(info.doc_id)))
    return documents, failures


def _sources(manifest: Path, config: dict) -> list[tuple[Path, dict]]:
    base = manifest.parent
    explicit = {(base / e["file"]).resolve(): e for e in config.get("sources", []) if "file" in e}
    out: dict[Path, dict] = {}
    for entry in config.get("sources", []):
        if "files" in entry:
            for path in sorted(base.glob(entry["files"])):
                out.setdefault(path.resolve(), entry)
    out |= explicit  # явная запись о файле важнее шаблона
    return sorted(out.items())


def _load_overrides(path: Path | None) -> dict[str, Overrides]:
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {doc_id: Overrides.model_validate(v or {}) for doc_id, v in data.items()}


def write(documents: list[CorpusDocument], out: Path, failures: list[str] = ()) -> list[str]:
    """YAML-снимок на документ и REPORT.md. Возвращает ошибки сборки (пусто — всё в порядке)."""
    out.mkdir(parents=True, exist_ok=True)
    errors = list(failures)
    seen = Counter(d.document.doc_id for d in documents)
    errors += [f"doc_id {k} встречается {n} раза" for k, n in seen.items() if n > 1]
    for doc in documents:
        errors += [f"{doc.document.doc_id}: {w}" for w in doc.warnings if w.startswith("ОШИБКА")]
        (out / f"{doc.document.doc_id}.yaml").write_text(_dump(doc), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(documents, errors), encoding="utf-8")
    return errors


def _dump(doc: CorpusDocument) -> str:
    class Dumper(yaml.SafeDumper):
        pass

    def text(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    Dumper.add_representer(str, text)
    data = doc.model_dump(exclude_defaults=True)
    return yaml.dump(data, Dumper=Dumper, allow_unicode=True, sort_keys=False, width=100)


def _report(documents: list[CorpusDocument], errors: list[str]) -> str:
    lines = ["# Корпус нормативов: отчёт сборки", ""]
    lines += ["| doc_id | Документ | Редакция | Пункты | Компетенции | Профстандарты | Части |"]
    lines += ["| --- | --- | --- | --- | --- | --- | --- |"]
    for doc in documents:
        kinds = Counter(f.kind for f in doc.fragments if f.part == 0)
        parts = sum(1 for f in doc.fragments if f.part > 0)
        d = doc.document
        lines.append(
            f"| {d.doc_id} | {d.title} | {d.edition} | {kinds['clause']} | "
            f"{kinds['competency']} | {kinds['prof_standard_ref']} | {parts} |"
        )
    lines += ["", "## Замечания", ""]
    for doc in documents:
        if doc.warnings:
            lines.append(f"**{doc.document.doc_id}** ({doc.document.source_file})")
            lines += [f"- {w}" for w in doc.warnings]
            lines.append("")
    if errors:
        lines += ["## Ошибки", "", *[f"- {e}" for e in errors], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сборка корпуса нормативов для RAG")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/corpus"))
    args = parser.parse_args(argv)
    documents, failures = build(args.manifest, args.overrides)
    errors = write(documents, args.out, failures)
    for doc in documents:
        total = sum(1 for f in doc.fragments if f.part == 0)
        print(f"{doc.document.doc_id}: {total} фрагментов, замечаний: {len(doc.warnings)}")
    print(f"Отчёт: {args.out / 'REPORT.md'}")
    for error in errors:
        print(f"ОШИБКА: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
