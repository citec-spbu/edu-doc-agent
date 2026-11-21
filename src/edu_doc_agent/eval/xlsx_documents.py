"""Воспроизводимый замер XLSX-плана и матрицы на локальных эталонах.

Подготовка плана: --prepare-plan-reference (OOXML, без запуска извлекателя).
Замер: --output data/evaluation/xlsx_quality.json --report docs/extraction-xlsx-evaluation.md.
Без полных эталонов или исходников запуск завершается ошибкой, а не успешным пустым замером.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from edu_doc_agent.eval.matrix_gold import score_matrix
from edu_doc_agent.eval.plan_gold import ENTRY_FIELDS, score_plan
from edu_doc_agent.eval.program_ner import compare_groups, matrix_records, plan_records, record
from edu_doc_agent.eval.xlsx_reference import build_plan_reference, read_xlsx_xml
from edu_doc_agent.extraction.curriculum import CurriculumPlan, extract_plan_file
from edu_doc_agent.extraction.matrix import extract_matrix


def _member(data_dir: Path, archive: str) -> tuple[bytes, str]:
    matches = [p for p in data_dir.glob("*.zip") if unicodedata.normalize("NFC", p.name) == archive]
    if len(matches) != 1:
        raise FileNotFoundError(f"Ожидался один архив {data_dir / archive}")
    with ZipFile(matches[0]) as z:
        members = [
            n for n in z.namelist() if n.lower().endswith(".xlsx") and not n.startswith("__MACOSX/")
        ]
        if len(members) != 1:
            raise ValueError(f"{archive}: ожидался один XLSX")
        return z.read(members[0]), members[0]


def _field_value(entry: dict, name: str) -> object:
    return entry.get(
        {"contact_hours": "contact", "self_study_hours": "self", "active_hours": "active"}.get(
            name, name
        ),
        entry.get(name),
    )


def _numeric_values(value: object) -> object:
    """JSON эталона может хранить 2, а Pydantic — 2.0: это одно значение поля."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        return {k: _numeric_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_numeric_values(v) for v in value]
    return value


def plan_groups(data: dict, *, reference: bool = False) -> dict:
    groups = plan_records(data)
    for name in ("semester", "part", "group", "profile", "trajectory"):
        groups[f"relation.discipline_{name}"] = [
            record(e.get("line", e.get("source", {}).get("line")), e[name])
            for e in data["entries"]
            if e.get(name) is not None
        ]
    groups["record.discipline_complete"] = [
        record(
            *(
                json.dumps(
                    _numeric_values(_field_value(e, name) if reference else e.get(name)),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for name in ENTRY_FIELDS
            )
        )
        for e in data["entries"]
    ]
    return groups


def _sources(plan: CurriculumPlan, gold: dict) -> dict:
    expected = [(e["line"], gold["sheet"], e["code"], e["name"]) for e in gold["entries"]]
    actual = [(e.source.line, e.source.section, e.code, e.name) for e in plan.entries]
    correct = sum(
        (Counter(record(*r) for r in expected) & Counter(record(*r) for r in actual)).values()
    )
    return {
        "correct": correct,
        "total": len(gold["entries"]),
        "accuracy": correct / len(gold["entries"]) if gold["entries"] else None,
        "literal_identity_matches": sum((Counter(expected) & Counter(actual)).values()),
        "normalization": "Сопоставление названий без регистра, ё/е, вида тире и лишних пробелов",
    }


def evaluate(data_dir: Path, repeats: int = 5) -> dict:
    if repeats < 1:
        raise ValueError("Число повторов должно быть положительным")
    result = {"documents": {}, "repeats": repeats}
    for name, archive in (
        ("plan_xlsx", "Учебный план.zip"),
        ("matrix_xlsx", "Матрица компетенций.zip"),
    ):
        source, member = _member(data_dir, archive)
        gold_path = (
            data_dir
            / "evaluation"
            / ("plan_xlsx_gold_full.json" if name == "plan_xlsx" else "matrix_gold_full.json")
        )
        gold_bytes = gold_path.read_bytes()
        gold = json.loads(gold_bytes)
        if not gold.get("entries" if name == "plan_xlsx" else "rows"):
            raise ValueError(f"{gold_path}: пустой эталон не является замером качества")
        digest = hashlib.sha256(source).hexdigest()
        if name == "plan_xlsx" and gold.get("source_sha256") != digest:
            raise ValueError("Хеш XLSX-плана отличается от зафиксированного эталона")
        timings, outputs = [], []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{name}.xlsx"
            path.write_bytes(source)
            for _ in range(repeats):
                started = time.perf_counter()
                doc = extract_plan_file(path) if name == "plan_xlsx" else extract_matrix(path)
                timings.append(time.perf_counter() - started)
                outputs.append(hashlib.sha256(doc.model_dump_json().encode()).hexdigest())
        fields = score_plan(doc, gold) if name == "plan_xlsx" else score_matrix(doc, gold)
        groups = compare_groups(
            plan_groups(doc.model_dump())
            if name == "plan_xlsx"
            else matrix_records(doc.model_dump()),
            plan_groups(gold, reference=True) if name == "plan_xlsx" else matrix_records(gold),
        )
        raw_sheets = read_xlsx_xml(source)
        formulas = sum(len(s.formulas) for s in raw_sheets if s.state == "visible")
        no_cache = sum(
            pos not in s.cells for s in raw_sheets if s.state == "visible" for pos in s.formulas
        )
        errors = sum(len(s.errors) for s in raw_sheets if s.state == "visible")
        result["documents"][name] = {
            "archive": archive,
            "member": member,
            "source_sha256": digest,
            "gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
            "reference_method": gold.get("annotation", gold.get("note", "локальный эталон")),
            "fields": {
                n: {
                    "correct": fields.correct[n],
                    "total": fields.total[n],
                    "accuracy": fields.accuracy(n),
                }
                for n in sorted(fields.total)
            },
            "field_micro_accuracy": fields.overall,
            "field_correct": sum(fields.correct.values()),
            "field_total": sum(fields.total.values()),
            "groups": groups,
            "field_metric_passed": fields.passed(),
            "group_metric_passed": all(s["f1"] >= 0.95 for s in groups.values()),
            "missing": fields.missing,
            "mismatches": fields.mismatches,
            "count_errors": fields.count_errors,
            "warnings": doc.warnings,
            "seconds_parse_and_extract": {
                "runs": timings,
                "median": statistics.median(timings),
                "min": min(timings),
                "max": max(timings),
            },
            "deterministic": len(set(outputs)) == 1,
            "output_sha256": outputs[-1],
            "formulas": {"total": formulas, "missing_cache": no_cache, "errors": errors},
        }
        if name == "plan_xlsx":
            item = result["documents"][name]
            item["hours"] = {
                "correct": fields.hour_cells_ok,
                "total": fields.hour_cells,
                "accuracy": fields.hours_cell_accuracy,
            }
            item["sources"] = _sources(doc, gold)
            item["uncoded_entries"] = sum(e.code is None for e in doc.entries)
    return result


def _provenance() -> dict:
    root = Path(__file__).resolve().parents[3]
    paths = [
        p
        for folder in ("extraction", "parsing", "eval")
        for p in (root / "src/edu_doc_agent" / folder).glob("*.py")
        if " " not in p.name
    ]
    return {
        "python": sys.version,
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("openpyxl", "pydantic", "yargy")
        },
        "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree": "измеряется рабочее дерево; код определяется хешами файлов",
        "code_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)
        },
    }


def _passed(document: dict) -> bool:
    return (
        document["field_metric_passed"]
        and document["group_metric_passed"]
        and document["deterministic"]
        and document.get("sources", {}).get("accuracy", 1) >= 0.95
    )


def markdown_report(result: dict, baseline: dict | None = None) -> str:
    lines = [
        "# Качество извлечения XLSX-документов",
        "",
        "Дата замера: 21 ноября 2026 года.",
        "",
        "Полные записи сравниваются один-к-одному; дубли дают FP. Метрики относятся к двум "
        "локальным файлам одной программы. Вызовов LLM нет.",
        "",
        "## Сущности, связи и полные записи",
        "",
        "| Документ / группа | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, doc in result["documents"].items():
        for group, s in doc["groups"].items():
            lines.append(
                f"| {name} / {group} | {s['tp']} | {s['fp']} | {s['fn']} | "
                f"{s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} |"
            )
    lines += [
        "",
        "## Поля",
        "",
        "| Документ / поле | Верно | Всего | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, doc in result["documents"].items():
        for field, s in doc["fields"].items():
            lines.append(
                f"| {name} / {field} | {s['correct']} | {s['total']} | {s['accuracy']:.4f} |"
            )
        lines.append(
            f"| **{name} / micro** | **{doc['field_correct']}** | "
            f"**{doc['field_total']}** | **{doc['field_micro_accuracy']:.4f}** |"
        )
    if baseline:
        lines += [
            "",
            "## До и после исправления",
            "",
            "Эталон и исходник не менялись. Слово «Семестр» в названии группы выбора "
            "ошибочно сбрасывало часть и группу у 75 записей. Классификация именованных "
            "контекстов теперь предшествует распознаванию семестра.",
            "",
            "| Показатель XLSX-плана | До | После |",
            "| --- | ---: | ---: |",
        ]
        before = baseline["documents"]["plan_xlsx"]
        after = result["documents"]["plan_xlsx"]
        for name in ("part", "group"):
            a, b = before["fields"][name], after["fields"][name]
            lines.append(
                f"| {name} | {a['correct']}/{a['total']} ({a['accuracy']:.4f}) | "
                f"{b['correct']}/{b['total']} ({b['accuracy']:.4f}) |"
            )
        for name in (
            "record.discipline_complete",
            "relation.discipline_part",
            "relation.discipline_group",
        ):
            lines.append(
                f"| F1 {name} | {before['groups'][name]['f1']:.4f} | "
                f"{after['groups'][name]['f1']:.4f} |"
            )
    lines += ["", "## Воспроизводимость и диагностика", ""]
    for name, doc in result["documents"].items():
        timing = doc["seconds_parse_and_extract"]
        lines += [
            f"- **{name}:** источник SHA-256 `{doc['source_sha256']}`; "
            f"эталон `{doc['gold_sha256']}`.",
            f"  Парсинг и извлечение: медиана {timing['median']:.3f} с, "
            f"диапазон {timing['min']:.3f}–{timing['max']:.3f} с "
            f"({result['repeats']} повторов); детерминизм: {doc['deterministic']}.",
            f"  Формул: {doc['formulas']['total']}; без кэша: "
            f"{doc['formulas']['missing_cache']}; ошибок: {doc['formulas']['errors']}; "
            f"предупреждений извлекателя: {len(doc['warnings'])}.",
        ]
        if "hours" in doc:
            lines.append(
                f"  Именованные ячейки часов: {doc['hours']['correct']}/{doc['hours']['total']}; "
                f"строки/листы источников: {doc['sources']['correct']}/{doc['sources']['total']}; "
                f"записей без кода в исходнике: {doc['uncoded_entries']}."
            )
    lines += [
        "",
        "## Методика и ограничения",
        "",
        "- Эталон XLSX-плана зафиксирован до первого сравнения. Он собран отдельным "
        "reference reader через ZIP/XML, без openpyxl, парсера и экстрактора. "
        "Области листа и правила контекста заданы по исходнику; хеш запрещает применять "
        "эту разметку к другой версии файла.",
        "- Использован существующий полный локальный эталон матрицы. Его происхождение "
        "описано в eval/matrix_gold.py: независимое чтение XML.",
        "- Независимой полной проверки человеком и теста на новой программе нет. "
        "Это локальная сверка с программно подготовленным эталоном и регрессия "
        "на документах разработки, а не оценка обобщения.",
        "- Entity-level дисциплина сравнивает код и название; контекст отдельно входит "
        "в отношения и record.discipline_complete (все 16 полей PlanEntry). "
        "Перестановка часов по видам должна снижать полевую метрику и полноту записи.",
        "- Для источников сравниваются исходная строка, лист, код и нормализованное название. "
        "Типографские варианты тире не считаются ошибкой координат.",
        "- Для разреженной матрицы отдельно учитываются положительные связи: "
        "высокая доля верных нулевых ячеек не заменяет relation-F1.",
        "- Формулы не пересчитывались: проверяются сохранённые значения. Ошибки "
        "и отсутствие кэша проверяются также синтетическими регрессиями.",
        "- DOCX и XLSX плана разных версий не сравниваются как один эталон. "
        "Междокументное сопоставление и span-level NER здесь не оцениваются.",
        "- Полные исходники, эталоны, ответы и ошибки остаются в data/evaluation. "
        "JSON сохраняет базовый Git-коммит и SHA-256 каждого файла кода рабочего дерева.",
        "",
        "## Повторение замера",
        "",
        "```bash",
        "uv run python -m edu_doc_agent.eval.xlsx_documents --prepare-plan-reference",
        "uv run python -m edu_doc_agent.eval.xlsx_documents \\",
        "  --output data/evaluation/xlsx_quality_after.json \\",
        "  --report docs/extraction-xlsx-evaluation.md",
        "```",
        "",
        "Первый запуск не вызывает извлекатель и не перезаписывает отличающийся эталон. "
        "Нужны оба локальных архива и matrix_gold_full.json. Для таблицы до/после "
        "добавляется `--baseline data/evaluation/xlsx_quality_before.json`. Базовый JSON "
        "сохранён до исправления; команда проверяет неизменность хешей обоих исходников "
        "и эталонов. Порог каждой группы и поля — 0.95, детерминизм обязателен. "
        "Непрохождение даёт код 1, отсутствие данных — код 2.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR") or "data"))
    parser.add_argument("--prepare-plan-reference", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats должно быть положительным")
    try:
        if args.prepare_plan_reference:
            source, _ = _member(args.data_dir, "Учебный план.zip")
            target = args.data_dir / "evaluation" / "plan_xlsx_gold_full.json"
            gold = build_plan_reference(source)
            if target.exists():
                if json.loads(target.read_text()) != gold:
                    raise ValueError(
                        "Существующий эталон отличается: автоматическая перезапись запрещена"
                    )
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Эталон зафиксирован: {target}")
            return 0
        if args.output is None:
            parser.error("для замера требуется --output")
        baseline = json.loads(args.baseline.read_text()) if args.baseline else None
        result = evaluate(args.data_dir, args.repeats)
        result["provenance"] = _provenance()
        if baseline:
            for name, doc in result["documents"].items():
                for field in ("source_sha256", "gold_sha256"):
                    if doc[field] != baseline["documents"][name][field]:
                        raise ValueError("До/после требует одинаковые исходники и эталоны")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(markdown_report(result, baseline), encoding="utf-8")
        for name, doc in result["documents"].items():
            print(
                f"{name}: поля {doc['field_correct']}/{doc['field_total']}, "
                f"accuracy={doc['field_micro_accuracy']:.4f}, "
                f"passed={_passed(doc)}"
            )
        return 0 if all(_passed(d) for d in result["documents"].values()) else 1
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Замер невозможен: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
