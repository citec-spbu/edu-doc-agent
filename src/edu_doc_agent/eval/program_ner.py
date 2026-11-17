"""Entity-level P/R/F1 документов ОП: полные записи и связи, а не доля верных ячеек.

Эталон не содержит символьных границ упоминаний: это оценка структурированного
извлечения, не span-level NER. Тексты сравниваются после нормализации регистра,
пробелов, ё/е и тире, без стемминга и подстрочного совпадения. Коды приводятся к
каноническому виду. Повторные упоминания сохраняются; лишние дубли считаются FP.
Связи оцениваются отдельно от сущностей. Числовые поля оцениваются прежними
полевыми метриками, а не подмешиваются к NER.
"""

import argparse
import hashlib
import io
import json
import os
import tempfile
import time
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from edu_doc_agent.eval.characteristic_gold import score_characteristic
from edu_doc_agent.eval.characteristic_llm_gold import (
    _key as stem_key,
)
from edu_doc_agent.eval.characteristic_llm_gold import expected_items, score_characteristic_llm
from edu_doc_agent.eval.ckmv_gold import score_ckmv
from edu_doc_agent.eval.llm_gold import Score
from edu_doc_agent.eval.matrix_gold import score_matrix
from edu_doc_agent.eval.plan_gold import score_plan
from edu_doc_agent.extraction.characteristic import extract_characteristic, function_key
from edu_doc_agent.extraction.characteristic_llm import CharacteristicLLM
from edu_doc_agent.extraction.ckmv import extract_ckmv
from edu_doc_agent.extraction.codes import canonical_code
from edu_doc_agent.extraction.curriculum import extract_plan
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.matrix import extract_matrix
from edu_doc_agent.parsing import parse_document

Record = tuple[str, ...]
Groups = dict[str, list[Record]]


def record(*values: object) -> Record:
    return tuple(normalize(str(v)) if v is not None else "" for v in values)


def score_records(predicted: Iterable[Record], expected: Iterable[Record]) -> Score:
    """Пересечение мультимножеств: каждая запись эталона может дать только один TP."""
    have, want = Counter(predicted), Counter(expected)
    tp = sum((have & want).values())
    return Score(tp=tp, fp=sum(have.values()) - tp, fn=sum(want.values()) - tp)


def plan_records(data: dict) -> Groups:
    return {
        "entity.competence": [
            record(canonical_code(c["code"]), c["name"]) for c in data["competences"]
        ],
        "entity.discipline": [record(e["code"], e["name"]) for e in data["entries"]],
        "entity.final_attestation": [record(a["name"]) for a in data["final_attestation"]],
        "relation.discipline_competence": [
            record(e.get("line", e.get("source", {}).get("line")), canonical_code(code))
            for e in data["entries"]
            for code in e["competences"]
        ],
        "relation.attestation_competence": [
            record(a["name"], canonical_code(code))
            for a in data["final_attestation"]
            for code in a["competences"]
        ],
    }


def ckmv_records(data: dict) -> Groups:
    groups: Groups = {
        name: []
        for name in (
            "entity.competence",
            "entity.indicator",
            "entity.outcome",
            "relation.role_competence",
            "relation.competence_indicator",
            "relation.indicator_outcome",
        )
    }
    for c in data["competences"]:
        code = canonical_code(c["code"])
        scope = c.get("line", c.get("source", {}).get("line"))
        groups["entity.competence"].append(record(code, c["name"]))
        if c["role"]:
            groups["relation.role_competence"].append(
                record(c["role_section"], c["role"], c["group"], code)
            )
        for i in c["indicators"]:
            ind = canonical_code(i["code"])
            groups["entity.indicator"].append(record(ind, i["name"], i["level"]))
            groups["relation.competence_indicator"].append(record(scope, code, ind))
            ind_scope = i.get("line", i.get("source", {}).get("line"))
            for o in i["outcomes"]:
                out = canonical_code(o["code"])
                groups["entity.outcome"].append(record(out, o["kind"], o["text"]))
                groups["relation.indicator_outcome"].append(record(scope, ind_scope, ind, out))
    return groups


def characteristic_records(data: dict) -> Groups:
    groups: Groups = {
        "entity.competence": [
            record(canonical_code(c["code"]), c["name"], c["group"]) for c in data["competences"]
        ],
        "entity.qualification": [record(q) for q in data["qualification"]],
        "entity.area": [record(a["code"], a["name"]) for a in data["areas"]],
        "entity.standard": [record(s["code"], s["name"]) for s in data["standards"]],
        "entity.employer": [record(e["text"]) for e in data["employers"]],
        "entity.object": [record(o["text"]) for o in data["objects"]],
        "entity.activity": [record(a["name"]) for a in data["activities"]],
        "entity.okved": [
            record(o["code"], o["name"]) for a in data["activities"] for o in a["okved"]
        ],
        "entity.labor_function": [
            record(function_key(f["code"]), f["name"])
            for m in data["mappings"]
            for f in m["functions"]
        ],
        "relation.competence_function": [
            record(canonical_code(m["competence_code"]), function_key(f["code"]))
            for m in data["mappings"]
            for f in m["functions"]
        ],
        "relation.activity_okved": [
            record(a["name"], o["code"]) for a in data["activities"] for o in a["okved"]
        ],
        "relation.activity_task": [
            record(a["name"], t["text"]) for a in data["activities"] for t in a["tasks"]
        ],
    }
    return groups


def matrix_records(data: dict) -> Groups:
    return {
        "entity.competence": [
            record(canonical_code(c if isinstance(c, str) else c["code"])) for c in data["columns"]
        ],
        "entity.discipline": [record(r["number"], r["discipline"]) for r in data["rows"]],
        "relation.discipline_competence": [
            record(r["number"], canonical_code(code))
            for r in data["rows"]
            for code in r["competences"]
        ],
    }


def metrics(score: Score) -> dict:
    return asdict(score) | {
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "attribute_accuracy": score.attribute_accuracy,
    }


def compare_groups(predicted: Groups, expected: Groups) -> dict:
    result = {}
    for name in sorted(predicted.keys() | expected.keys()):
        have, want = predicted.get(name, []), expected.get(name, [])
        result[name] = metrics(score_records(have, want)) | {
            "false_positive": list((Counter(have) - Counter(want)).elements()),
            "false_negative": list((Counter(want) - Counter(have)).elements()),
        }
    return result


def anchored_score(predicted: list[str], expected: list[str]) -> Score:
    """Мягкое совпадение требует начального слова эталона: «группы» не равно «руководитель группы».

    Это лексическая проверка полноты названия, а не проверка смысловой эквивалентности.
    Сначала строятся допустимые пары, затем максимальное паросочетание один-к-одному.
    """
    have, want = [stem_key(s) for s in predicted], [stem_key(s) for s in expected]
    edges = {
        p: [
            g
            for g, b in enumerate(want)
            if a
            and b
            and b.split()[0] in a.split()
            and (f" {a} " in f" {b} " or f" {b} " in f" {a} ")
        ]
        for p, a in enumerate(have)
    }
    owners: dict[int, int] = {}

    def match(p: int, seen: set[int]) -> bool:
        for g in edges[p]:
            if g in seen:
                continue
            seen.add(g)
            if g not in owners or match(owners[g], seen):
                owners[g] = p
                return True
        return False

    for p in range(len(have)):
        match(p, set())
    return Score(tp=len(owners), fp=len(have) - len(owners), fn=len(want) - len(owners))


def _member(archive: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(archive) as z:
        matches = [
            n
            for n in z.namelist()
            if unicodedata.normalize("NFC", n).endswith(suffix) and not n.startswith("__MACOSX/")
        ]
        if len(matches) != 1:
            raise ValueError(f"{archive.name}: ожидался один файл с суффиксом {suffix}")
        return z.read(matches[0])


def evaluate(data_dir: Path, llm_prediction: Path | None = None) -> dict:
    spec = {
        "plan": (
            "Учебный план.zip",
            "Report_26_5003_1.docx",
            extract_plan,
            plan_records,
            score_plan,
        ),
        "ckmv": ("ЦКМВ.zip", "ЦКМВ.docx", extract_ckmv, ckmv_records, score_ckmv),
        "characteristic": (
            "Характеристика.zip",
            "(ТОП)_corr.docx",
            extract_characteristic,
            characteristic_records,
            score_characteristic,
        ),
        "matrix": (
            "Матрица компетенций.zip",
            ".xlsx",
            extract_matrix,
            matrix_records,
            score_matrix,
        ),
    }
    result = {"documents": {}, "llm": {}}
    for name, (archive, suffix, extract, records, field_score) in spec.items():
        source = _member(data_dir / archive, suffix)
        gold_path = data_dir / "evaluation" / f"{name}_gold_full.json"
        gold_bytes = gold_path.read_bytes()
        gold = json.loads(gold_bytes)
        started = time.perf_counter()
        if name == "matrix":
            doc = extract(io.BytesIO(source), "matrix.xlsx")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"{name}.docx"
                path.write_bytes(source)
                md = parse_document(path).to_markdown()
            doc = extract(md, f"{name}.docx")
        duration = time.perf_counter() - started
        fields = field_score(doc, gold)
        result["documents"][name] = {
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
            "seconds_parse_and_extract": duration,
            "groups": compare_groups(records(doc.model_dump()), records(gold)),
            "fields": {
                n: {
                    "correct": fields.correct[n],
                    "total": fields.total[n],
                    "accuracy": fields.accuracy(n),
                }
                for n in fields.total
            },
            "field_metric_passed": fields.passed(),
            "warnings": doc.warnings,
        }
    prediction_path = (
        llm_prediction or data_dir / "evaluation" / "characteristic_llm_prediction.json"
    )
    gold_path = data_dir / "evaluation" / "characteristic_llm_gold.json"
    prediction_bytes, gold_bytes = prediction_path.read_bytes(), gold_path.read_bytes()
    prediction = CharacteristicLLM.model_validate_json(prediction_bytes)
    gold = json.loads(gold_bytes)
    expected = expected_items(gold)
    exact_predicted, exact_expected = {}, {}
    for name, items in expected.items():
        # Для принципов эталон задаёт фрагмент цитаты; это не разметка полной длины цитаты.
        if name == "principles":
            exact_predicted[name] = [record(e.quote) for e in getattr(prediction, name)]
            exact_expected[name] = [record(e["quote"]) for e in items]
        else:
            with_kind = name in ("roles", "projects")
            exact_predicted[name] = [
                record(e.name, e.kind if with_kind else None) for e in getattr(prediction, name)
            ]
            exact_expected[name] = [record(e["name"], e.get("kind")) for e in items]
    tolerant = score_characteristic_llm(prediction, gold)
    anchored = {}
    for name, items in expected.items():
        key = "quote" if name == "principles" else "name"
        anchored[name] = metrics(
            anchored_score(
                [getattr(e, key) for e in getattr(prediction, name)], [e[key] for e in items]
            )
        )
    result["llm"] = {
        "mode": "offline: сохранённый ответ последнего прогона, без вызовов модели",
        "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
        "exact": compare_groups(exact_predicted, exact_expected),
        "tolerant": {name: metrics(s) for name, s in tolerant.scores.items()},
        "anchored": anchored,
        "warnings": prediction.warnings,
    }
    return result


def markdown_report(result: dict) -> str:
    lines = [
        "# Качество извлечения сущностей и связей",
        "",
        f"Проверяемая ревизия: `{result.get('revision', 'не указана')}`.",
        "",
        "Сравнение полных нормализованных записей один-к-одному; лишние дубли дают FP.",
        "Разметки символьных границ нет: это entity-level, а не span-level NER.",
        "",
        "| Документ / группа | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    totals = {"entity": Score(), "relation": Score()}
    for doc_name, doc in result["documents"].items():
        for name, s in doc["groups"].items():
            lines.append(
                f"| {doc_name} / {name} | {s['tp']} | {s['fp']} | {s['fn']} | "
                f"{s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} |"
            )
            totals[name.split(".")[0]].add(Score(tp=s["tp"], fp=s["fp"], fn=s["fn"]))
    for name, s in totals.items():
        lines.append(
            f"| **Micro {name}** | {s.tp} | {s.fp} | {s.fn} | "
            f"{s.precision:.3f} | {s.recall:.3f} | {s.f1:.3f} |"
        )
    lines += [
        "",
        "## LLM характеристики: сохранённый ответ",
        "",
        "Exact: точное равенство нормализованных названий и kind; у принципов — цитат.",
        "Мягкая метрика: основы слов, вхождение и сохранение начального слова; kind отдельно.",
        "",
        "Дополнительная проверка требует сохранения начального слова эталона в названии.",
        "Обе метрики теперь отклоняют неполные должности без главного слова.",
        "| Группа | Exact TP / FP / FN | Exact F1 | Мягкий F1 | С сохранением начала | kind |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, s in result["llm"]["exact"].items():
        soft = result["llm"]["tolerant"][name]
        kind = f"{soft['attribute_accuracy']:.3f}" if soft["attr_total"] else "—"
        anchored_f1 = result["llm"]["anchored"][name]["f1"]
        lines.append(
            f"| {name} | {s['tp']} / {s['fp']} / {s['fn']} | {s['f1']:.3f} | "
            f"{soft['f1']:.3f} | {anchored_f1:.3f} | {kind} |"
        )
    lines += [
        "",
        "## Границы оценки",
        "",
        "- Одна образовательная программа, по одному документу каждого типа.",
        "- Правила отлаживались на этих документах; "
        "промпт характеристики уточнялся по этому эталону.",
        "- Эталон создан моделью, без полной независимой проверки человеком.",
        "- Различия exact и мягкой метрики требуют ручной оценки формулировок и границ цитат.",
        "- Качество междокументного сопоставления отдельно не измерено: "
        "эталона связей между файлами нет.",
        "- JSON с ошибками и хешами исходников остаётся в data/evaluation и не коммитится.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR") or "data"))
    parser.add_argument("--revision", default="не указана")
    parser.add_argument("--llm-prediction", type=Path, help="Сохранённый ответ выбранного прогона")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.data_dir, args.llm_prediction)
    except FileNotFoundError as exc:
        parser.exit(2, f"Нет локального файла: {exc.filename}\n")
    result["revision"] = args.revision
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = markdown_report(result)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
