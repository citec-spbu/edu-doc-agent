"""Метрика качества извлечения общей характеристики ОП на размеченных эталонах.

Эталоны (`data/evaluation/characteristic_gold*.json`) получены по сырому Markdown, а не по выводу
экстрактора. Компетенции и строки раздела 3 сопоставляются по коду и порядку появления; списки
(области, стандарты, работодатели, виды деятельности) — по позиции. Для найденного сравнивается
каждое поле отдельно, ненайденное считается ошибкой по всем своим полям. Поля, которых нет в
эталоне (у ручного набора нет раздела компетенции и текстов задач), не оцениваются.
"""

import json
from collections import defaultdict
from pathlib import Path

from edu_doc_agent.eval.datasets import local_gold
from edu_doc_agent.eval.field_report import THRESHOLD, FieldReport
from edu_doc_agent.extraction.characteristic import ProgramCharacteristic, function_key
from edu_doc_agent.extraction.llm_extract import normalize

CHAR_GOLD = local_gold("characteristic_gold.json")  # титул, 1.x, часть компетенций, вручную
CHAR_GOLD_FULL = local_gold("characteristic_gold_full.json")  # всё, разбор сырых строк

__all__ = ["CHAR_GOLD", "CHAR_GOLD_FULL", "THRESHOLD", "load_char_gold", "score_characteristic"]

TITLE_FIELDS = (
    "group_code",
    "group_name",
    "direction_code",
    "direction",
    "level",
    "profile",
    "profile_en",
    "program_name",
    "program_name_en",
    "program_code",
    "study_form",
    "languages",
    "duration",
)
FUNCTION_FIELDS = ("code", "standard_code", "function_letter", "level", "name")


def load_char_gold(path: Path = CHAR_GOLD) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _eq(got: object, want: object) -> bool:
    if isinstance(got, str) or isinstance(want, str):
        return normalize(str(got or "")) == normalize(str(want or ""))
    return got == want


def _nth(seen: dict[str, int], key: str) -> int:
    n = seen[key]
    seen[key] += 1
    return n


def score_characteristic(doc: ProgramCharacteristic, gold: dict | None = None) -> FieldReport:
    gold = gold or load_char_gold()
    r = FieldReport()

    for name in TITLE_FIELDS:
        got, want = getattr(doc, name), gold["title"][name]
        r.add(f"title.{name}", _eq(got, want), f"{got!r} != {want!r}")

    r.add(
        "qualification",
        [normalize(q) for q in doc.qualification] == [normalize(q) for q in gold["qualification"]],
        f"{doc.qualification!r}",
    )
    _score_list(r, "area", doc.areas, gold["areas"], ("code", "name"))
    _score_list(r, "standard", doc.standards, gold["standards"], ("code", "name"))
    _score_list(r, "employer", doc.employers, gold["employers"], ("text",))
    if "text" in (gold["objects"][0] if gold.get("objects") else {}):
        _score_list(r, "object", doc.objects, gold["objects"], ("text",))
    _score_activities(r, doc, gold)
    _score_competences(r, doc, gold)
    _score_mappings(r, doc, gold)
    _check_totals(r, doc, gold)
    return r


def _score_list(r: FieldReport, label: str, got: list, want: list[dict], fields: tuple) -> None:
    for i, g in enumerate(want):
        item = got[i] if i < len(got) else None
        if item is None:
            r.missing.append(f"{label} #{i + 1}")
        for f in fields:
            value = getattr(item, f) if item is not None else None
            r.add(f"{label}.{f}", item is not None and _eq(value, g[f]), f"#{i + 1}: {value!r}")


def _score_activities(r: FieldReport, doc: ProgramCharacteristic, gold: dict) -> None:
    for i, g in enumerate(gold["activities"]):
        a = doc.activities[i] if i < len(doc.activities) else None
        if a is None:
            r.missing.append(f"вид деятельности #{i + 1}")
        r.add("activity.name", a is not None and _eq(a.name, g["name"]), f"#{i + 1}")
        got_okved = [(o.code, normalize(o.name)) for o in a.okved] if a else []
        want_okved = [(o["code"], normalize(o["name"])) for o in g["okved"]]
        r.add("activity.okved", got_okved == want_okved, f"{g['name']}: {got_okved}")
        if "tasks" in g:
            got_tasks = [normalize(t.text) for t in a.tasks] if a else []
            want_tasks = [normalize(t["text"]) for t in g["tasks"]]
            r.add("activity.tasks", got_tasks == want_tasks, f"{g['name']}: {len(got_tasks)} задач")
    if "task_counts" in gold["totals"]:
        got_counts = [len(a.tasks) for a in doc.activities]
        r.add("activity.task_counts", got_counts == gold["totals"]["task_counts"], f"{got_counts}")


def _score_competences(r: FieldReport, doc: ProgramCharacteristic, gold: dict) -> None:
    by_code = defaultdict(list)
    for c in doc.competences:
        by_code[c.code].append(c)
    seen: dict[str, int] = defaultdict(int)
    for g in gold["competences"]:
        found = by_code.get(g["code"], [])
        nth = _nth(seen, g["code"])
        if nth >= len(found):
            r.missing.append(f"компетенция {g['code']}")
        c = found[nth] if nth < len(found) else None
        for f in ("name", "group", "section"):
            if f in g:
                got = getattr(c, f) if c is not None else None
                r.add(f"competence.{f}", c is not None and _eq(got, g[f]), f"{g['code']}: {got!r}")
    if "group_counts" in gold:
        counts: dict[str, int] = defaultdict(int)
        for c in doc.competences:
            counts[c.group] += 1
        r.add("competence.group_counts", dict(counts) == gold["group_counts"], f"{dict(counts)}")


def _score_mappings(r: FieldReport, doc: ProgramCharacteristic, gold: dict) -> None:
    by_code = defaultdict(list)
    for m in doc.mappings:
        by_code[m.competence_code].append(m)
    seen: dict[str, int] = defaultdict(int)
    for g in gold["mappings"]:
        code = g["competence_code"]
        found = by_code.get(code, [])
        nth = _nth(seen, code)
        if nth >= len(found):
            r.missing.append(f"раздел 3: {code}")
            continue
        got = found[nth].functions
        r.add("mapping.n_functions", len(got) == len(g["functions"]), f"{code}: {len(got)}")
        for i, gf in enumerate(g["functions"]):
            f_got = got[i] if i < len(got) else None
            for f in FUNCTION_FIELDS:
                value = getattr(f_got, f) if f_got is not None else None
                ok = f_got is not None and (
                    function_key(str(value)) == function_key(str(gf[f]))
                    if f == "code"
                    else _eq(value, gf[f])
                )
                r.add(f"function.{f}", ok, f"{code} #{i + 1}: {value!r} != {gf[f]!r}")


def _check_totals(r: FieldReport, doc: ProgramCharacteristic, gold: dict) -> None:
    if not gold.get("complete"):
        return
    got = {
        "competences": len(doc.competences),
        "mappings": len(doc.mappings),
        "functions": sum(len(m.functions) for m in doc.mappings),
        "objects": len(doc.objects),
        "areas": len(doc.areas),
        "standards": len(doc.standards),
        "employers": len(doc.employers),
        "activities": len(doc.activities),
    }
    for name, want in gold["totals"].items():
        if got[name] != want:
            r.count_errors.append(f"{name}: {got[name]}, ожидалось {want}")
