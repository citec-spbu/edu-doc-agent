"""Метрика качества извлечения матрицы компетенций на размеченных эталонах.

Эталоны (`data/evaluation/matrix_gold*.json`) получены независимым чтением xlsx (XML напрямую, без
`openpyxl`). Строки сопоставляются по номеру строки матрицы; для найденной сравнивается каждое поле
(метаданные и набор компетенций), отдельно считаются все ячейки «строка × компетенция» и итоги.
"""

import json
from pathlib import Path

from edu_doc_agent.eval.datasets import local_gold
from edu_doc_agent.eval.field_report import THRESHOLD, FieldReport
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.matrix import CompetenceMatrix

MATRIX_GOLD = local_gold("matrix_gold.json")  # 5 строк вручную
MATRIX_GOLD_FULL = local_gold("matrix_gold_full.json")  # вся матрица, независимое чтение

__all__ = ["MATRIX_GOLD", "MATRIX_GOLD_FULL", "THRESHOLD", "load_matrix_gold", "score_matrix"]

ROW_FIELDS = ("semester", "profile", "part", "credits", "discipline", "competences")


def load_matrix_gold(path: Path = MATRIX_GOLD) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _same(field: str, got: object, want: object) -> bool:
    if field == "discipline":
        return normalize(str(got)) == normalize(str(want))
    return got == want


def score_matrix(matrix: CompetenceMatrix, gold: dict | None = None) -> FieldReport:
    gold = gold or load_matrix_gold()
    r = FieldReport()
    by_number = {row.number: row for row in matrix.rows}
    codes = [c.code for c in matrix.columns]

    for g in gold["rows"]:
        row = by_number.get(g["number"])
        if row is None:
            r.missing.append(f"строка №{g['number']}")
            for f in ROW_FIELDS:
                r.add(f"row.{f}", False)
            for _ in gold.get("columns", []):
                r.add("cells", False)
            continue
        for f in ROW_FIELDS:
            got = getattr(row, f)
            r.add(f"row.{f}", _same(f, got, g[f]), f"№{g['number']}: {got!r} != {g[f]!r}")
        if "columns" in gold:  # каждая ячейка «строка × компетенция»
            want = set(g["competences"])
            have = set(row.competences)
            for code in gold["columns"]:
                r.add("cells", (code in have) == (code in want), f"№{g['number']} {code}")
    for label, want_totals in gold["totals"].items():
        got_totals = matrix.totals.get(label)
        for code, want in want_totals.items():
            got = got_totals.get(code) if got_totals else None
            r.add("totals", got == want, f"{label} {code}: {got!r} != {want!r}")

    if gold.get("columns"):
        r.add("columns", codes == gold["columns"], f"{len(codes)} столбцов")
    elif "columns_count" in gold:
        ok = len(codes) == gold["columns_count"] and codes[0] == gold["first_column"]
        r.add("columns", ok and codes[-1] == gold["last_column"], f"{len(codes)} столбцов")
    if "totals_counts" in gold:
        c = gold["totals_counts"]
        got = {
            "columns": len(codes),
            "rows": len(matrix.rows),
            "ones": sum(len(x.competences) for x in matrix.rows),
        }
        for k, want in c.items():
            if got[k] != want:
                r.count_errors.append(f"{k}: {got[k]}, ожидалось {want}")
    return r
