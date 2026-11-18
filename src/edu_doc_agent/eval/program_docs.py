"""Сводный замер правиловых экстракторов документов ОП на реальных документах из `data/`.

Запуск: `uv run python -m edu_doc_agent.eval.program_docs`. Для каждого типа документа (учебный
план, ЦКМВ, характеристика, матрица) извлечение сверяется с полным эталоном из `data/evaluation/`;
затем выполняется сквозная сверка документов между собой. Без `data/` печатает сообщение и выходит.
"""

import io
import sys

from edu_doc_agent.eval.characteristic_gold import (
    CHAR_GOLD_FULL,
    load_char_gold,
    score_characteristic,
)
from edu_doc_agent.eval.ckmv_gold import CKMV_GOLD_FULL, load_ckmv_gold, score_ckmv
from edu_doc_agent.eval.matrix_gold import MATRIX_GOLD_FULL, load_matrix_gold, score_matrix
from edu_doc_agent.eval.plan_gold import PLAN_GOLD_FULL, load_plan_gold, score_plan
from edu_doc_agent.extraction.characteristic import extract_characteristic
from edu_doc_agent.extraction.ckmv import extract_ckmv
from edu_doc_agent.extraction.crossdoc import check_documents, compare_matrix_with_plan
from edu_doc_agent.extraction.curriculum import extract_plan
from edu_doc_agent.extraction.matrix import extract_matrix


def main() -> int:
    from tests.real_data import (
        real_characteristic_markdown,
        real_ckmv_markdown,
        real_matrix_xlsx,
        real_plan_markdown,
    )

    gold_paths = (PLAN_GOLD_FULL, CKMV_GOLD_FULL, CHAR_GOLD_FULL, MATRIX_GOLD_FULL)
    if not all(p.exists() for p in gold_paths):
        print("нет локальных эталонов data/evaluation/*_gold_full.json — замер невозможен")
        return 1
    plan_md, ckmv_md, xlsx = real_plan_markdown(), real_ckmv_markdown(), real_matrix_xlsx()
    corrected, original = real_characteristic_markdown(True), real_characteristic_markdown(False)
    if not (plan_md and ckmv_md and xlsx and corrected):
        print("нет data/ (план, ЦКМВ, матрица, характеристика _corr) — замер невозможен")
        return 1

    plan = extract_plan(plan_md, "plan.docx")
    ckmv = extract_ckmv(ckmv_md, "ckmv.docx")
    matrix = extract_matrix(io.BytesIO(xlsx), "matrix.xlsx")
    char = extract_characteristic(corrected, "char_corr.docx")
    reports = {
        "учебный план": score_plan(plan, load_plan_gold(PLAN_GOLD_FULL)),
        "ЦКМВ": score_ckmv(ckmv, load_ckmv_gold(CKMV_GOLD_FULL)),
        "характеристика (_corr)": score_characteristic(char, load_char_gold(CHAR_GOLD_FULL)),
        "матрица": score_matrix(matrix, load_matrix_gold(MATRIX_GOLD_FULL)),
    }
    for name, r in reports.items():
        print(f"\n===== {name}: всего {sum(r.total.values())}, верно {sum(r.correct.values())}")
        print(r.summary())
    if original:
        print("\n===== исходная характеристика против эталона _corr (другая версия)")
        print(
            score_characteristic(
                extract_characteristic(original, "char.docx"), load_char_gold(CHAR_GOLD_FULL)
            ).summary()
        )

    print("\n===== сквозная сверка")
    cmp = compare_matrix_with_plan(matrix, plan)
    print(
        f"матрица↔план: строк {len(matrix.rows)}, сопоставлено {cmp.matched}, совпало {cmp.equal}, "
        f"различаются {len(cmp.different)}, без пары {len(cmp.unmatched)}"
    )
    warnings = check_documents(plan, characteristic=char, ckmv=ckmv, matrix=matrix)
    print(f"предупреждений check_documents: {len(warnings)}")
    for w in warnings:
        print(" -", w)
    return 0 if all(r.passed() for r in reports.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
