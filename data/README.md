# Локальные эталоны извлечения (#31)

В `evaluation/` хранятся реальные эталоны `plan_gold.json`, `plan_gold_full.json`,
`ckmv_gold.json`, `ckmv_gold_full.json`, `characteristic_gold.json`,
`characteristic_gold_full.json`, `matrix_gold.json`, `matrix_gold_full.json` и
`characteristic_llm_gold.json`. Они не входят в Git; без них проверки качества
реальных документов пропускаются. Архивы документов лежат рядом с `evaluation/`.
При заданном `DATA_DIR` используется его подкаталог `evaluation`.

`characteristic_llm_prediction.json` и `characteristic_llm_report.txt` — локальные
результаты ручного LLM-замера. Публиковать можно сводные метрики и ограничения
оценки без исходных документов и ответов модели.

`plan_xlsx_gold_full.json` — полный эталон XLSX-версии плана, отдельно от DOCX.
Он готовится через `python -m edu_doc_agent.eval.xlsx_documents --prepare-plan-reference`:
читается OOXML известного файла, без парсера и извлекателя. Версия ограничена
SHA-256 исходника; другой файл требует новой разметки. Независимой полной
проверки человеком нет. `xlsx_quality_before.json` и `xlsx_quality_after.json`
содержат метрики до/после исправления, ошибки, хеши исходников, эталонов и кода.
