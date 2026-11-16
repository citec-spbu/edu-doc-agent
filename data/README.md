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
