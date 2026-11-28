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

# Нормативы для RAG (#32)

`regulations/` — в git только `manifest.yaml` (какие документы входят в корпус) и
`overrides.yaml` (ручные правки нарезки). Сами PDF кладутся рядом и в git не попадают:

- `regulations/fgos/*.pdf` — ФГОС ВО 3++ (выгрузки с fgosvo.ru);
- `regulations/spbu/*.pdf` — Образовательный стандарт СПбГУ (раздел «Образовательные
  стандарты» на spbu.ru).

Сборка корпуса (YAML-снимки по документам и `REPORT.md` в `data/corpus/`):

```bash
uv run python -m edu_doc_agent.retrieval.corpus.build data/regulations/manifest.yaml \
    --overrides data/regulations/overrides.yaml --out data/corpus
```

Загрузка в базу (PostgreSQL с pgvector и Ollama с моделью `bge-m3` должны быть запущены,
миграции применены: `uv run alembic upgrade head`):

```bash
uv run python -m edu_doc_agent.retrieval.corpus.load data/regulations/manifest.yaml \
    --overrides data/regulations/overrides.yaml
```

Повторная загрузка неизменённого корпуса ничего не пишет; эмбеддинги пересчитываются только
для изменившихся фрагментов. `--prune` удаляет из базы документы, которых нет в манифесте.

Проверка выдачи и оценка качества поиска (`golden.yaml` — вопросы и правильные пункты):

```bash
uv run python -m edu_doc_agent.retrieval.search "объем программы" --direction 02.03.02
uv run python -m edu_doc_agent.retrieval.evaluate data/regulations/golden.yaml
```

В `REPORT.md` — число пунктов, компетенций и профстандартов по документам и замечания:
пропуски в нумерации, УК/ОПК не подряд, строки таблиц без кода. Замечание чинится правкой
в `overrides.yaml`, исходный PDF не трогаем.
