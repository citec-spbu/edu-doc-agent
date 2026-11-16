# Извлечение документов образовательной программы

Задача #31 расширяет извлечение РПД на учебный план, общую характеристику,
ЦКМВ и матрицу компетенций. Числа и коды извлекаются правилами; свободный текст
характеристики можно дополнительно обработать через LLM.

## Использование

```python
from pathlib import Path

from edu_doc_agent.extraction.characteristic import extract_characteristic
from edu_doc_agent.extraction.ckmv import extract_ckmv
from edu_doc_agent.extraction.crossdoc import check_documents
from edu_doc_agent.extraction.curriculum import extract_plan
from edu_doc_agent.extraction.matrix import extract_matrix
from edu_doc_agent.parsing import parse_document

plan = extract_plan(parse_document(Path("plan.docx")).to_markdown(), "plan.docx")
char_md = parse_document(Path("characteristic.docx")).to_markdown()
characteristic = extract_characteristic(char_md, "characteristic.docx")
ckmv = extract_ckmv(parse_document(Path("ckmv.docx")).to_markdown(), "ckmv.docx")
matrix = extract_matrix(Path("matrix.xlsx"))
warnings = check_documents(plan, characteristic=characteristic, ckmv=ckmv, matrix=matrix)
```

Функции `extract_plan_file`, `extract_ckmv_file`, `extract_characteristic_file`
принимают **Markdown-файлы**, уже полученные парсером. `extract_matrix` читает
первый лист XLSX. XLSX-версия самого учебного плана пока не поддерживается.

```python
from edu_doc_agent.llm import OpenAICompatibleClient

client = OpenAICompatibleClient.from_settings()
characteristic = extract_characteristic(char_md, "characteristic.docx", llm=client)
print(characteristic.model_dump_json(indent=2))
```

Без `llm` сетевых вызовов нет, `characteristic.llm` равен `None`. С клиентом
в `llm` появляются роли, должности, траектории, проекты, нормативные акты,
принципы миссии, профили и средства индивидуализации. Табличные поля сохраняются.
Используются три задачи: аннотация, миссия, раздел 5 вместе с подразделами.
Каждая задача получает собственный раздел и JSON-схему. Невалидный ответ
повторяется не более двух раз; сбой одной задачи не отменяет остальные.

Цитата LLM должна дословно присутствовать в переданном тексте раздела после
нормализации Markdown; склейки фрагментов с многоточиями не принимаются.
Наличие цитаты подтверждает место в документе, но само по себе не доказывает
правильность смысловой интерпретации. Неподтверждённые элементы отбрасываются.
Предупреждения доступны и в `llm.warnings`, и в общих `warnings` характеристики.

## Источники и сверки

У сущностей есть `source`: имя документа, раздел, номер строки и исходная строка
`quote`. Нумерация Markdown начинается с 1. Для XLSX `line` означает строку листа,
`section` — имя листа; у столбцов дополнительно есть `column`.

Значения титула остаются строками. Их источники находятся в `sources` под именами
полей; у характеристики этот словарь также содержит аннотацию и миссию.
`qualification_sources` соответствует списку квалификаций по порядку.
Источники итоговых строк матрицы хранятся в `total_sources` по подписи строки.
LLM-сущности дополнительно сохраняют свою проверенную цитату в `quote`.

Внутренние проверки документов помещают расхождения в их `warnings`.
`check_documents` возвращает **междокументные** предупреждения: коды и названия
компетенций, реквизиты программы, соответствие строк матрицы учебному плану.
Для полного отчёта нужно учитывать оба списка. Расхождение требует проверки
человеком: документы могут относиться к разным версиям программы.

## Оценка качества

В `tests/fixtures/*_gold*.json` для новых типов документов лежат вымышленные
наборы, проверяющие метрики. Реальные эталоны хранятся локально:
`data/evaluation/{plan,ckmv,characteristic,matrix}_gold{,_full}.json` и
`data/evaluation/characteristic_llm_gold.json`. При `DATA_DIR` используется
его подкаталог `evaluation`. Эти файлы и ответы модели не коммитятся.

```bash
uv run python -m edu_doc_agent.eval.program_docs
uv run pytest tests/unit/test_characteristic_llm_real.py --run-llm -q -s
```

Первый запуск сравнивает четыре документа с локальными эталонами, выводит
междокументные расхождения и возвращает ненулевой код при непрохождении метрик.
Исходная характеристика отдельно сравнивается с эталоном исправленной версии
как диагностический пример; это сравнение не влияет на результат проверки.
При отсутствии обязательного документа или эталона замер сообщает об этом.

Тест LLM делает реальные вызовы только с `--run-llm`. Нужны настройки `LLM_*`,
архив характеристики и локальный эталон. Ответ и таблица метрик сохраняются
рядом с эталоном. Повторный подсчёт без обращения к модели:

```bash
uv run python -m edu_doc_agent.eval.characteristic_llm_gold \
  --gold data/evaluation/characteristic_llm_gold.json \
  --prediction data/evaluation/characteristic_llm_prediction.json
```

Метрика LLM — P/R/F1 по каждой группе и точность `kind` у найденных ролей и
проектов. Сопоставление один-к-одному по основам слов и вхождению, у принципов —
по цитате; дубли и лишние элементы штрафуются как FP. Порог каждого поля и
атрибутов — 0.8, ошибки извлечения также не позволяют пройти проверку.
Табличные метрики проверяют поля и контрольные счётчики с порогом 0.95.

Реальные документы относятся к одной ОП, эталоны не прошли полную проверку
человеком. Результаты не подтверждают качество на других шаблонах.
Исторические замеры: [документы ОП](extraction-evaluation-program-docs.md),
[целые РПД](extraction-evaluation-documents.md), [фрагменты РПД](extraction-evaluation.md).
