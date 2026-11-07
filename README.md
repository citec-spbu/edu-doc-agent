# edu-doc-agent

![CI](https://github.com/citec-spbu/edu-doc-agent/actions/workflows/ci.yml/badge.svg)

Агентная ИИ-система интеллектуального анализа, автоматизированного формирования
и проверки учебно-методических документов образовательной программы.

Система анализирует пакет документов ОП (РПД, учебный план, характеристика программы,
наборы компетенций), связывает компетенции с дисциплинами и оценочными средствами,
проверяет соответствие ФГОС ВО и профстандартам, формирует консолидированный пакет
и протокол проверки с рекомендациями.

**Заказчик:** ОЦАД СПбГУ

**Кураторы:** Коваленко Лев, Севрюков С.Ю., И.С. Блеканов

## Быстрый старт

```bash
git clone https://github.com/citec-spbu/edu-doc-agent.git
cd edu-doc-agent

# 1. Зависимости
uv sync
uv run pre-commit install

# 2. Окружение
cp .env.example .env       # заполнить LLM_API_KEY, DATA_DIR

# 3. База данных (только она — в Docker)
docker compose up -d db

# 4. API и UI локально
make run-api   # http://localhost:8000/docs
make run-ui    # http://localhost:8501
```
Полностью в Docker (как будет на демо):
```bash
docker compose up --build
```

## Команды
См. `make help`. Основное: `make test`, `make lint`, `make fmt`, `make up-db`, `make run-api`, `make run-ui`, `make up`.


## Структура репозитория

```
src/edu_doc_agent/   основной код (api, orchestration, parsing, extraction, …)
ui/                  Streamlit-интерфейс
rules/               YAML-каталог правил проверки
prompts/             промпты как файлы
docs/                документация → GitHub Pages
tests/               unit, integration, fixtures (только синтетика)
notebooks/           исследования (nbstripout вычищает выводы)
data/                реальные документы — вне git
```

## Команда

| Роль | Имя | Контакты |
| --- | --- | --- |
| PM, тестировщик | Грязнов Эдуард | @apophis142 |
| Data Science | Подтуров Павел | @clerdmy |
| Data Science | Измайлов Никита | @sskkyyrreexx |
| Data Science | | |
| Backend | Епишкин Роман | @oxxytentacion |

## Документация

- [Разбор проекта](docs/project-review.md)
- [Архитектура](docs/architecture.md)
- [Как мы работаем](docs/contributing.md)
- [Решения (ADR)](docs/adr/index.md)

## Лицензия

**TBD**
