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

Для локального запуска нужны Python 3.12, uv, Docker с Compose и make.
Команды ниже рассчитаны на Bash (Linux/macOS или WSL).

```bash
git clone https://github.com/citec-spbu/edu-doc-agent.git
cd edu-doc-agent

# 1. Зависимости
uv sync
uv run pre-commit install

# 2. Окружение
cp .env.example .env       # заполнить LLM_API_KEY, DATA_DIR
# В .env задайте REDIS_URL=redis://127.0.0.1:6379/0 для локальных API и worker.

# 3. Инфраструктура (БД и Redis в Docker)
# Локальное дополнение публикует Redis только на loopback-интерфейсе.
docker compose -f compose.yaml -f - up -d --wait db redis <<'YAML'
services:
  redis:
    ports:
      - "127.0.0.1:6379:6379"
YAML
```

В основном [compose.yaml](compose.yaml) порт Redis не опубликован.
Дополнение выше передаётся через stdin и не создаёт файлов в репозитории.
Дождитесь готовности инфраструктуры: API подключается к Redis при старте.
Порты 5432 и 6379 должны быть свободны. Примените миграции командой `make migrate`.

В трёх отдельных терминалах из корня проекта запустите:

```bash
make run-api     # API: http://localhost:8000/docs
make run-worker  # Фоновый воркер ARQ для обработки задач
make run-ui      # UI: http://localhost:8501
```

Проверка постановки задачи:

```bash
curl -i -X POST http://localhost:8000/api/v1/sessions/local-demo/process
```

Ожидается `202 Accepted` с `run_id`, `session_id` и `status: queued`.
В терминале worker должно появиться сообщение о завершении с тем же `run_id`.
Сейчас выполняется задача-заглушка; это проверка связки API → Redis → worker.

Чтобы завершить локальный запуск, остановите API, worker и UI через `Ctrl+C`,
затем выполните `docker compose stop db redis`.

Полностью в Docker (как будет на демо), после остановки локальных процессов:

```bash
make migrate-docker && make up
```

`make migrate-docker` дожидается готовности PostgreSQL и запускает Alembic
в одноразовом контейнере из образа API с настройками БД из Compose.
Выполняйте эту команду перед запуском стека после добавления новых миграций.

Задавайте `POSTGRES_PASSWORD` в исходном виде. Локальные команды по умолчанию
подключаются к `localhost:5432` и учитывают явно заданный `DATABASE_URL`.
Контейнеры API, worker и Alembic собирают подключение к `db:5432` из `POSTGRES_*`.

Для буквального `$NAME` в общем `.env`, который читают Compose и Pydantic,
используйте одинарные кавычки:

```dotenv
POSTGRES_PASSWORD='synthetic@pass%40$NAME'
```

Для буквального `${NAME}` передавайте пароль через окружение процесса,
поскольку Pydantic может интерполировать эту форму в `.env` даже в кавычках:

```bash
export POSTGRES_PASSWORD='synthetic@pass%40${NAME}'
```

Запускайте Compose и локальные команды из того же окружения. Пароль из
`POSTGRES_*` передаётся в `URL.create()` без ручного URL-кодирования. Явный
`DATABASE_URL` сохраняет приоритет для локальных команд; в полном DSN
спецсимволы пароля нужно URL-кодировать.

В этом режиме API и worker используют `redis://redis:6379/0`, заданный в
[compose.yaml](compose.yaml); публиковать Redis на localhost не требуется.

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
