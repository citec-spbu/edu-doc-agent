# Тесты API, очереди и worker

Обычный запуск не требует Redis/Docker и не читает настоящий `.env`:

```bash
uv run --no-sync pytest
```

Асинхронные сценарии используют `asyncio.run`, дополнительный pytest-плагин
не нужен. PyYAML для проверки Compose уже установлен через dev-зависимость
pre-commit.

## Известные дефекты PR #29

Регрессии из ревью отмечены `pr_review` и выполняются как обычные обязательные
тесты. На них нет `xfail`: падение завершает pytest с ненулевым кодом и блокирует
обязательную проверку `test` в GitHub Actions.

- [P1: публикация Redis](https://github.com/citec-spbu/edu-doc-agent/pull/29#discussion_r4478019738).
- [P2: deadline и ошибки очереди](https://github.com/citec-spbu/edu-doc-agent/pull/29#discussion_r4478029967).
- [P3: конфликт постановки](https://github.com/citec-spbu/edu-doc-agent/pull/29#discussion_r4478047239).

Две дополнительные регрессии lifespan (закрытие при исключении и повторный
startup) сохраняют `xfail(strict=True, raises=AssertionError)`. Они не относятся
к опубликованным замечаниям ревью и пока не блокируют CI. После исправления
дефекта XPASS ломает прогон: нужно снять соответствующую отметку.

Предлагаемый контракт тестов: конфликт создания задания → `409` с текстовым
`detail`; ошибка/таймаут Redis → `503` либо `504` с текстовым `detail`.
Окончательный выбор между `503` и `504` остаётся за реализацией API.
Тест молчащего Redis задаёт защитный предел **2 секунды**: deadline приложения
должен быть меньше. Если согласован другой предел, обновить его в тесте.
Timeout не запускает автоматическую повторную постановку: тест имитирует запись
задания, ответ на которую потерян.

Для отдельного запуска обязательных проверок ревью (нужен тестовый Redis):

```bash
TEST_REDIS_URL=redis://127.0.0.1:6379/15 \
  uv run --no-sync pytest -m pr_review --run-integration -ra
```

Пока замечания ревью не исправлены, этот прогон и обычный `make test` падают.
Для воспроизведения дополнительных известных падений lifespan без xfail:

```bash
uv run --no-sync pytest tests/unit --runxfail
```

## Настоящие Redis и ARQ

Явно указать адрес отдельного тестового Redis:

```bash
TEST_REDIS_URL=redis://127.0.0.1:6379/15 \
  uv run --no-sync pytest tests/integration/test_queue.py --run-integration
```

Без адреса тесты пропускаются; при указанном, но недоступном Redis — падают.
Каждый сценарий использует уникальную очередь. Очистка удаляет только собственную
очередь, heartbeat и ключи известных тестовых run ID; FLUSHDB/FLUSHALL не используются.
Worker запускается в burst-режиме с конфигурацией приложения. Трёхсекундная
пауза заглушки подменяется только в модуле задачи, без изменения polling ARQ.
Отдельный локальный smoke запускает настоящие процессы Uvicorn и ARQ CLI,
делает localhost HTTP-запрос и дожидается результата реальной задачи. Его
временные модули задают только уникальное имя очереди; production-код не меняется.

## Контейнерный smoke

Заранее собрать образ приложения и подготовить образы БД/Redis:

```bash
docker build -t edu-doc-agent:tests .
docker pull pgvector/pgvector:pg17
docker pull redis:7-alpine
TEST_APP_IMAGE=edu-doc-agent:tests \
  uv run --no-sync pytest tests/integration/test_compose_smoke.py --run-compose
```

Тест не строит и не скачивает образы. Он создаёт отдельный Compose-проект на
основе [основной конфигурации](../compose.yaml): синтетические настройки,
временное хранилище, отдельные тома и случайные localhost-порты API/UI.
Настоящий `.env` не подключается. Публикация портов БД/Redis убрана в тестовой
конфигурации; безопасность исходной публикации проверяется отдельно в unit-тесте.
Проверяются health, Swagger, UI, HTTP → ARQ → результат и логи worker.
В `finally` удаляются только контейнеры и тома уникального тестового проекта.

## Локальная инструкция запуска

[Замечание P2](https://github.com/citec-spbu/edu-doc-agent/pull/29#discussion_r4478035619)
требует обновления README и docs; тесты не исправляют инструкцию автоматически.
Ручная приёмка обновлённой инструкции на отдельном тестовом окружении:

1. Поднять `db redis`; проверить localhost-адрес в `REDIS_URL`.
2. Запустить `make run-api`, `make run-worker` и `make run-ui` в отдельных терминалах.
3. Отправить POST `/api/v1/sessions/local-synthetic/process`; получить `202 queued`.
4. Убедиться, что worker завершил задачу с тем же `run_id`, UI и `/docs` доступны.
5. Проверить, что при отсутствии Redis startup API завершается с ошибкой.

## CI

В [workflow CI](../.github/workflows/ci.yml) выполняются обе группы проверок:

- Job `unit-integration` поднимает PostgreSQL и Redis как service containers, задаёт
  `TEST_REDIS_URL` и запускает unit- и Redis-интеграционные тесты с
  `--run-integration` и покрытием ветвей. Compose smoke в этом job пропускается.
- Job `docker-build` собирает один образ `edu-doc-agent:ci` из текущего checkout
  и загружает его в Docker runner через `load: true`. Затем скачивает готовые
  образы PostgreSQL/Redis и запускает Compose smoke с `--run-compose` и
  `TEST_APP_IMAGE=edu-doc-agent:ci`. Образ используется для API, worker и UI.
- Итоговый job `test` зависит от `unit-integration` и `docker-build` и выполняется
  с `if: always()`. Он успешен только при результате `success` у обоих jobs;
  ошибка, отмена или пропуск любого из них приводят к ошибке итоговой проверки.

Сборка образа и smoke выполняются в одном job. Публикация образа в registry
не требуется. Активный GitHub ruleset
[protect-main](https://github.com/citec-spbu/edu-doc-agent/rules/24883623)
требует проверки `lint` и `test` от GitHub Actions. Сохранено имя обязательного
status check `test`: менять правила репозитория для этой схемы не требуется.
Блокировка заработает для PR после публикации этих изменений и запуска CI.
Исключения `xfail` в остальных тестах не отключаются глобально.

Локальный `make test` сохраняет быстрый режим по составу проверок:
Redis-интеграция и Compose smoke запускаются только с явными флагами и
настройками, приведёнными выше. Обязательные unit-регрессии ревью выполняются
и локально, и в CI.
