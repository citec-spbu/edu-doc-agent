# Тесты API, очереди и worker

Обычный запуск не требует Redis/Docker и не читает настоящий `.env`:

```bash
uv run --no-sync pytest
```

Асинхронные сценарии используют `asyncio.run`, дополнительный pytest-плагин
не нужен. PyYAML для проверки Compose уже установлен через dev-зависимость
pre-commit.

## Контракты и ограничения backend

Реализован контур FastAPI → Redis → ARQ worker. Контракт ответов эндпоинта
описан в [архитектуре backend](../docs/backend-architecture.md#асинхронная-обработка-документов-api-arq).

HTTP-дедлайн постановки — **2 секунды**, включая случай зависшей очистки
Redis pipeline. Результат `enqueue_job() → None` также означает неподтверждённую
постановку и даёт `503`. При потере подтверждения задание могло сохраниться;
API делает одну попытку постановки на запрос. Внешняя отмена запроса
распространяется в очередь и не превращается в HTTP-ошибку.

Пул Redis, созданный приложением, закрывается после обычного выхода,
исключения и каждого повторного startup/shutdown; ссылка удаляется из
`app.state`. Внедрённый извне пул приложение не закрывает.

В PostgreSQL модели Session/Run ограничивают набор статусов и допускают только
один активный run (`queued`, `running`, `waiting_for_input`) на сессию.
API создаёт Run до постановки в очередь, worker фиксирует переходы
`queued → running → completed/failed`. Повторный POST при активном Run даёт `409`;
сбой постановки переводит в `failed` только ещё не взятый worker запуск
исходного поколения 0. Запоздалая компенсация не меняет восстановленное поколение.

## Настоящий PostgreSQL 17 и Alembic

Для DAO указать DSN **отдельной тестовой БД**, а не рабочего приложения:

```bash
TEST_DATABASE_URL=postgresql+psycopg://edu:edu@127.0.0.1:5432/edu \
  uv run --no-sync pytest tests/integration/dao --run-integration
```

Без `--run-integration` эти тесты пропускаются. С флагом отсутствие DSN,
недоступная БД или версия PostgreSQL, отличная от 17, приводят к ошибке.
Каждый тест создаёт случайную собственную схему, передаёт её через search_path
в подключения и Alembic, затем удаляет только её. Нужны права на создание схем;
URL-регрессия также создаёт и удаляет собственную временную LOGIN-роль, поэтому
тестовый пользователь должен быть superuser. CI использует отдельный service container.
Рабочий `.env` дочерние процессы Alembic не читают.

Проверяются ограничения при INSERT/UPDATE, конкуренция независимых транзакций
с commit/rollback, освобождение слота и возобновление с сохранением run_id,
канонические статусы, отказ на пустую строку при INSERT/UPDATE, SQL-каскад и поля моделей.
Завершение get_db при обычном выходе и исключении проверяется через возврат
соединения в пул (checkin), затем независимая сессия подтверждает откат незавершённой записи.
После ORM-изменения и commit updated_at читается на том же объекте без refresh,
совпадает с БД; created_at сохраняется. Транзакция обновления A начинается до
создания и commit записи в независимой B: возврат к onupdate=now() обнаруживается.
Это контракт обычного ORM flush/commit, не raw SQL или bulk UPDATE.

Один набор проверок ограничений выполняется на схемах из `alembic upgrade head`
и `Base.metadata.create_all()`, включая конкуренцию, SQL DEFAULT счётчика `0`
и запрет отрицательных значений при INSERT/UPDATE. Сравнение metadata включает
серверные DEFAULT. Ревизия ограничений проверяется с валидными данными,
при downgrade/reapply и при атомарном отказе из-за существующего отрицательного значения.

Миграции проверяются с нуля, при повторном применении, после downgrade и поверх
первой ревизии с данными. Корректирующая ревизия не очищает данные: при двух
активных runs откатывается целиком, сохраняя строки и прежний номер ревизии.
Online/offline URL-регрессии используют синтетический пароль с percent-encoding.
Offline генерирует SQL с ожидаемой ревизией при заведомо недоступном адресе БД;
online действительно применяет миграции. В обоих режимах и при контролируемом
online-сбое проверяется отсутствие исходного/кодированного пароля и полного DSN в выводе.

## Настоящие PostgreSQL, Redis и ARQ

### Матрица жизненного цикла Session/Run (#36)

В [test_session_runs.py](integration/test_session_runs.py) API и Runner работают
с настоящим PostgreSQL в отдельной схеме после Alembic. Enqueue подменён:
это позволяет управлять порядком подтверждения очереди и переходов worker.
Нужен только `TEST_DATABASE_URL`, Redis для этого файла не требуется:

```bash
TEST_DATABASE_URL=postgresql+psycopg://edu:edu@127.0.0.1:5432/edu \
  uv run --no-sync pytest tests/integration/test_session_runs.py --run-integration
```

| Сценарий | Проверка |
| --- | --- |
| Commit до enqueue | Независимое соединение видит Session и queued Run внутри enqueue |
| Два POST для новой сессии | 202/409, одна Session, один активный Run, один enqueue |
| Активные и терминальные статусы | Три активных статуса запрещают новый Run, три терминальных разрешают |
| Посторонний IntegrityError | Настоящий конфликт PK не превращается в 409; транзакция откатывается |
| Сбой enqueue | Timeout, RedisError, None, внутренняя ошибка и deadline; failed и успешный новый POST |
| Компенсация раньше worker | Поздняя job пропускается без вызова бизнес-функции |
| Worker раньше компенсации | running и completed сохраняются |
| Две попытки одного Run | Один атомарный захват и один вызов бизнес-функции |
| Параллельные разные Runs | Работа при пуле из одного соединения; ошибка одного не затрагивает другой |
| Исключение бизнес-функции | Исходный объект исключения выходит после commit failed |
| GET | Шесть статусов, идентификаторы и timestamps из БД, 404 для отсутствующего/чужого Run, без Redis |

Отмена до/после записи статуса, повторная отмена, защита чужого Run и сохранение
терминального исхода покрыты [unit-регрессиями Runner](unit/worker/test_runner_cancellation.py).
В `test_queue.py::test_worker_shutdown_leaves_job_queued_for_retry` настоящий ARQ
после остановки worker повторно выполняет тот же Run: проверяются queued в БД,
успех нового worker и completed через GET. Для этого сценария нужны обе тестовые DSN.
В `test_queue.py::test_arq_timeout_fails_run_and_allows_a_new_post` настоящий
таймаут ARQ проверяется на попытках 1 и 2: результат job — `TimeoutError` без retry,
канонический статус — `failed`, следующий POST принимается с новым run_id.
Используется короткий `WorkerSettings.job_timeout`; внутренний лимит Runner не задаётся.
В [test_runner_failures.py](unit/worker/test_runner_failures.py) проверяются исходное
исключение бизнес-функции при сбое записи `failed`, ошибка записи `completed`
и логирование сбоя обработки отмены. Unit-тест API закрепляет логирование
неудачной компенсации с сохранением исходного HTTP-ответа на сбой enqueue.

В [test_runner_heartbeat.py](unit/worker/test_runner_heartbeat.py) проверяются
остановка логики по локальному сроку lease при ошибках и зависании БД, продление
lease подтверждёнными heartbeat, задержка ответа на захват и гонка отмены Runner
с завершением heartbeat. Независимый таймер отменяет логику, даже пока очистка
SQLAlchemy ожидает завершения; задержанное подтверждение захвата вызывает
первый пульс сразу, если его интервал уже прошёл. В [test_runner_failures.py](unit/worker/test_runner_failures.py)
проверяется также контракт произвольного `Awaitable`, включая `Future`, его отмену
и синхронное исключение при вызове бизнес-функции.

В [test_run_generation.py](unit/worker/test_run_generation.py) проверяются
недействительность старых поколений, повторные отмены cron до enqueue и после
принятия job Redis, доставка в именованную очередь и обход 101 queued-запуска
партиями по 100. Ожидающая доставка повторяет то же поколение без расхода лимита;
подтверждённые восстановления сохраняют защиту от poison pill.
Проверки настоящего SQL для резервирования, подтверждения и обхода находятся в
[test_session_runs.py](integration/test_session_runs.py); новая миграция признака
`reconcile_pending` и её откат проверяются в
[test_migrations.py](integration/dao/test_migrations.py).
В той же миграции добавлен `last_queue_check_at` для порядка обхода. Интеграции
проверяют через GET сохранение публичного `updated_at` при heartbeat, проверке
очереди и резервировании следующего поколения, а также запоздалую компенсацию POST
после подтверждённой доставки поколения 1. Сохранённая ошибка настоящего ARQ
подтверждает доставку pending-поколения и позволяет продолжить восстановление.

В [test_worker_config.py](unit/test_worker_config.py) проверяются `NaN`
и бесконечные значения, предел 365 дней для обоих порогов и значения сразу выше
него. Максимальный допустимый порог проверяется всеми тремя запросами DAO
в [test_session_runs.py](integration/test_session_runs.py).

Отдельные unit-сценарии приёмки #47 находятся в
[test_reconciler.py](unit/worker/test_reconciler.py), а интеграционные с настоящими
PostgreSQL, Redis и ARQ — в [test_queue.py](integration/test_queue.py):

| Сценарий | Интеграционный тест |
| --- | --- |
| Устаревший running и одна бизнес-обработка нового поколения | `test_reconciler_recovers_stale_running_through_real_arq` |
| Потерянный queued и доставка в именованную очередь | `test_reconciler_recovers_lost_queued_through_real_arq` |
| Исчерпание лимита для running и queued | `test_reconciler_exhausted_limit_fails_and_allows_new_http_post` |
| Возвращение старого владельца после захвата новым worker | `test_reconciler_returning_old_runner_cannot_modify_the_recovered_owner` |
| Потеря ответа Redis до/после фактической записи | `test_reconciler_unknown_redis_write_outcome_preserves_generation_and_executes_once` |
| Сохранённая ошибка ARQ после потери ответа Redis и сбоя захвата run | `test_saved_arq_failure_confirms_delivery_and_allows_recovery` |

Каждый интеграционный сценарий проверяет освобождение сессии через HTTP:
после терминального GET следующий POST получает `202` с новым `run_id`.
Изолированные очереди и схемы очищаются только по собственным идентификаторам,
включая job ID восстановлений. Целевой запуск (нужны обе тестовые DSN):

```bash
uv run --no-sync pytest tests/unit/worker tests/unit/test_worker_config.py \
  tests/unit/api/test_sessions.py tests/integration/dao \
  tests/integration/test_session_runs.py tests/integration/test_queue.py \
  --run-integration
```

CI запускает эти тесты существующей командой `pytest --run-integration`.
Регистрация cron с `run_at_startup=True` проверяется в unit-тесте конфигурации
worker; это постановка при старте, а не барьер перед обработкой обычных задач.
В сценариях отдельной job, таймаута и остановки/повтора worker cron отключён
через `cron_jobs=[]`, чтобы служебная задача не попадала в счётчики и очередь
проверяемого запуска. В сценариях восстановления Reconciler вызывается явно.

Для сбора покрытия backend указывайте каталоги исходников:

```bash
uv run --no-sync pytest tests/integration/test_session_runs.py \
  tests/integration/test_queue.py::test_worker_shutdown_leaves_job_queued_for_retry \
  tests/integration/test_queue.py::test_arq_timeout_fails_run_and_allows_a_new_post \
  tests/unit/worker tests/unit/dao tests/unit/api/test_sessions.py \
  --run-integration --cov=src/edu_doc_agent/dao --cov=src/edu_doc_agent/worker \
  --cov=src/edu_doc_agent/api --cov-branch --cov-report=term-missing
```

Перед запуском задать обе тестовые DSN, как описано ниже. В локальном окружении
с coverage 7.16.2 и SQLAlchemy 2.1.8 выбор вложенных модулей именами
(`--cov=edu_doc_agent.dao.repositories` и аналогично для `worker.runner`/`api.router`)
вызывает повторный импорт SQLAlchemy при поиске источников coverage. Python-код
и Cython-расширение получают разные объекты enum `InternalTraversal`, из-за чего
ORM INSERT падает с `TypeError: 'InternalTraversal' object is not callable`.
Каталоги в `--cov` сохраняют branch coverage и исключают этот предварительный
импорт. Используемый в CI `--cov=edu_doc_agent` также не требует импорта вложенных
пакетов при поиске источников.

### Запуск контура очереди

Явно указать адреса отдельных тестовых PostgreSQL 17 и Redis:

```bash
TEST_DATABASE_URL=postgresql+psycopg://edu:edu@127.0.0.1:5432/edu \
  TEST_REDIS_URL=redis://127.0.0.1:6379/15 \
  uv run --no-sync pytest tests/integration/test_queue.py --run-integration
```

Без адреса Redis тесты пропускаются; при указанном, но недоступном Redis — падают.
Для БД требуется явный `TEST_DATABASE_URL`. Каждый тест получает собственную
схему с миграциями Alembic; API и worker используют настоящую БД в этой схеме.
Подключения не переиспользуются между event loop. После теста схема удаляется.
Каждый сценарий использует уникальную очередь. Очистка удаляет только собственную
очередь, heartbeat и ключи известных тестовых run ID; FLUSHDB/FLUSHALL не используются.
Worker запускается в burst-режиме с конфигурацией приложения. Трёхсекундная
пауза заглушки подменяется только в модуле задачи, без изменения polling ARQ.
Отдельный smoke запускает настоящие процессы Uvicorn и ARQ CLI,
делает localhost HTTP-запрос и дожидается результата реальной задачи. Его
временные модули задают уникальное имя очереди, а окружение передаёт DSN той же
тестовой схемы; production-код не меняется.

В том же файле есть два fault-сценария с собственным временным контейнером
Redis: остановка и приостановка после успешного запроса. Оба проверяют `503`
при отказе и `202` после восстановления через тот же пул API. Эти тесты требуют
Docker daemon и заранее доступного образа `redis:7-alpine`; образы не скачиваются.
Чужой Redis из `TEST_REDIS_URL` не останавливается. Контейнер использует
случайный localhost-порт, persistence отключён, удаление выполняется в `finally`.
Для запуска только этих двух сценариев внешний Redis не нужен:

```bash
TEST_DATABASE_URL=postgresql+psycopg://edu:edu@127.0.0.1:5432/edu \
  uv run --no-sync pytest tests/integration/test_queue.py --run-integration \
  -k redis_recovers_after_outage
```

## Контейнерный smoke

Smoke сверяет хэши исходников внутри образа, ждёт доступности опубликованного
порта API и проверяет результат и логи с тем же `run_id`. Тестовые контейнеры
и тома изолированы от рабочего Compose; рабочая очередь не используется.

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
Из того же образа API запускается `alembic upgrade head`, затем номер ревизии
читается напрямую из PostgreSQL и через штатный async_session_maker внутри API и worker.
Их синтетический `env_file` содержит недоступный хостовый `DATABASE_URL` на
`127.0.0.1:1`: сброс в Compose должен вытеснить этот DSN и сохранить подключение к `db`.
Используются исходные POSTGRES_* из Compose с синтетическим паролем, содержащим
`@` и `%`, без подмены bind, движка или DATABASE_URL фабрики DAO.
Проверка свежести образа включает миграции и alembic.ini.
В `finally` удаляются только контейнеры и тома уникального тестового проекта.

Отдельный сценарий удерживает свежий PostgreSQL тестовым init-скриптом:
сокет уже доступен, TCP ещё нет. Он наблюдает неуспешный healthcheck в этой фазе,
затем освобождает инициализацию и выполняет ожидание `db` → одноразовый Alembic
без запуска остального стека, проверяя номер ревизии.
Ещё один сценарий передаёт Compose и локальному Alembic общий синтетический `.env`
с паролем в одинарных кавычках (`@`, `%`, буквальный `$NAME`) и заданным `NAME`
в окружении. Локальный Alembic действительно аутентифицируется и применяет миграции;
пароль не должен появляться в выводе. Каждый сценарий использует собственную БД/том.

## CI

В [workflow CI](../.github/workflows/ci.yml) выполняются обе группы проверок:

- Job `unit-integration` поднимает PostgreSQL и Redis как service containers, задаёт
  `TEST_DATABASE_URL`/`TEST_REDIS_URL` и запускает unit- и PostgreSQL/Redis-тесты с
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
Корпус РПД извлекается из исходных DOCX с геометрией таблиц. Прежние два `xfail`
для разной сетки шапки и тела сняты; оба сценария проверяются обычными assertions.

`make test` сохраняет быстрый режим по составу проверок: Redis-интеграция и Compose smoke запускаются только с явными флагами и настройками, приведёнными выше. Обязательные unit-регрессии ревью выполняются в CI.
