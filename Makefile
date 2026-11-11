.PHONY: help sync lint fmt test test-llm cov run-api run-ui run-worker up up-db down logs clean

help:  ## Показать список целей
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync:  ## Установить/обновить зависимости
	uv sync

lint:  ## Проверить стиль без правок
	uv run ruff check .
	uv run ruff format --check .

fmt:  ## Автоформатирование
	uv run ruff check . --fix
	uv run ruff format .

test:  ## Юнит-тесты (без LLM)
	uv run pytest

test-llm:  ## Тесты с настоящей LLM (нужен LLM_API_KEY)
	uv run pytest --run-llm

cov:  ## Тесты с покрытием
	uv run pytest --cov=edu_doc_agent --cov-report=term-missing

run-api:  ## API локально с автоперезагрузкой
	uv run uvicorn edu_doc_agent.main:app --reload

run-ui:  ## UI локально
	uv run streamlit run ui/app.py

run-worker:  ## ARQ-воркер локально
	uv run arq edu_doc_agent.worker.settings.WorkerSettings

up:  ## Вся система в Docker
	docker compose up --build

up-db:  ## Только база в Docker (для локальной разработки)
	docker compose up -d db

down:  ## Остановить Docker
	docker compose down

logs:  ## Логи api
	docker compose logs -f api

clean:  ## Почистить кэши
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
