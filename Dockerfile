FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# 1) Зависимости — слой кэшируется, пока не меняются pyproject.toml и uv.lock
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# 2) Код и вспомогательные ресурсы
COPY README.md ./
COPY src ./src
COPY ui ./ui
COPY rules ./rules
COPY prompts ./prompts
RUN uv sync --locked --no-dev

COPY alembic.ini ./
COPY alembic ./alembic

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

CMD ["uvicorn", "edu_doc_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
