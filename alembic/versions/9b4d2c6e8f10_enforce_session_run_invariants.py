"""Enforce session/run statuses and one active run per session.

Revision ID: 9b4d2c6e8f10
Revises: 504d7cc74177

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "9b4d2c6e8f10"
down_revision: str | Sequence[str] | None = "504d7cc74177"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Невалидные существующие данные должны прервать всю транзакцию миграции.
    # Автоматического удаления/перевода запусков в терминальный статус нет.
    op.create_check_constraint("ck_sessions_status", "sessions", "status IN ('active', 'archived')")
    op.create_check_constraint(
        "ck_runs_status",
        "runs",
        "status IN ('queued', 'running', 'waiting_for_input', 'completed', 'failed', 'cancelled')",
    )
    op.create_index(
        "uq_runs_active_session",
        "runs",
        ["session_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running', 'waiting_for_input')"),
    )


def downgrade() -> None:
    op.drop_index("uq_runs_active_session", table_name="runs")
    op.drop_constraint("ck_runs_status", "runs", type_="check")
    op.drop_constraint("ck_sessions_status", "sessions", type_="check")
