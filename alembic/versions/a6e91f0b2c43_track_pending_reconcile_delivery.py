"""Track reserved recovery generations awaiting delivery.

Revision ID: a6e91f0b2c43
Revises: 9b4d2c6e8f10
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a6e91f0b2c43"
down_revision: str | Sequence[str] | None = "9b4d2c6e8f10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Уже существующие поколения сохраняют прежний учёт попыток.
    op.add_column(
        "runs",
        sa.Column("reconcile_pending", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("runs", "reconcile_pending")
