"""Create regulation documents and fragments for RAG (ADR-0005)

Revision ID: c4e8a1f37b20
Revises: edd8a4ac3a60

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "c4e8a1f37b20"
down_revision: str | Sequence[str] | None = "edd8a4ac3a60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Расширение — на всю базу; при откате не удаляем: им могут пользоваться и другие таблицы.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "regulation_documents",
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("edition", sa.String(), nullable=False),
        sa.Column("doc_type", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("order", sa.String(), nullable=True),
        sa.Column("direction_code", sa.String(), nullable=True),
        sa.Column("level", sa.String(), nullable=True),
        sa.Column("source_file", sa.String(), nullable=False),
        sa.Column(
            "loaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "doc_type IN ('fgos', 'spbu_standard', 'prof_standard')",
            name="ck_regulation_documents_doc_type",
        ),
        sa.PrimaryKeyConstraint("doc_id", "edition"),
    )
    op.create_table(
        "regulation_fragments",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("edition", sa.String(), nullable=False),
        sa.Column("part", sa.Integer(), nullable=False),
        sa.Column("doc_type", sa.String(), nullable=False),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("number", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("codes", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("parent", sa.String(), nullable=True),
        sa.Column("group", sa.String(), nullable=True),
        sa.Column("changed_by", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('russian', path || ' ' || text)", persisted=True),
            nullable=True,
        ),
        sa.CheckConstraint(
            "kind IN ('clause', 'competency', 'prof_standard_ref')",
            name="ck_regulation_fragments_kind",
        ),
        sa.CheckConstraint("part >= 0", name="ck_regulation_fragments_part_nonnegative"),
        sa.ForeignKeyConstraint(
            ["doc_id", "edition"],
            ["regulation_documents.doc_id", "regulation_documents.edition"],
            name="fk_regulation_fragments_document",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", "edition", "part"),
    )
    op.create_index("ix_regulation_fragments_doc", "regulation_fragments", ["doc_id", "edition"])
    op.create_index(
        "ix_regulation_fragments_tsv", "regulation_fragments", ["tsv"], postgresql_using="gin"
    )
    op.create_index(
        "ix_regulation_fragments_codes", "regulation_fragments", ["codes"], postgresql_using="gin"
    )


def downgrade() -> None:
    op.drop_index("ix_regulation_fragments_codes", table_name="regulation_fragments")
    op.drop_index("ix_regulation_fragments_tsv", table_name="regulation_fragments")
    op.drop_index("ix_regulation_fragments_doc", table_name="regulation_fragments")
    op.drop_table("regulation_fragments")
    op.drop_table("regulation_documents")
