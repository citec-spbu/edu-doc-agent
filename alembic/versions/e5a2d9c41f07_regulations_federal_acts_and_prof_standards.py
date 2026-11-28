"""Regulations: federal acts and professional standard functions (#50)

Revision ID: e5a2d9c41f07
Revises: c4e8a1f37b20

"""

from collections.abc import Sequence

from alembic import op

revision: str = "e5a2d9c41f07"
down_revision: str | Sequence[str] | None = "c4e8a1f37b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_DOC_TYPES = "doc_type IN ('fgos', 'spbu_standard', 'prof_standard')"
NEW_DOC_TYPES = "doc_type IN ('fgos', 'spbu_standard', 'prof_standard', 'federal_act')"
OLD_KINDS = "kind IN ('clause', 'competency', 'prof_standard_ref')"
NEW_KINDS = (
    "kind IN ('clause', 'competency', 'prof_standard_ref', 'prof_standard', "
    "'generalized_function', 'labor_function')"
)


def _replace(table: str, name: str, condition: str) -> None:
    op.drop_constraint(name, table, type_="check")
    op.create_check_constraint(name, table, condition)


def upgrade() -> None:
    _replace("regulation_documents", "ck_regulation_documents_doc_type", NEW_DOC_TYPES)
    _replace("regulation_fragments", "ck_regulation_fragments_doc_type", NEW_DOC_TYPES)
    _replace("regulation_fragments", "ck_regulation_fragments_kind", NEW_KINDS)


def downgrade() -> None:
    # новые документы и фрагменты не подходят под старые ограничения — удаляем их
    op.execute(
        "DELETE FROM regulation_fragments WHERE kind IN "
        "('prof_standard', 'generalized_function', 'labor_function')"
    )
    op.execute("DELETE FROM regulation_documents WHERE doc_type = 'federal_act'")
    _replace("regulation_fragments", "ck_regulation_fragments_kind", OLD_KINDS)
    _replace("regulation_fragments", "ck_regulation_fragments_doc_type", OLD_DOC_TYPES)
    _replace("regulation_documents", "ck_regulation_documents_doc_type", OLD_DOC_TYPES)
