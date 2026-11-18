"""Конфликт активного run определяется диагностикой PostgreSQL, а не текстом."""

from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

from edu_doc_agent.dao.repositories import _is_active_run_conflict


@pytest.mark.parametrize(
    ("constraint_name", "expected"),
    [("uq_runs_active_session", True), ("runs_session_id_fkey", False), (None, False)],
)
def test_only_active_run_constraint_is_classified_as_conflict(
    constraint_name: str | None, expected: bool
) -> None:
    error = Exception("synthetic DETAIL: session_id=uq_runs_active_session")
    error.diag = SimpleNamespace(constraint_name=constraint_name)
    assert _is_active_run_conflict(IntegrityError("synthetic INSERT", {}, error)) is expected


def test_missing_diagnostics_does_not_fall_back_to_error_text() -> None:
    error = Exception("synthetic DETAIL: session_id=uq_runs_active_session")
    assert not _is_active_run_conflict(IntegrityError("synthetic INSERT", {}, error))
