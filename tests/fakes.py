"""Тестовые двойники для unit-тестов."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from edu_doc_agent.dao import ActiveRunConflictError, Run, Session

ACTIVE_STATUSES = {"queued", "running", "waiting_for_input"}


class InMemorySessionRunDAO:
    """Реализация интерфейса SessionRunDAO в памяти, без подключения к PostgreSQL."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.runs: dict[UUID, Run] = {}

    async def create_queued_run(self, session_id: str, run_id: UUID | None = None) -> Run:
        now = datetime.now(UTC)
        self.sessions.setdefault(
            session_id,
            Session(id=session_id, status="active", created_at=now, updated_at=now),
        )
        if any(
            r.session_id == session_id and r.status in ACTIVE_STATUSES for r in self.runs.values()
        ):
            raise ActiveRunConflictError(session_id)

        run = Run(
            id=run_id or uuid4(),
            session_id=session_id,
            status="queued",
            reconcile_attempts=0,
            created_at=now,
            updated_at=now,
        )
        self.runs[run.id] = run
        return run

    def _transition(self, run_id: UUID, allowed: tuple[str, ...], new_status: str) -> bool:
        run = self.runs.get(run_id)
        if run is None or run.status not in allowed:
            return False
        run.status = new_status
        run.updated_at = datetime.now(UTC)
        if new_status == "running":
            run.last_heartbeat_at = run.updated_at
        return True

    async def mark_failed_if_queued(self, run_id: UUID) -> bool:
        return self._transition(run_id, ("queued",), "failed")

    async def claim_run_for_execution(self, run_id: UUID) -> bool:
        return self._transition(run_id, ("queued",), "running")

    async def mark_run_completed(self, run_id: UUID) -> bool:
        return self._transition(run_id, ("running",), "completed")

    async def mark_run_failed(self, run_id: UUID) -> bool:
        return self._transition(run_id, ("queued", "running"), "failed")

    async def requeue_running_run(self, run_id: UUID) -> bool:
        return self._transition(run_id, ("running",), "queued")

    async def get_run(self, session_id: str, run_id: UUID) -> Run | None:
        run = self.runs.get(run_id)
        return run if run is not None and run.session_id == session_id else None
