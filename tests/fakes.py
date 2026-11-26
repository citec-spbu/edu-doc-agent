"""Тестовые двойники для unit-тестов."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from edu_doc_agent.dao import ActiveRunConflictError, ReconciledRun, Run, Session

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
            reconcile_pending=False,
            last_queue_check_at=None,
            created_at=now,
            updated_at=now,
        )
        self.runs[run.id] = run
        return run

    def _transition(
        self,
        run_id: UUID,
        allowed: tuple[str, ...],
        new_status: str,
        *,
        expected_reconcile_attempts: int | None = None,
    ) -> bool:
        run = self.runs.get(run_id)
        if run is None or run.status not in allowed:
            return False
        if (
            expected_reconcile_attempts is not None
            and run.reconcile_attempts != expected_reconcile_attempts
        ):
            return False
        if run.status != new_status:
            run.updated_at = datetime.now(UTC)
            if new_status == "queued":
                run.last_queue_check_at = None
        run.status = new_status
        if new_status == "running":
            run.last_heartbeat_at = run.updated_at
        return True

    async def mark_failed_if_queued(self, run_id: UUID) -> bool:
        return self._transition(run_id, ("queued",), "failed", expected_reconcile_attempts=0)

    async def claim_run_for_execution(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int = 0,
    ) -> int | None:
        if not self._transition(
            run_id, ("queued",), "running", expected_reconcile_attempts=expected_reconcile_attempts
        ):
            return None
        self.runs[run_id].reconcile_pending = False
        return self.runs[run_id].reconcile_attempts

    async def touch_heartbeat(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int | None = None,
    ) -> bool:
        run = self.runs.get(run_id)
        if run is None or run.status != "running":
            return False
        if (
            expected_reconcile_attempts is not None
            and run.reconcile_attempts != expected_reconcile_attempts
        ):
            return False
        run.last_heartbeat_at = datetime.now(UTC)
        return True

    async def mark_run_completed(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int | None = None,
    ) -> bool:
        return self._transition(
            run_id,
            ("running",),
            "completed",
            expected_reconcile_attempts=expected_reconcile_attempts,
        )

    async def mark_run_failed(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int | None = None,
    ) -> bool:
        return self._transition(
            run_id,
            ("queued", "running"),
            "failed",
            expected_reconcile_attempts=expected_reconcile_attempts,
        )

    async def requeue_running_run(
        self,
        run_id: UUID,
        *,
        expected_reconcile_attempts: int | None = None,
    ) -> bool:
        return self._transition(
            run_id,
            ("running",),
            "queued",
            expected_reconcile_attempts=expected_reconcile_attempts,
        )

    async def get_run(self, session_id: str, run_id: UUID) -> Run | None:
        run = self.runs.get(run_id)
        return run if run is not None and run.session_id == session_id else None

    @staticmethod
    def _reconciled(run: Run) -> ReconciledRun:
        return ReconciledRun(run.id, run.session_id, run.reconcile_attempts, run.reconcile_pending)

    async def fail_exhausted_stale_running_runs(
        self, *, stale_running_seconds: float, max_reconcile_attempts: int, limit: int = 100
    ) -> list[UUID]:
        cutoff = datetime.now(UTC) - timedelta(seconds=stale_running_seconds)
        candidates = [
            run
            for run in self.runs.values()
            if run.status == "running"
            and (run.last_heartbeat_at or run.updated_at) < cutoff
            and run.reconcile_attempts >= max_reconcile_attempts
        ][:limit]
        for run in candidates:
            self._transition(run.id, ("running",), "failed")
        return [run.id for run in candidates]

    async def requeue_stale_running_runs(
        self, *, stale_running_seconds: float, max_reconcile_attempts: int, limit: int = 100
    ) -> list[ReconciledRun]:
        cutoff = datetime.now(UTC) - timedelta(seconds=stale_running_seconds)
        candidates = [
            run
            for run in self.runs.values()
            if run.status == "running"
            and (run.last_heartbeat_at or run.updated_at) < cutoff
            and run.reconcile_attempts < max_reconcile_attempts
        ][:limit]
        for run in candidates:
            self._transition(run.id, ("running",), "queued")
            run.reconcile_attempts += 1
            run.reconcile_pending = True
        return [self._reconciled(run) for run in candidates]

    async def list_stale_queued_runs(
        self, *, stale_queued_seconds: float, limit: int = 100
    ) -> list[ReconciledRun]:
        cutoff = datetime.now(UTC) - timedelta(seconds=stale_queued_seconds)
        candidates = sorted(
            (
                r
                for r in self.runs.values()
                if r.status == "queued" and (r.last_queue_check_at or r.updated_at) < cutoff
            ),
            key=lambda r: (r.last_queue_check_at or r.updated_at, r.id),
        )[:limit]
        return [self._reconciled(run) for run in candidates]

    async def bump_stale_queued_run(
        self, run_id: UUID, *, expected_reconcile_attempts: int
    ) -> ReconciledRun | None:
        run = self.runs[run_id]
        if run.reconcile_pending or not self._transition(
            run_id, ("queued",), "queued", expected_reconcile_attempts=expected_reconcile_attempts
        ):
            return None
        run.reconcile_attempts += 1
        run.reconcile_pending = True
        run.last_queue_check_at = datetime.now(UTC)
        return self._reconciled(run)

    async def fail_stale_queued_run(
        self, run_id: UUID, *, expected_reconcile_attempts: int
    ) -> bool:
        if self.runs[run_id].reconcile_pending:
            return False
        return self._transition(
            run_id, ("queued",), "failed", expected_reconcile_attempts=expected_reconcile_attempts
        )

    async def confirm_reconcile_enqueue(
        self, run_id: UUID, *, expected_reconcile_attempts: int
    ) -> bool:
        run = self.runs[run_id]
        if run.reconcile_attempts != expected_reconcile_attempts or not run.reconcile_pending:
            return False
        run.reconcile_pending = False
        return True

    async def mark_queued_run_checked(
        self, run_id: UUID, *, expected_reconcile_attempts: int
    ) -> bool:
        updated = self._transition(
            run_id, ("queued",), "queued", expected_reconcile_attempts=expected_reconcile_attempts
        )
        if updated:
            self.runs[run_id].last_queue_check_at = datetime.now(UTC)
        return updated
