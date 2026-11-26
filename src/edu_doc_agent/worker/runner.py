"""Инфраструктурная обёртка (Runner) для запуска бизнес-логики/графа с фиксацией статусов в БД."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from typing import Any
from uuid import UUID

from edu_doc_agent.dao import SessionRunDAO

logger = logging.getLogger(__name__)

# Бизнес-функция принимает только session_id (thread_id) и не знает про run_id и ARQ.
BusinessCallable = Callable[[str], Awaitable[Any]]


async def _finish_status_update[T](operation: Coroutine[Any, Any, T]) -> T:
    """Дожидается операции БД даже при повторной отмене, затем распространяет отмену."""
    task = asyncio.create_task(operation)
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc
        except Exception as exc:
            if cancellation is None:
                raise
            logger.exception("Status update failed while worker cancellation was pending")
            raise cancellation from exc
        else:
            if cancellation is not None:
                raise cancellation
            return result


async def _stop_heartbeat(task: asyncio.Task[None] | None) -> None:
    """Отменяет heartbeat_task и дожидается её завершения даже при повторном CancelledError."""
    if task is None:
        return
    task.cancel()
    current = asyncio.current_task()
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            # shield может завершиться одновременно с heartbeat. Сначала
            # сохраняем отмену самого Runner, затем проверяем дочернюю задачу.
            if current is not None and current.cancelling():
                cancellation = exc
            if task.done():
                break
    try:
        with suppress(asyncio.CancelledError):
            task.result()
    finally:
        if cancellation is not None:
            raise cancellation


class SessionRunner:
    """Управляет жизненным циклом Run вокруг вызова бизнес-логики."""

    def __init__(
        self,
        dao: SessionRunDAO,
        heartbeat_interval: float | None = 30.0,
        *,
        lease_timeout: float = 120.0,
    ) -> None:
        if lease_timeout <= 0:
            raise ValueError("lease_timeout must be positive")
        self._dao = dao
        self._heartbeat_interval = heartbeat_interval
        self._lease_timeout = lease_timeout

    async def _heartbeat_loop(
        self,
        run_id: UUID,
        interval: float,
        expected_reconcile_attempts: int | None,
        on_lease_lost: Callable[[], None] | None = None,
        *,
        lease_deadline: float,
        on_lease_renewed: Callable[[float], None],
    ) -> None:
        """Продлевает lease; без подтверждения БД останавливает работу к локальному сроку."""
        loop = asyncio.get_running_loop()
        op_timeout = min(interval, 10.0)
        next_pulse = lease_deadline - self._lease_timeout + interval
        while True:
            try:
                # Срок охватывает и ожидание следующего пульса, и запрос БД.
                async with asyncio.timeout_at(lease_deadline):
                    await asyncio.sleep(max(0, next_pulse - loop.time()))
                    request_started = loop.time()
                    next_pulse = request_started + interval
                    try:
                        async with asyncio.timeout(op_timeout):
                            updated = await self._dao.touch_heartbeat(
                                run_id,
                                expected_reconcile_attempts=expected_reconcile_attempts,
                            )
                    except Exception:
                        logger.warning(
                            "Не удалось обновить heartbeat для run_id=%s",
                            run_id,
                            exc_info=True,
                        )
                        continue
            except TimeoutError:
                logger.warning(
                    "Локальный lease истёк без подтверждения heartbeat: run_id=%s",
                    run_id,
                )
                if on_lease_lost is not None:
                    on_lease_lost()
                return

            if not updated or loop.time() >= lease_deadline:
                logger.warning(
                    "Остановка heartbeat: lease потерян или run_id=%s больше не в 'running'",
                    run_id,
                )
                if on_lease_lost is not None:
                    on_lease_lost()
                return
            # Начало запроса предшествует записи clock_timestamp() в БД.
            # Отсчёт от получения ответа мог бы продлить локальный lease сверх БД.
            lease_deadline = request_started + self._lease_timeout
            on_lease_renewed(lease_deadline)

    async def execute(
        self,
        session_id: str,
        run_id: UUID | str,
        logic_callable: BusinessCallable,
        job_try: int = 1,
        max_tries: int = 3,
        execution_timeout: float | None = None,
        job_deadline: float | None = None,
        *,
        expected_reconcile_attempts: int = 0,
    ) -> dict[str, str]:
        """Атомарно переводит Run в 'running', вызывает бизнес-логику и фиксирует исход."""
        run_uuid = UUID(str(run_id)) if not isinstance(run_id, UUID) else run_id

        lease_token: int | None = None
        lease_lost = False
        loop = asyncio.get_running_loop()
        lease_deadline = loop.time()
        heartbeat_enabled = self._heartbeat_interval is not None and self._heartbeat_interval > 0
        is_last_try = job_try >= max_tries

        async def claim_run() -> None:
            nonlocal lease_token, lease_deadline
            lease_deadline = loop.time() + self._lease_timeout
            lease_token = await self._dao.claim_run_for_execution(
                run_uuid,
                expected_reconcile_attempts=expected_reconcile_attempts,
            )

        try:
            await _finish_status_update(claim_run())
            if lease_token is None:
                logger.warning(
                    "Пропуск выполнения: run_id=%s (session_id=%s) "
                    "не находится в 'queued' или поколение задачи устарело",
                    run_uuid,
                    session_id,
                )
                return {
                    "status": "skipped",
                    "session_id": session_id,
                    "run_id": str(run_uuid),
                }

            logger.info(
                "Запущено выполнение сессии: session_id=%s, run_id=%s (попытка %d/%d)",
                session_id,
                run_uuid,
                job_try,
                max_tries,
            )

            if heartbeat_enabled and loop.time() >= lease_deadline:
                return {
                    "status": "conflict",
                    "session_id": session_id,
                    "run_id": str(run_uuid),
                }

            async def invoke_logic() -> Any:
                return await logic_callable(session_id)

            logic_task: asyncio.Task[Any] = asyncio.create_task(invoke_logic())

            def _handle_lease_lost() -> None:
                nonlocal lease_lost
                if lease_lost:
                    return
                lease_lost = True
                if not logic_task.done():
                    logic_task.cancel()

            lease_timer: asyncio.TimerHandle | None = None

            def _renew_lease_timer(deadline: float) -> None:
                nonlocal lease_timer
                if lease_lost:
                    return
                if lease_timer is not None:
                    lease_timer.cancel()
                # Этот callback не ждёт отмены heartbeat или очистки сессии БД.
                lease_timer = loop.call_at(deadline, _handle_lease_lost)

            heartbeat_task: asyncio.Task[None] | None = None
            if heartbeat_enabled:
                _renew_lease_timer(lease_deadline)
                heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(
                        run_uuid,
                        self._heartbeat_interval,
                        lease_token,
                        on_lease_lost=_handle_lease_lost,
                        lease_deadline=lease_deadline,
                        on_lease_renewed=_renew_lease_timer,
                    )
                )

            try:
                try:
                    # По умолчанию длительность job ограничивает только ARQ.
                    async with asyncio.timeout(execution_timeout):
                        await logic_task
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if lease_lost and (current is None or not current.cancelling()):
                        logger.warning(
                            "Выполнение прервано из-за потери lease: session_id=%s, run_id=%s",
                            session_id,
                            run_uuid,
                        )
                        return {
                            "status": "conflict",
                            "session_id": session_id,
                            "run_id": str(run_uuid),
                        }
                    raise
                finally:
                    try:
                        if not logic_task.done():
                            logic_task.cancel()
                            try:
                                await logic_task
                            except asyncio.CancelledError:
                                pass
                            except Exception:
                                logger.exception(
                                    "Ошибка при отмене logic_task. session_id=%s, run_id=%s",
                                    session_id,
                                    run_uuid,
                                )
                    finally:
                        try:
                            await _stop_heartbeat(heartbeat_task)
                        finally:
                            if lease_timer is not None:
                                lease_timer.cancel()
            except Exception:
                if lease_lost:
                    raise
                try:
                    await _finish_status_update(
                        self._dao.mark_run_failed(
                            run_uuid,
                            expected_reconcile_attempts=lease_token,
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed to mark run as failed after business logic error. "
                        "session_id=%s, run_id=%s",
                        session_id,
                        run_uuid,
                    )
                logger.exception(
                    "Ошибка при выполнении сессии. session_id=%s, run_id=%s",
                    session_id,
                    run_uuid,
                )
                raise

            # Бизнес-функция могла подавить отмену, а heartbeat уже завершиться.
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise asyncio.CancelledError
            if lease_lost:
                return {
                    "status": "conflict",
                    "session_id": session_id,
                    "run_id": str(run_uuid),
                }

            completed = await _finish_status_update(
                self._dao.mark_run_completed(
                    run_uuid,
                    expected_reconcile_attempts=lease_token,
                )
            )
            if not completed:
                logger.warning(
                    "mark_run_completed returned False "
                    "(run was not in 'running' state or lease lost). "
                    "session_id=%s, run_id=%s",
                    session_id,
                    run_uuid,
                )
                return {
                    "status": "conflict",
                    "session_id": session_id,
                    "run_id": str(run_uuid),
                }

        except asyncio.CancelledError:
            if lease_token is None or lease_lost:
                raise
            deadline_expired = (
                job_deadline is not None and asyncio.get_running_loop().time() >= job_deadline
            )
            try:
                if is_last_try or deadline_expired:
                    await _finish_status_update(
                        self._dao.mark_run_failed(
                            run_uuid,
                            expected_reconcile_attempts=lease_token,
                        )
                    )
                else:
                    await _finish_status_update(
                        self._dao.requeue_running_run(
                            run_uuid,
                            expected_reconcile_attempts=lease_token,
                        )
                    )
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "Failed to update run status during CancelledError handling. "
                    "session_id=%s, run_id=%s",
                    session_id,
                    run_uuid,
                )
            raise

        logger.info("Выполнение сессии завершено: session_id=%s, run_id=%s", session_id, run_uuid)
        return {
            "status": "completed",
            "session_id": session_id,
            "run_id": str(run_uuid),
        }
