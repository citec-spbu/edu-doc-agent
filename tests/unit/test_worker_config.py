"""Временные настройки worker должны быть положительными конечными числами."""

from datetime import timedelta
from math import nextafter

import pytest
from pydantic import ValidationError

from edu_doc_agent.config import Settings


@pytest.mark.parametrize(
    "field",
    [
        "run_heartbeat_interval_seconds",
        "run_stale_running_seconds",
        "run_stale_queued_seconds",
    ],
)
@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "1e309"])
def test_nonfinite_worker_timing_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    timings = {
        "run_heartbeat_interval_seconds": 30.0,
        "run_stale_running_seconds": 120.0,
        "run_stale_queued_seconds": 120.0,
    }
    timings.pop(field)
    monkeypatch.setenv(field.upper(), value)

    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, **timings)

    assert any(
        error["loc"] == (field,) and error["type"] == "finite_number"
        for error in caught.value.errors()
    )


def test_finite_fractional_worker_timings_are_accepted() -> None:
    settings = Settings(
        _env_file=None,
        run_heartbeat_interval_seconds=0.5,
        run_stale_running_seconds=1.0,
        run_stale_queued_seconds=0.25,
    )
    assert settings.run_heartbeat_interval_seconds == 0.5
    assert settings.run_stale_running_seconds == 1.0
    assert settings.run_stale_queued_seconds == 0.25


@pytest.mark.parametrize("field", ["run_stale_running_seconds", "run_stale_queued_seconds"])
@pytest.mark.parametrize(
    "value",
    [
        "3e11",
        "1e20",
        str((timedelta.max.days + 1) * 86400),
        str(timedelta(days=365).total_seconds() + 1),
        str(nextafter(timedelta(days=365).total_seconds(), float("inf"))),
    ],
)
def test_unsafe_stale_threshold_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    monkeypatch.setenv(field.upper(), value)
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None)
    assert any(
        error["loc"] == (field,) and error["type"] == "less_than_equal"
        for error in caught.value.errors()
    )


@pytest.mark.parametrize("field", ["run_stale_running_seconds", "run_stale_queued_seconds"])
def test_maximum_safe_stale_threshold_environment_is_accepted(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    seconds = timedelta(days=365).total_seconds()
    monkeypatch.setenv(field.upper(), str(seconds))
    settings = Settings(_env_file=None)
    assert getattr(settings, field) == seconds


def test_stale_running_must_cover_two_heartbeat_intervals() -> None:
    with pytest.raises(ValidationError, match="must be at least 2x"):
        Settings(
            _env_file=None,
            run_heartbeat_interval_seconds=0.5,
            run_stale_running_seconds=0.75,
            run_stale_queued_seconds=1.0,
        )
