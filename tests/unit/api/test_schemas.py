"""Публичная схема ответа на постановку задания."""

from uuid import UUID

import pytest
from pydantic import ValidationError

from edu_doc_agent.api.schemas import ProcessResponse

RUN_ID = "fcbb00f8-380e-49af-b3a2-7b5043e7c485"


def test_process_response_defaults_and_json() -> None:
    response = ProcessResponse(run_id=RUN_ID, session_id="synthetic")

    assert response.run_id == UUID(RUN_ID)
    assert response.status == "queued"
    assert response.model_dump(mode="json") == {
        "run_id": RUN_ID,
        "session_id": "synthetic",
        "status": "queued",
    }


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"session_id": "test"}, "run_id"),
        ({"run_id": RUN_ID}, "session_id"),
        ({"run_id": "not-a-uuid", "session_id": "test"}, "run_id"),
    ],
)
def test_process_response_rejects_invalid_payload(payload: dict, field: str) -> None:
    with pytest.raises(ValidationError) as error:
        ProcessResponse.model_validate(payload)
    assert error.value.errors()[0]["loc"] == (field,)
