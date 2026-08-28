import json

import pytest

from api.observability import SAFE_LOG_FIELDS, request_log_record


def test_request_log_record_contains_only_the_safe_allowlist() -> None:
    record = request_log_record(
        request_id="abc123",
        method="POST",
        path="/predict",
        status_code=200,
        duration_ms=12.345,
        service_version="1.3.0",
        model_version="1.2.0",
    )

    assert set(record) == SAFE_LOG_FIELDS
    assert record["event"] == "http_request"
    assert record["duration_ms"] == 12.35


def test_request_log_json_cannot_contain_applicant_or_network_fields() -> None:
    record = request_log_record("id", "POST", "/predict", 422, 1.0, "1.3.0", "1.2.0")
    serialised = json.dumps(record).lower()

    for forbidden in ("income", "credit_amount", "reason_codes", "client_ip", "query", "body"):
        assert forbidden not in serialised


@pytest.mark.parametrize(
    ("status_code", "duration_ms"),
    [(99, 0.0), (600, 0.0), (200, -0.01)],
)
def test_request_log_rejects_invalid_transport_values(status_code: int, duration_ms: float) -> None:
    with pytest.raises(ValueError):
        request_log_record("id", "GET", "/health", status_code, duration_ms, "1.3.0", "1.2.0")
