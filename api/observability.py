"""Privacy-safe structured fields for request-level operational telemetry."""

SAFE_LOG_FIELDS = frozenset(
    {
        "event",
        "request_id",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "service_version",
        "model_version",
    }
)


def request_log_record(
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    service_version: str,
    model_version: str | None,
) -> dict[str, str | int | float | None]:
    """Return the allowlisted metadata for one HTTP request, without payload data."""
    if not 100 <= status_code <= 599:
        raise ValueError("status_code must be between 100 and 599")
    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")
    return {
        "event": "http_request",
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "service_version": service_version,
        "model_version": model_version,
    }
