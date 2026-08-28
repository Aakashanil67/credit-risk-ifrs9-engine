import pytest

from scripts.load_test import percentile, summarise_results


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.50) == pytest.approx(25.0)
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.95) == pytest.approx(38.5)


def test_summary_separates_success_rate_limits_and_server_errors() -> None:
    summary = summarise_results(
        [
            {"status": 200, "duration_ms": 10.0},
            {"status": 200, "duration_ms": 20.0},
            {"status": 429, "duration_ms": 5.0},
            {"status": 503, "duration_ms": 30.0},
        ]
    )

    assert summary["requests"] == 4
    assert summary["successes"] == 2
    assert summary["rate_limited"] == 1
    assert summary["server_errors"] == 1
    assert summary["p50_ms"] == pytest.approx(15.0)
