import pytest

from scripts.load_test import load_test_exit_code, percentile, summarise_results


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.50) == pytest.approx(25.0)
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.95) == pytest.approx(38.5)


def test_summary_separates_success_rate_limits_and_server_errors() -> None:
    summary = summarise_results(
        [
            {"status": 200, "duration_ms": 10.0, "transport_error": False},
            {"status": 200, "duration_ms": 20.0, "transport_error": False},
            {"status": 429, "duration_ms": 5.0, "transport_error": False},
            {"status": 503, "duration_ms": 30.0, "transport_error": False},
        ]
    )

    assert summary["requests"] == 4
    assert summary["successes"] == 2
    assert summary["rate_limited"] == 1
    assert summary["server_errors"] == 1
    assert summary["p50_ms"] == pytest.approx(15.0)


def test_summary_counts_transport_errors_separately():
    summary = summarise_results(
        [
            {"status": 200, "duration_ms": 10.0, "transport_error": False},
            {"status": 0, "duration_ms": 5.0, "transport_error": True},
        ]
    )

    assert summary["successes"] == 1
    assert summary["transport_errors"] == 1


@pytest.mark.parametrize(
    ("summary", "allow_rate_limits", "expected"),
    [
        (
            {
                "requests": 2,
                "successes": 2,
                "rate_limited": 0,
                "transport_errors": 0,
                "client_errors": 0,
                "server_errors": 0,
            },
            False,
            0,
        ),
        (
            {
                "requests": 2,
                "successes": 1,
                "rate_limited": 0,
                "transport_errors": 1,
                "client_errors": 0,
                "server_errors": 0,
            },
            False,
            1,
        ),
        (
            {
                "requests": 2,
                "successes": 1,
                "rate_limited": 0,
                "transport_errors": 0,
                "client_errors": 0,
                "server_errors": 1,
            },
            False,
            1,
        ),
        (
            {
                "requests": 2,
                "successes": 1,
                "rate_limited": 1,
                "transport_errors": 0,
                "client_errors": 1,
                "server_errors": 0,
            },
            True,
            0,
        ),
    ],
)
def test_load_test_exit_policy(summary, allow_rate_limits, expected):
    assert load_test_exit_code(summary, allow_rate_limits) == expected
