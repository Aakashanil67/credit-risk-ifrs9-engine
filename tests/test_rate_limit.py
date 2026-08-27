from api.rate_limit import PredictionRateLimiter


def test_rate_limiter_blocks_the_next_request_after_the_window_quota() -> None:
    limiter = PredictionRateLimiter(limit=2, window_seconds=60)

    assert limiter.allow("198.51.100.7", now=100.0)
    assert limiter.allow("198.51.100.7", now=101.0)
    assert not limiter.allow("198.51.100.7", now=102.0)


def test_rate_limiter_allows_requests_again_after_the_window_expires() -> None:
    limiter = PredictionRateLimiter(limit=1, window_seconds=60)

    assert limiter.allow("198.51.100.7", now=100.0)
    assert limiter.allow("198.51.100.7", now=160.0)
