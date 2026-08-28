"""Bounded concurrent timing probe for the public prediction API."""

import argparse
import asyncio
import json
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import httpx

VALID_APPLICANT = {
    "age_years": 35,
    "years_employed": 5,
    "income_total": 180_000,
    "credit_amount": 450_000,
    "annuity": 22_500,
    "goods_price": 450_000,
    "owns_car": True,
    "owns_realty": True,
    "num_children": 1,
    "family_members": 3,
    "education": "Higher education",
}


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without a new dependency."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarise_results(results: Sequence[dict[str, float | int]]) -> dict[str, int | float | None]:
    """Summarise status counts and successful-request latency without exposing payloads."""
    statuses = Counter(int(result["status"]) for result in results)
    success_latencies = [
        float(result["duration_ms"]) for result in results if result["status"] == 200
    ]
    summary: dict[str, int | float | None] = {
        "requests": len(results),
        "successes": statuses[200],
        "rate_limited": statuses[429],
        "client_errors": sum(count for status, count in statuses.items() if 400 <= status < 500),
        "server_errors": sum(count for status, count in statuses.items() if 500 <= status < 600),
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
    }
    if success_latencies:
        summary.update(
            {
                "p50_ms": percentile(success_latencies, 0.50),
                "p95_ms": percentile(success_latencies, 0.95),
                "p99_ms": percentile(success_latencies, 0.99),
            }
        )
    summary["status_counts"] = dict(sorted(statuses.items()))
    return summary


async def run_load_test(
    url: str, requests: int, concurrency: int, timeout: float
) -> tuple[list[dict[str, float | int]], float]:
    """Run bounded concurrent POST requests and retain status plus elapsed time only."""
    semaphore = asyncio.Semaphore(concurrency)

    async def send(client: httpx.AsyncClient) -> dict[str, float | int]:
        async with semaphore:
            started_at = time.perf_counter()
            try:
                response = await client.post(f"{url.rstrip('/')}/predict", json=VALID_APPLICANT)
                status = response.status_code
            except httpx.HTTPError:
                status = 0
            return {"status": status, "duration_ms": (time.perf_counter() - started_at) * 1_000}

    started_at = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await asyncio.gather(*(send(client) for _ in range(requests)))
    return results, (time.perf_counter() - started_at) * 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=15)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--allow-rate-limit-test", action="store_true")
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be at least one")
    if args.requests > 20 and not args.allow_rate_limit_test:
        parser.error("--requests above 20 requires --allow-rate-limit-test")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least one")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


async def main() -> None:
    """Print a single aggregate JSON measurement for a deliberately small request batch."""
    args = parse_args()
    results, wall_time_ms = await run_load_test(
        args.url, args.requests, args.concurrency, args.timeout
    )
    summary: dict[str, Any] = summarise_results(results)
    summary["wall_time_ms"] = wall_time_ms
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
