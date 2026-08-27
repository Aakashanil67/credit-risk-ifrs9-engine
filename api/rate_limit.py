"""Small in-process guard for the public prediction endpoint.

Render runs this demo as a single instance. A per-IP fixed window is enough to stop an accidental
loop from tying up SHAP calculations; a production multi-instance service should enforce the same
policy at the edge or in shared storage.
"""

from collections import defaultdict, deque
from time import monotonic


class PredictionRateLimiter:
    def __init__(self, limit: int = 20, window_seconds: float = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_id: str, now: float | None = None) -> bool:
        now = monotonic() if now is None else now
        requests = self._requests[client_id]
        window_start = now - self.window_seconds
        while requests and requests[0] <= window_start:
            requests.popleft()
        if len(requests) >= self.limit:
            return False
        requests.append(now)
        return True
