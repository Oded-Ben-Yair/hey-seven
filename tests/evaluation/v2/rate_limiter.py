"""Token bucket rate limiter for Gemini API RPM management."""

import asyncio
import time


class TokenBucketRateLimiter:
    """Async token bucket rate limiter.

    Refills tokens at a steady rate based on the configured RPM.
    Supports burst allowance for initial requests.
    """

    def __init__(self, rpm: int = 50, burst: int = 5):
        self.rpm = rpm
        self.burst = burst
        self._tokens = float(burst)
        self._max_tokens = float(burst)
        self._rate = rpm / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._max_tokens, self._tokens + elapsed * self._rate
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            # Wait before retrying — avoid busy-spinning
            await asyncio.sleep(0.1)
