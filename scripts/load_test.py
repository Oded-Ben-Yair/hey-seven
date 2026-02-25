#!/usr/bin/env python3
"""Standalone load tester for Hey Seven API.

Usage:
    python scripts/load_test.py --url http://localhost:8080 --concurrency 10 --requests 50

Output: JSON summary of results including latency percentiles.
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class LoadTestResult:
    total_requests: int = 0
    successful: int = 0
    rate_limited: int = 0
    errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        lats = sorted(self.latencies_ms) if self.latencies_ms else [0]
        return {
            "total_requests": self.total_requests,
            "successful": self.successful,
            "rate_limited": self.rate_limited,
            "errors": self.errors,
            "p50_ms": round(lats[len(lats) // 2], 1),
            "p95_ms": round(lats[int(len(lats) * 0.95)], 1),
            "p99_ms": round(lats[int(len(lats) * 0.99)], 1),
            "max_ms": round(lats[-1], 1),
        }


async def send_request(
    client: httpx.AsyncClient, url: str, idx: int,
) -> tuple[int, float]:
    """Send a single /chat request and return (status_code, latency_ms)."""
    start = time.monotonic()
    try:
        r = await client.post(
            f"{url}/chat",
            json={"message": f"What restaurants are open? (load test #{idx})"},
            timeout=60,
        )
        latency = (time.monotonic() - start) * 1000
        return r.status_code, latency
    except Exception:
        latency = (time.monotonic() - start) * 1000
        return 0, latency


async def run_load_test(
    url: str, concurrency: int, total_requests: int,
) -> LoadTestResult:
    """Run load test with specified concurrency."""
    result = LoadTestResult()
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client: httpx.AsyncClient, idx: int) -> None:
        async with semaphore:
            status, latency = await send_request(client, url, idx)
            result.total_requests += 1
            result.latencies_ms.append(latency)
            if status == 200:
                result.successful += 1
            elif status == 429:
                result.rate_limited += 1
            else:
                result.errors += 1

    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(client, i) for i in range(total_requests)]
        await asyncio.gather(*tasks)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hey Seven API Load Tester")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=50)
    args = parser.parse_args()

    print(
        f"Load test: {args.requests} requests, "
        f"{args.concurrency} concurrent -> {args.url}"
    )
    result = asyncio.run(
        run_load_test(args.url, args.concurrency, args.requests),
    )
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
