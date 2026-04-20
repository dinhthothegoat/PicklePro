"""Async local load simulator for PickleCoach.

Run against a live server for realistic HTTP behavior:
    python backend/load_test.py --users 1000 --duration 60 --base-url http://127.0.0.1:8000

Use --in-process for a fast app-only smoke load without starting uvicorn:
    python backend/load_test.py --in-process --users 50 --duration 10
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass, field
import logging
import random
import re
import statistics
import time
from typing import Awaitable, Callable
from urllib.parse import quote_plus

import httpx


logging.getLogger("httpx").setLevel(logging.WARNING)

Scenario = Callable[[httpx.AsyncClient, "Metrics", random.Random], Awaitable[None]]


@dataclass
class Metrics:
    latencies_ms: list[float] = field(default_factory=list)
    status_counts: Counter = field(default_factory=Counter)
    errors: Counter = field(default_factory=Counter)
    requests: int = 0

    def record(self, method: str, path: str, status_code: int, elapsed_ms: float) -> None:
        self.requests += 1
        self.latencies_ms.append(elapsed_ms)
        self.status_counts[status_code] += 1
        if status_code >= 400:
            self.errors[f"{method} {path} -> {status_code}"] += 1

    def record_exception(self, method: str, path: str, exc: Exception, elapsed_ms: float) -> None:
        self.requests += 1
        self.latencies_ms.append(elapsed_ms)
        self.errors[f"{method} {path} -> {type(exc).__name__}"] += 1


async def timed_request(
    client: httpx.AsyncClient,
    metrics: Metrics,
    method: str,
    path: str,
    **kwargs,
) -> httpx.Response | None:
    started = time.perf_counter()
    try:
        response = await client.request(method, path, **kwargs)
        elapsed = (time.perf_counter() - started) * 1000
        metrics.record(method, path, response.status_code, elapsed)
        return response
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        metrics.record_exception(method, path, exc, elapsed)
        return None


async def visitor_journey(client: httpx.AsyncClient, metrics: Metrics, rng: random.Random) -> None:
    await timed_request(client, metrics, "GET", "/")
    await timed_request(client, metrics, "GET", "/upload")
    await timed_request(client, metrics, "GET", "/tips")
    await timed_request(client, metrics, "GET", "/api/ml/models")


async def coach_search_journey(client: httpx.AsyncClient, metrics: Metrics, rng: random.Random) -> None:
    locations = ["Gold Coast", "Sydney", "Ho Chi Minh City", "Da Nang"]
    specialties = ["Dinks", "Footwork", "Serve returns", "Advanced drills"]
    location = quote_plus(rng.choice(locations))
    specialty = quote_plus(rng.choice(specialties))
    page = rng.randint(1, 4)

    await timed_request(client, metrics, "GET", f"/coaches?page={page}&sort=rating")
    await timed_request(client, metrics, "GET", f"/coaches?location={location}&specialty={specialty}")
    await timed_request(client, metrics, "GET", "/coaches/sarah-johnson")
    await timed_request(client, metrics, "GET", "/book?coach=Sarah%20Johnson")


async def analytics_journey(client: httpx.AsyncClient, metrics: Metrics, rng: random.Random) -> None:
    await timed_request(client, metrics, "GET", "/tiers")
    await timed_request(client, metrics, "GET", "/stats")
    await timed_request(client, metrics, "GET", "/history")
    await timed_request(client, metrics, "GET", "/favicon.ico")


async def booking_write_journey(client: httpx.AsyncClient, metrics: Metrics, rng: random.Random) -> None:
    response = await timed_request(client, metrics, "GET", "/book?coach=Sarah%20Johnson")
    if response is None:
        return

    match = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
    csrf = match.group(1) if match else ""
    user_id = rng.randint(1, 10_000_000)
    await timed_request(
        client,
        metrics,
        "POST",
        "/book",
        data={
            "coach": "Sarah Johnson",
            "name": f"Load User {user_id}",
            "email": f"load{user_id}@example.com",
            "preferred_date": "2026-05-01",
            "preferred_time": "10:30",
            "preferred_slot": "Mon 5:00 PM",
            "message": "Load test booking",
            "csrf_token": csrf,
        },
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


async def virtual_user(
    user_id: int,
    client: httpx.AsyncClient,
    metrics: Metrics,
    scenarios: list[tuple[int, Scenario]],
    deadline: float,
    think_min: float,
    think_max: float,
    ramp_up: float,
) -> None:
    rng = random.Random(user_id)
    if ramp_up > 0:
        await asyncio.sleep(rng.random() * ramp_up)

    weights = [weight for weight, _ in scenarios]
    handlers = [handler for _, handler in scenarios]
    while time.perf_counter() < deadline:
        scenario = rng.choices(handlers, weights=weights, k=1)[0]
        await scenario(client, metrics, rng)
        await asyncio.sleep(rng.uniform(think_min, think_max))


async def run_load(args: argparse.Namespace) -> Metrics:
    metrics = Metrics()
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=args.users, max_keepalive_connections=min(args.users, 200))

    scenarios: list[tuple[int, Scenario]] = [
        (45, visitor_journey),
        (40, coach_search_journey),
        (15, analytics_journey),
    ]
    if args.include_booking_writes:
        scenarios.append((args.booking_weight, booking_write_journey))

    if args.in_process:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from httpx import ASGITransport
        import main

        transport = ASGITransport(app=main.app)
        client_cm = httpx.AsyncClient(transport=transport, base_url="http://test", timeout=timeout)
    else:
        client_cm = httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=timeout, limits=limits)

    deadline = time.perf_counter() + args.duration
    async with client_cm as client:
        tasks = [
            asyncio.create_task(
                virtual_user(
                    user_id=index,
                    client=client,
                    metrics=metrics,
                    scenarios=scenarios,
                    deadline=deadline,
                    think_min=args.think_min,
                    think_max=args.think_max,
                    ramp_up=args.ramp_up,
                )
            )
            for index in range(args.users)
        ]
        await asyncio.gather(*tasks)
    return metrics


def print_summary(metrics: Metrics, duration: float) -> None:
    latencies = metrics.latencies_ms
    ok = metrics.requests - sum(metrics.errors.values())
    error_count = sum(metrics.errors.values())
    rps = metrics.requests / duration if duration > 0 else 0
    mean = statistics.mean(latencies) if latencies else 0.0

    print("\nLoad test summary")
    print("=================")
    print(f"Requests:      {metrics.requests}")
    print(f"Successful:    {ok}")
    print(f"Errors:        {error_count}")
    print(f"Throughput:    {rps:.2f} req/s")
    print(f"Mean latency:  {mean:.1f} ms")
    print(f"p50 latency:   {percentile(latencies, 50):.1f} ms")
    print(f"p90 latency:   {percentile(latencies, 90):.1f} ms")
    print(f"p95 latency:   {percentile(latencies, 95):.1f} ms")
    print(f"p99 latency:   {percentile(latencies, 99):.1f} ms")
    print(f"Status codes:  {dict(sorted(metrics.status_counts.items()))}")

    if metrics.errors:
        print("\nTop errors")
        for label, count in metrics.errors.most_common(10):
            print(f"  {count:5d}  {label}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate real PickleCoach users against the app.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Live server base URL.")
    parser.add_argument("--users", type=int, default=1000, help="Number of virtual users.")
    parser.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds.")
    parser.add_argument("--ramp-up", type=float, default=10.0, help="Seconds over which users start.")
    parser.add_argument("--timeout", type=float, default=15.0, help="Per-request timeout in seconds.")
    parser.add_argument("--think-min", type=float, default=0.2, help="Minimum user think time between journeys.")
    parser.add_argument("--think-max", type=float, default=1.2, help="Maximum user think time between journeys.")
    parser.add_argument("--in-process", action="store_true", help="Run against the ASGI app in-process.")
    parser.add_argument(
        "--include-booking-writes",
        action="store_true",
        help="Also POST booking requests. This creates database rows.",
    )
    parser.add_argument("--booking-weight", type=int, default=5, help="Relative weight for booking writes.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Fail if error rate exceeds this value.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    metrics = asyncio.run(run_load(args))
    elapsed = time.perf_counter() - started
    print_summary(metrics, elapsed)

    error_count = sum(metrics.errors.values())
    error_rate = error_count / metrics.requests if metrics.requests else 1.0
    return 1 if error_rate > args.max_error_rate else 0


if __name__ == "__main__":
    raise SystemExit(main())
