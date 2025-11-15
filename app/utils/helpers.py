"""Asynchronous helper utilities used throughout the solver."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import AsyncIterator, Awaitable, Callable, Optional


@dataclass
class Stopwatch:
    """Simple stopwatch to track elapsed time."""

    start_time: datetime

    @classmethod
    def start(cls) -> "Stopwatch":
        return cls(datetime.utcnow())

    def elapsed(self) -> timedelta:
        return datetime.utcnow() - self.start_time

    def remaining(self, budget: timedelta) -> timedelta:
        remaining = budget - self.elapsed()
        return max(remaining, timedelta())


@asynccontextmanager
async def time_budget(seconds: int) -> AsyncIterator[Stopwatch]:
    """Context manager to enforce an overall time budget."""

    timer = Stopwatch.start()
    deadline = timer.start_time + timedelta(seconds=seconds)
    try:
        yield timer
    finally:
        if datetime.utcnow() > deadline:
            raise TimeoutError("Time budget exceeded")


async def cancel_on_timeout(coro: Awaitable[object], timeout: float) -> Optional[object]:
    """Await a coroutine with cancellation support on timeout."""

    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return None


async def run_with_retry(
    func: Callable[[], Awaitable[object]],
    *,
    attempts: int,
    initial_delay: float,
    multiplier: float = 2.0,
) -> object:
    """Execute ``func`` with exponential backoff between retries."""

    delay = initial_delay
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except Exception as exc:  # pragma: no cover - behaviour validated via tests
            last_exc = exc
            if attempt == attempts:
                raise
            await asyncio.sleep(delay)
            delay *= multiplier
    if last_exc:
        raise last_exc
    raise RuntimeError("run_with_retry reached unexpected state")
