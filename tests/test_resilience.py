"""Tests for crashbytes-resilience."""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from typing import Any

import pytest

from crashbytes_resilience import (
    BulkheadFullError,
    CircuitBreaker,
    CircuitOpenError,
    RateLimiter,
    ResilienceError,
    RetriesExhaustedError,
    TimeoutExceededError,
    bulkhead,
    circuit_breaker,
    fallback,
    pipeline,
    rate_limiter,
    retry,
    timeout,
)

# ── Retry ───────────────────────────────────────────────────────────


class TestRetry:
    def test_succeeds_first_try(self) -> None:
        @retry(max_attempts=3, delay=0)
        def succeed() -> str:
            return "ok"

        assert succeed() == "ok"

    def test_retries_then_succeeds(self) -> None:
        call_count = 0

        @retry(max_attempts=3, delay=0)
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3

    def test_exhausted(self) -> None:
        @retry(max_attempts=2, delay=0)
        def always_fail() -> None:
            raise ValueError("nope")

        with pytest.raises(RetriesExhaustedError, match="nope"):
            always_fail()

    def test_specific_exceptions(self) -> None:
        @retry(max_attempts=3, delay=0, exceptions=(ValueError,))
        def raise_type() -> None:
            raise TypeError("wrong")

        with pytest.raises(TypeError):
            raise_type()

    def test_backoff_delay(self) -> None:
        times: list[float] = []

        @retry(max_attempts=3, delay=0.05, backoff=2.0)
        def track_time() -> None:
            times.append(time.monotonic())
            raise ValueError("fail")

        with pytest.raises(RetriesExhaustedError):
            track_time()

        assert len(times) == 3
        # First delay ~0.05s, second ~0.1s
        assert times[1] - times[0] >= 0.04
        assert times[2] - times[1] >= 0.08


class TestRetryAsync:
    async def test_async_retry(self) -> None:
        call_count = 0

        @retry(max_attempts=3, delay=0)
        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("fail")
            return "ok"

        result = await flaky()
        assert result == "ok"
        assert call_count == 2

    async def test_async_exhausted(self) -> None:
        @retry(max_attempts=2, delay=0)
        async def always_fail() -> None:
            raise ValueError("nope")

        with pytest.raises(RetriesExhaustedError):
            await always_fail()


# ── Circuit Breaker ─────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_closes_on_success(self) -> None:
        cb = circuit_breaker(failure_threshold=3)

        @cb
        def success() -> str:
            return "ok"

        assert success() == "ok"
        assert cb.state == CircuitBreaker.CLOSED

    def test_opens_after_threshold(self) -> None:
        cb = circuit_breaker(failure_threshold=2, recovery_timeout=10)

        @cb
        def fail() -> None:
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                fail()

        assert cb.state == CircuitBreaker.OPEN

        with pytest.raises(CircuitOpenError):
            fail()

    def test_half_open_recovery(self) -> None:
        cb = circuit_breaker(failure_threshold=1, recovery_timeout=0.05)

        call_count = 0

        @cb
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("fail")
            return "ok"

        with pytest.raises(ValueError):
            flaky()

        assert cb.state == CircuitBreaker.OPEN
        time.sleep(0.06)
        assert cb.state == CircuitBreaker.HALF_OPEN

        assert flaky() == "ok"
        assert cb.state == CircuitBreaker.CLOSED

    async def test_async_circuit_breaker(self) -> None:
        cb = circuit_breaker(failure_threshold=1)

        @cb
        async def fail() -> None:
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await fail()

        with pytest.raises(CircuitOpenError):
            await fail()


# ── Rate Limiter ────────────────────────────────────────────────────


class TestRateLimiter:
    def test_allows_within_limit(self) -> None:
        rl = rate_limiter(max_calls=5, period=1.0)

        @rl
        def call() -> str:
            return "ok"

        for _ in range(5):
            assert call() == "ok"

    def test_blocks_over_limit(self) -> None:
        rl = rate_limiter(max_calls=2, period=10.0)

        @rl
        def call() -> str:
            return "ok"

        call()
        call()
        with pytest.raises(ResilienceError, match="Rate limit"):
            call()

    def test_acquire_method(self) -> None:
        rl = RateLimiter(max_calls=1, period=10.0)
        assert rl.acquire() is True
        assert rl.acquire() is False

    async def test_async_rate_limiter(self) -> None:
        rl = rate_limiter(max_calls=1, period=10.0)

        @rl
        async def call() -> str:
            return "ok"

        assert await call() == "ok"
        with pytest.raises(ResilienceError):
            await call()


# ── Timeout ─────────────────────────────────────────────────────────


class TestTimeout:
    def test_within_timeout(self) -> None:
        @timeout(1.0)
        def fast() -> str:
            return "ok"

        assert fast() == "ok"

    def test_exceeds_timeout(self) -> None:
        @timeout(0.05)
        def slow() -> str:
            time.sleep(1.0)
            return "ok"

        with pytest.raises(TimeoutExceededError):
            slow()

    def test_propagates_exception(self) -> None:
        @timeout(1.0)
        def fail() -> None:
            raise ValueError("inner")

        with pytest.raises(ValueError, match="inner"):
            fail()

    async def test_async_timeout(self) -> None:
        @timeout(1.0)
        async def fast() -> str:
            return "ok"

        assert await fast() == "ok"

    async def test_async_exceeds_timeout(self) -> None:
        @timeout(0.05)
        async def slow() -> str:
            await asyncio.sleep(1.0)
            return "ok"

        with pytest.raises(TimeoutExceededError):
            await slow()


# ── Bulkhead ────────────────────────────────────────────────────────


class TestBulkhead:
    def test_allows_within_limit(self) -> None:
        @bulkhead(max_concurrent=2)
        def call() -> str:
            return "ok"

        assert call() == "ok"

    def test_blocks_over_limit(self) -> None:
        barrier = threading.Barrier(2)

        @bulkhead(max_concurrent=1)
        def call() -> str:
            barrier.wait(timeout=1)
            return "ok"

        errors: list[Exception] = []
        results: list[str] = []

        def run() -> None:
            try:
                results.append(call())
            except BulkheadFullError as e:
                errors.append(e)

        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start()
        time.sleep(0.02)  # Let t1 acquire the semaphore
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

        assert len(errors) == 1

    async def test_async_bulkhead(self) -> None:
        @bulkhead(max_concurrent=2)
        async def call() -> str:
            return "ok"

        assert await call() == "ok"


# ── Fallback ────────────────────────────────────────────────────────


class TestFallback:
    def test_no_fallback_on_success(self) -> None:
        @fallback(lambda: "fallback")
        def call() -> str:
            return "ok"

        assert call() == "ok"

    def test_fallback_on_failure(self) -> None:
        @fallback(lambda: "fallback")
        def call() -> str:
            raise ValueError("fail")

        assert call() == "fallback"

    async def test_async_fallback(self) -> None:
        @fallback(lambda: "fallback")
        async def call() -> str:
            raise ValueError("fail")

        assert await call() == "fallback"

    async def test_async_fallback_fn(self) -> None:
        async def fb() -> str:
            return "async fallback"

        @fallback(fb)
        async def call() -> str:
            raise ValueError("fail")

        assert await call() == "async fallback"


# ── Pipeline ────────────────────────────────────────────────────────


class TestPipeline:
    def test_compose_retry_and_fallback(self) -> None:
        call_count = 0

        @pipeline(retry(max_attempts=2, delay=0), fallback(lambda: "fallback"))
        def call() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        result = call()
        assert result == "fallback"
        assert call_count == 2

    def test_compose_order(self) -> None:
        order: list[str] = []

        def deco_a(fn: Any) -> Any:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                order.append("a")
                return fn(*args, **kwargs)
            return wrapper

        def deco_b(fn: Any) -> Any:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                order.append("b")
                return fn(*args, **kwargs)
            return wrapper

        @pipeline(deco_a, deco_b)
        def call() -> str:
            return "ok"

        call()
        # deco_b wraps deco_a wraps fn → b runs first, then a
        assert order == ["b", "a"]
