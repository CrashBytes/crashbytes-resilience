"""Unified resilience patterns for Python."""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
import time
from typing import Any, TypeVar

T = TypeVar("T")


# ── Exceptions ──────────────────────────────────────────────────────


class ResilienceError(Exception):
    """Base for all resilience errors."""


class CircuitOpenError(ResilienceError):
    """Raised when the circuit breaker is open."""


class BulkheadFullError(ResilienceError):
    """Raised when the bulkhead is at capacity."""


class TimeoutExceededError(ResilienceError):
    """Raised when an operation exceeds its timeout."""


class RetriesExhaustedError(ResilienceError):
    """Raised when all retry attempts fail."""

    def __init__(self, last_exception: Exception) -> None:
        self.last_exception = last_exception
        super().__init__(f"All retries exhausted. Last error: {last_exception}")


# ── Retry ───────────────────────────────────────────────────────────


def retry(
    max_attempts: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """Decorator: retry on failure with exponential backoff.

    Works with both sync and async functions.
    """

    def decorator(fn: Any) -> Any:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                current_delay = delay
                last_exc: Exception | None = None
                for attempt in range(max_attempts):
                    try:
                        return await fn(*args, **kwargs)
                    except exceptions as exc:
                        last_exc = exc
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                assert last_exc is not None
                raise RetriesExhaustedError(last_exc)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            assert last_exc is not None
            raise RetriesExhaustedError(last_exc)

        return sync_wrapper

    return decorator


# ── Circuit Breaker ─────────────────────────────────────────────────


class CircuitBreaker:
    """Thread-safe circuit breaker.

    States: CLOSED -> OPEN -> HALF_OPEN -> CLOSED (or back to OPEN).
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._exceptions = exceptions
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if (
                self._state == self.OPEN
                and time.monotonic() - self._last_failure_time >= self._recovery_timeout
            ):
                self._state = self.HALF_OPEN
            return self._state

    def _record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = self.CLOSED

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self._failure_threshold:
                self._state = self.OPEN

    def __call__(self, fn: Any) -> Any:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if self.state == self.OPEN:
                    raise CircuitOpenError("Circuit breaker is open")
                try:
                    result = await fn(*args, **kwargs)
                except self._exceptions:
                    self._record_failure()
                    raise
                self._record_success()
                return result

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.state == self.OPEN:
                raise CircuitOpenError("Circuit breaker is open")
            try:
                result = fn(*args, **kwargs)
            except self._exceptions:
                self._record_failure()
                raise
            self._record_success()
            return result

        return sync_wrapper


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> CircuitBreaker:
    """Create a circuit breaker decorator."""
    return CircuitBreaker(failure_threshold, recovery_timeout, exceptions)


# ── Rate Limiter ────────────────────────────────────────────────────


class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, max_calls: int, period: float = 1.0) -> None:
        self._max_calls = max_calls
        self._period = period
        self._tokens = float(max_calls)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        refill = self._tokens + elapsed * self._max_calls / self._period
        self._tokens = min(self._max_calls, refill)
        self._last_refill = now

    def acquire(self) -> bool:
        """Try to acquire a token. Returns True if allowed."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def __call__(self, fn: Any) -> Any:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self.acquire():
                    raise ResilienceError("Rate limit exceeded")
                return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.acquire():
                raise ResilienceError("Rate limit exceeded")
            return fn(*args, **kwargs)

        return sync_wrapper


def rate_limiter(max_calls: int, period: float = 1.0) -> RateLimiter:
    """Create a rate limiter decorator."""
    return RateLimiter(max_calls, period)


# ── Timeout ─────────────────────────────────────────────────────────


def timeout(seconds: float) -> Any:
    """Decorator: raise TimeoutError_ if the function exceeds *seconds*.

    For async functions, uses asyncio.wait_for.
    For sync functions, uses threading.
    """

    def decorator(fn: Any) -> Any:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await asyncio.wait_for(fn(*args, **kwargs), timeout=seconds)
                except asyncio.TimeoutError:
                    raise TimeoutExceededError(f"Operation timed out after {seconds}s") from None

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            result_container: list[Any] = []
            exc_container: list[Exception] = []

            def target() -> None:
                try:
                    result_container.append(fn(*args, **kwargs))
                except Exception as e:
                    exc_container.append(e)

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=seconds)
            if thread.is_alive():
                raise TimeoutExceededError(f"Operation timed out after {seconds}s")
            if exc_container:
                raise exc_container[0]
            return result_container[0]

        return sync_wrapper

    return decorator


# ── Bulkhead ────────────────────────────────────────────────────────


def bulkhead(max_concurrent: int) -> Any:
    """Decorator: limit concurrent executions.

    Uses threading.Semaphore for sync, asyncio.Semaphore for async.
    """
    sync_sem = threading.Semaphore(max_concurrent)
    async_sem: asyncio.Semaphore | None = None

    def decorator(fn: Any) -> Any:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                nonlocal async_sem
                if async_sem is None:
                    async_sem = asyncio.Semaphore(max_concurrent)
                acquired = async_sem._value > 0  # noqa: SLF001
                if not acquired:
                    raise BulkheadFullError(
                        f"Bulkhead full (max {max_concurrent} concurrent)"
                    )
                async with async_sem:
                    return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            acquired = sync_sem.acquire(blocking=False)
            if not acquired:
                raise BulkheadFullError(
                    f"Bulkhead full (max {max_concurrent} concurrent)"
                )
            try:
                return fn(*args, **kwargs)
            finally:
                sync_sem.release()

        return sync_wrapper

    return decorator


# ── Fallback ────────────────────────────────────────────────────────


def fallback(fallback_fn: Any) -> Any:
    """Decorator: call *fallback_fn* if the wrapped function raises."""

    def decorator(fn: Any) -> Any:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await fn(*args, **kwargs)
                except Exception:
                    if inspect.iscoroutinefunction(fallback_fn):
                        return await fallback_fn(*args, **kwargs)
                    return fallback_fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception:
                return fallback_fn(*args, **kwargs)

        return sync_wrapper

    return decorator


# ── Pipeline ────────────────────────────────────────────────────────


def pipeline(*decorators: Any) -> Any:
    """Compose multiple resilience decorators into a single decorator.

    Applied inside-out: ``pipeline(retry(), circuit_breaker())`` means
    the circuit breaker wraps the retry which wraps the function.
    """

    def decorator(fn: Any) -> Any:
        result = fn
        for dec in decorators:
            result = dec(result)
        return result

    return decorator
