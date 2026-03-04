# crashbytes-resilience

Unified resilience patterns for Python — retry, circuit breaker, rate limiter, timeout, bulkhead, fallback. Zero dependencies. Sync + async. Thread-safe.

## Install

```bash
pip install crashbytes-resilience
```

## Usage

```python
from crashbytes_resilience import retry, circuit_breaker, timeout, fallback, pipeline

@retry(max_attempts=3, delay=0.5, backoff=2.0)
def fetch_data():
    ...

@circuit_breaker(failure_threshold=5, recovery_timeout=30)
def call_service():
    ...

@timeout(5.0)
async def slow_operation():
    ...

@fallback(lambda: {"cached": True})
def get_config():
    ...

# Compose patterns
@pipeline(retry(max_attempts=3, delay=0.1), timeout(5.0), fallback(lambda: None))
def resilient_call():
    ...
```

## Patterns

| Decorator | Description |
|-----------|-------------|
| `@retry()` | Retry with exponential backoff |
| `@circuit_breaker()` | Stop calling failing services |
| `@rate_limiter()` | Token-bucket rate limiting |
| `@timeout()` | Time-limit operations |
| `@bulkhead()` | Limit concurrent executions |
| `@fallback()` | Provide fallback on failure |
| `@pipeline()` | Compose multiple patterns |

All patterns work with both sync and async functions.

## License

MIT
