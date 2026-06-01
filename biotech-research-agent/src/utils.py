"""
utils.py
---------
Shared utilities: in-memory caching, retry logic, rate limiting.
"""

import time
import hashlib
import json
import functools
from typing import Callable, Any, Optional
from collections import OrderedDict
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests


# ------------------------------------------------------------------
# Simple LRU cache for API responses
# ------------------------------------------------------------------

class SimpleCache:
    """
    Thread-unsafe in-memory LRU cache with TTL.
    Sufficient for a single-process Streamlit app.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._store: OrderedDict = OrderedDict()

    def _key(self, fn_name: str, args: dict) -> str:
        payload = json.dumps({"fn": fn_name, "args": args}, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def get(self, fn_name: str, args: dict) -> Optional[Any]:
        key = self._key(fn_name, args)
        if key in self._store:
            value, timestamp = self._store[key]
            if time.time() - timestamp < self.ttl:
                self._store.move_to_end(key)
                logger.debug(f"[cache] HIT: {fn_name}")
                return value
            else:
                del self._store[key]
        return None

    def set(self, fn_name: str, args: dict, value: Any):
        key = self._key(fn_name, args)
        if len(self._store) >= self.max_size:
            self._store.popitem(last=False)  # evict oldest
        self._store[key] = (value, time.time())
        logger.debug(f"[cache] SET: {fn_name}")

    def clear(self):
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# Global cache instance
_cache = SimpleCache(max_size=500, ttl_seconds=3600)


def cached(fn: Callable) -> Callable:
    """Decorator that caches function results by name + kwargs."""
    @functools.wraps(fn)
    def wrapper(**kwargs):
        result = _cache.get(fn.__name__, kwargs)
        if result is not None:
            return result
        result = fn(**kwargs)
        _cache.set(fn.__name__, kwargs, result)
        return result
    return wrapper


# ------------------------------------------------------------------
# Retry decorator for flaky APIs
# ------------------------------------------------------------------

def with_retry(fn: Callable) -> Callable:
    """
    Decorator: retry up to 3 times with exponential backoff
    on connection errors and HTTP 5xx responses.
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError,
                                       requests.exceptions.Timeout)),
        reraise=True,
    )
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


# ------------------------------------------------------------------
# Rate limiter
# ------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, calls_per_second: float = 2.0):
        self.min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()


# Per-API rate limiters
pubmed_limiter = RateLimiter(calls_per_second=2.5)
default_limiter = RateLimiter(calls_per_second=2.0)


# ------------------------------------------------------------------
# Truncation helper
# ------------------------------------------------------------------

def truncate(text: str, max_chars: int = 500) -> str:
    """Truncate a string to max_chars with ellipsis."""
    if not text:
        return ""
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def format_tool_result_for_display(tool_name: str, result_json: str) -> str:
    """Format a tool result for display in the Streamlit UI."""
    try:
        data = json.loads(result_json)
        if "error" in data:
            return f"⚠️ Error: {data['error']}"
        return json.dumps(data, indent=2)[:2000]
    except Exception:
        return result_json[:2000]
