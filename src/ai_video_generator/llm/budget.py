"""One operation budget shared by transport, fallback and business repair.

The durable scheduler owns task attempts. This module bounds the provider calls
inside one attempt, including calls made by nested harnesses.
"""

from __future__ import annotations

import asyncio
import time
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .errors import LLMClientError

DEFAULT_OPERATION_SECONDS = 300.0
DEFAULT_CALL_BUDGET = 3
MAX_BUSINESS_REPAIRS = 1
MAX_CONTEXT_CHARACTERS = 120_000
MAX_OUTPUT_CHARACTERS = 40_000
MAX_OUTPUT_TOKENS = 8192
MAX_MEDIA_PARTS = 12
MAX_MEDIA_CHARACTERS = 24_000_000


@dataclass
class OperationBudget:
    deadline: float
    max_calls: int = DEFAULT_CALL_BUDGET
    calls: int = 0
    repairs: int = 0

    def claim_call(self) -> None:
        if time.monotonic() >= self.deadline:
            raise LLMClientError("LLM operation total deadline exceeded")
        if self.calls >= self.max_calls:
            raise LLMClientError("LLM operation provider call budget exhausted")
        self.calls += 1

    def claim_repair(self) -> bool:
        if self.repairs >= MAX_BUSINESS_REPAIRS or self.calls >= self.max_calls:
            return False
        self.repairs += 1
        return True


_budget: ContextVar[OperationBudget | None] = ContextVar("llm_operation_budget", default=None)


@asynccontextmanager
async def llm_operation(
    *, timeout_seconds: float = DEFAULT_OPERATION_SECONDS, max_calls: int = DEFAULT_CALL_BUDGET
) -> AsyncIterator[OperationBudget]:
    """Nested callers inherit the deadline and counters; they cannot refill them."""
    if timeout_seconds <= 0 or max_calls < 1:
        raise ValueError("LLM operation timeout and call budget must be positive")
    existing = _budget.get()
    if existing is not None:
        yield existing
        return
    budget = OperationBudget(time.monotonic() + timeout_seconds, max_calls=max_calls)
    token = _budget.set(budget)
    try:
        async with asyncio.timeout(timeout_seconds):
            yield budget
    except TimeoutError as exc:
        raise LLMClientError("LLM operation total deadline exceeded") from exc
    finally:
        _budget.reset(token)


def claim_business_repair() -> bool:
    budget = _budget.get()
    if budget is None:
        raise RuntimeError("business repair requires an LLM operation budget")
    return budget.claim_repair()


class _ProviderLimiter:
    def __init__(self) -> None:
        self.condition = asyncio.Condition()
        self.active = 0


_limiters: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


@asynccontextmanager
async def provider_request(provider: str, concurrency: int) -> AsyncIterator[None]:
    """Bound every provider request, including direct UI calls and fallbacks."""
    if concurrency < 1:
        raise ValueError("LLM concurrency must be positive")
    providers = _limiters.setdefault(asyncio.get_running_loop(), {})
    limiter = providers.setdefault(provider, _ProviderLimiter())
    async with limiter.condition:
        await limiter.condition.wait_for(lambda: limiter.active < concurrency)
        limiter.active += 1
    try:
        budget = _budget.get()
        if budget is not None:
            budget.claim_call()
        yield
    finally:
        async with limiter.condition:
            limiter.active -= 1
            limiter.condition.notify_all()
