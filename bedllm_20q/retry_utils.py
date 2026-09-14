"""Exponential-backoff retry helper shared by bedllm_20q/ API call sites.

A free function rather than a decorator, because the call sites are a mix of
methods and module-level functions; each passes its own attempt/delay policy.
"""

import time
from typing import Callable, Optional, Tuple


def retry_with_backoff(
    fn: Callable,
    *,
    max_attempts: int,
    initial_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: Optional[float] = None,
    retryable: Tuple[type, ...] = (Exception,),
    on_error: Optional[Callable[[int, int, BaseException], None]] = None,
):
    """Call ``fn()`` with exponential backoff and return its result.

    Retries on any exception in ``retryable`` up to ``max_attempts`` attempts,
    then re-raises. The delay starts at ``initial_delay``, is multiplied by
    ``multiplier`` after each failure, and is capped at ``max_delay`` when
    given. ``on_error(attempt, max_attempts, exc)`` runs on every failed
    attempt (1-based), e.g. for logging.
    """
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retryable as e:
            if on_error is not None:
                on_error(attempt, max_attempts, e)
            if attempt == max_attempts:
                raise
            time.sleep(delay)
            delay *= multiplier
            if max_delay is not None:
                delay = min(delay, max_delay)
