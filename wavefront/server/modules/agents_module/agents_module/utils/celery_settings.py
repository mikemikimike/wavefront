"""Settings for the Celery app that publishes tasks, and for its queue gate.

Read from the `[celery]` section of the application's config.ini and handed in
by the container, the same way `RedisSettings` reaches `CacheManager`. Values
arrive as strings, or as None from an entry point whose config.ini does not
define the key (the celery worker defines only the socket timeout), so every
field is coerced here and falls back to its default when it is not usable.
"""

from dataclasses import dataclass, fields
from typing import Any, Callable

from common_module.log.logger import logger

DEFAULT_MAX_QUEUE_DEPTH = 10000
DEFAULT_QUEUE_DEPTH_CACHE_TTL_SECONDS = 1.0
DEFAULT_QUEUE_DEPTH_WARN_RATIO = 0.8
DEFAULT_BACKPRESSURE_RETRY_AFTER_SECONDS = 30
DEFAULT_BROKER_SOCKET_TIMEOUT_SECONDS = 2.0


def _coerce(name: str, raw: Any, cast: Callable[[Any], Any], default: Any) -> Any:
    """`raw` as `cast`, or the default if it is not usable.

    A typo in a tuning knob must not crash a server at startup, nor silently
    disable the gate, so an unusable value is logged and replaced.
    """
    if raw is None or (isinstance(raw, str) and raw.strip() == ''):
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        logger.warning(f'Celery: {name}={raw!r} is not usable, using {default}')
        return default


@dataclass(frozen=True)
class CeleryProducerSettings:
    # Annotated with what readers get. The container passes config.ini strings
    # (or None for a key the ini does not define); __post_init__ coerces them.

    #: Ready tasks on the default queue at or above which new work is refused
    #: with a 429. Zero or negative disables the gate entirely, which also
    #: skips the depth probe so it costs nothing.
    #:
    #: This is a safety valve, not a quota. It exists so the broker approaches
    #: its `maxmemory` gradually and visibly instead of being driven into it by
    #: a burst, given the worker runs --pool=solo and drains one task at a time.
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH

    #: How long a measured depth is reused before re-probing the broker. The
    #: depth is read on every inference request, so without this the gate would
    #: add a Redis round trip per request; with it, a burst costs one probe per
    #: second per process. The cost is overshoot — see `queue_is_saturated`.
    queue_depth_cache_ttl_seconds: float = DEFAULT_QUEUE_DEPTH_CACHE_TTL_SECONDS

    #: Fraction of max_queue_depth above which the depth is logged as a
    #: warning, so the queue filling up is visible before it costs callers 429s.
    queue_depth_warn_ratio: float = DEFAULT_QUEUE_DEPTH_WARN_RATIO

    #: `Retry-After` on a backpressure 429.
    backpressure_retry_after_seconds: int = DEFAULT_BACKPRESSURE_RETRY_AFTER_SECONDS

    #: Socket timeout for the producer's broker connection, and how long the
    #: depth probe waits for a pooled connection. Unset, redis-py blocks
    #: forever, so a wedged Redis would hang the enqueue — and the probe —
    #: rather than failing them.
    #:
    #: Producer-side only. The worker's own consuming app must NOT get this:
    #: its BRPOP depends on long blocking reads (brpop_timeout), which a socket
    #: timeout would cut short.
    broker_socket_timeout_seconds: float = DEFAULT_BROKER_SOCKET_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        casts: dict[str, Callable[[Any], Any]] = {
            'max_queue_depth': int,
            'queue_depth_cache_ttl_seconds': float,
            'queue_depth_warn_ratio': float,
            'backpressure_retry_after_seconds': int,
            'broker_socket_timeout_seconds': float,
        }
        for field in fields(self):
            value = _coerce(
                f'[celery] {field.name}',
                getattr(self, field.name),
                casts[field.name],
                field.default,
            )
            object.__setattr__(self, field.name, value)
