"""Admission control for the Celery broker.

The broker is Redis, it is capped (`maxmemory` + `noeviction`), and it is
shared with the floware application cache. The worker runs --pool=solo, so it
drains one task at a time per container while the enqueue path accepts work as
fast as HTTP allows.
Nothing stood between a burst and the memory ceiling, and reaching that
ceiling fails *writes* — which on a shared instance means the app's cache and
sessions go down with the queue.

So: read the queue depth before publishing, and refuse new work above a limit.
Cheaper for the caller too, who gets a 429 they can retry rather than a 202 for
an execution that will sit in a queue for an hour.
"""

import asyncio
import threading
import time
from typing import NamedTuple, Optional, Tuple

from celery import Celery
from common_module.log.logger import logger

from agents_module.utils.celery_client import CELERY_QUEUE_NAME
from agents_module.utils.celery_settings import (
    DEFAULT_BACKPRESSURE_RETRY_AFTER_SECONDS,
    CeleryProducerSettings,
)


class QueueBackpressure(Exception):
    """Raised instead of publishing when the broker queue is over its limit.

    Deliberately not a ValueError: the async inference controllers catch
    ValueError and answer 422, which would present a transient capacity limit
    as a malformed request.
    """

    def __init__(
        self,
        depth: int,
        limit: int,
        retry_after: int = DEFAULT_BACKPRESSURE_RETRY_AFTER_SECONDS,
    ) -> None:
        super().__init__(
            f'Execution queue is saturated: {depth} tasks queued, limit {limit}'
        )
        self.depth = depth
        self.limit = limit
        self.retry_after = retry_after


class BrokerPoolExhausted(Exception):
    """No broker connection was free for the depth probe within the timeout.

    This is contention in *our* connection pool, which says nothing about
    whether Redis is healthy. It is handled like any other failed probe; it has
    its own type so the log says what happened, where kombu's LimitExceeded
    would print only the pool size.
    """


class _Reading(NamedTuple):
    depth: int
    measured_at: float

    @property
    def age(self) -> float:
        return time.monotonic() - self.measured_at


#: How long a reading keeps being used when no newer one can be taken — the
#: probe failed, or another thread's probe has not come back. Past this the
#: gate admits work ungated.
#:
#: It is a ceiling in both directions, and both matter. Without it a reading
#: over the limit would refuse work for as long as the broker stayed
#: unprobeable, so a probe that is merely broken (a kombu change, a renamed
#: queue) would take inference down with a perfectly healthy broker behind it.
#: And a reading under the limit would wave everything through indefinitely
#: while still looking like a measurement.
_MAX_STALENESS_SECONDS = 30.0

# Held only by whoever is currently probing, and acquired without blocking, so
# concurrent callers skip the probe instead of queueing for it.
_depth_lock = threading.Lock()

# The last successful measurement. One reference, replaced whole, so a reader
# can never pair a new depth with an old timestamp.
_reading: Optional[_Reading] = None


def measure_queue_depth(
    app: Celery,
    settings: CeleryProducerSettings,
    queue: str = CELERY_QUEUE_NAME,
) -> int:
    """Ready tasks on `queue`, read through kombu. Blocking.

    `queue_declare` on the Redis transport returns `_size(queue)`, which
    pipelines LLEN across every priority sub-key (`{celery}`, `{celery}\\x06\\x163`,
    ...). Going through kombu rather than issuing our own LLEN is what keeps
    this honest: the key naming and priority fan-out come from the same code
    that performs the LPUSH, and the connection comes from CELERY_BROKER_URL —
    which is configured independently of the app's own REDIS_* variables and
    may well point at a different server.

    `passive` must stay False. The Redis transport implements `_has_queue` as
    EXISTS over the priority keys, so a passive declare raises ChannelError
    whenever the queue is empty, which is the normal case. Non-passive is still
    a pure read here: Redis's `_new_queue` is a no-op unless auto_delete.

    Counts ready messages only, excluding `{celery}.unacked` (prefetched but
    not yet acked). With worker_prefetch_multiplier=1 and --pool=solo that is
    at most one per worker, so it is not worth reaching for.

    Acquires with an explicit timeout instead of using
    `app.connection_or_acquire()`, which is `pool.acquire(block=True)` with
    `timeout=None` against a `broker_pool_limit` of 10. Unbounded, a pool busy
    with publishes parks this probe forever — and since publishes run on the
    thread pool too, that contention is real rather than theoretical. With a
    timeout, an exhausted pool raises LimitExceeded, re-raised here as
    BrokerPoolExhausted so the caller can tell local contention apart from a
    broker that is actually unwell.

    That translation is why this function owns the acquire: `LimitExceeded`
    lives in kombu, which is a transitive dependency (via `celery[redis]`) that
    nothing else in the repo imports directly. `app.pool.LimitExceeded` is the
    class the pool itself raises — ConnectionPool does not override
    `_add_when_empty`, so it raises `self.LimitExceeded`, i.e. kombu's
    ConnectionLimitExceeded.
    """
    timeout = settings.broker_socket_timeout_seconds
    try:
        # `release` below is the pool-aware closure Resource.acquire attaches,
        # not Connection.release (which is an alias for close()).
        conn = app.pool.acquire(block=True, timeout=timeout)
    except app.pool.LimitExceeded as exc:
        raise BrokerPoolExhausted(
            f'no broker connection free after {timeout}s: {exc}'
        ) from exc
    try:
        # One connection attempt, no retries. Left to itself `default_channel`
        # retries the connect with 2s then 4s sleeps under kombu's 4s
        # `connect_timeout` budget, so a probe against a dead broker took 6s
        # to fail — measured — where a refused connection should fail at once
        # and an unreachable one in broker_socket_timeout_seconds. Nothing
        # here is worth waiting for: a probe that cannot connect is simply a
        # failed probe. This is a no-op once the connection is established,
        # and it leaves the publish path's own retry policy alone.
        conn.ensure_connection(max_retries=0)
        return conn.default_channel.queue_declare(queue=queue).message_count
    finally:
        conn.release()


# -inf rather than 0.0 for both timestamps: time.monotonic() counts from an
# arbitrary point (boot, on Linux), so on a host up for less than the window a
# 0.0 here would swallow the first warning.
_probe_failing = False
_probe_failure_warned_at = float('-inf')


def _note_probe_failure(exc: Exception) -> None:
    """Log a failed probe, at most once per staleness window.

    While the broker cannot be probed this runs on nearly every request, so an
    unthrottled warning would be one line per request for the length of the
    outage. Caller holds _depth_lock, which is what makes the two globals safe.
    """
    global _probe_failing, _probe_failure_warned_at

    _probe_failing = True
    now = time.monotonic()
    if now - _probe_failure_warned_at < _MAX_STALENESS_SECONDS:
        return
    _probe_failure_warned_at = now
    logger.warning(
        f'Celery backpressure: queue-depth probe failed ({exc}); admission '
        f'keeps being decided on the last reading until it is '
        f'{_MAX_STALENESS_SECONDS:.0f}s old; after that work is enqueued '
        f'ungated until a probe succeeds.'
    )


def _note_probe_success() -> None:
    """Log the recovery once, as the counterpart of the failure warning."""
    global _probe_failing, _probe_failure_warned_at

    if not _probe_failing:
        return
    _probe_failing = False
    # So that a second outage is announced straight away rather than being
    # swallowed by the window of the first.
    _probe_failure_warned_at = float('-inf')
    logger.warning('Celery backpressure: queue-depth probe is working again.')


def _refresh_reading(app: Celery, settings: CeleryProducerSettings) -> None:
    """Probe the broker and store the result. Caller must hold _depth_lock.

    On failure the previous reading is left exactly as it was, which is what
    lets `_current_reading` go on answering from it.

    There is deliberately no circuit breaker in front of this. A failed probe
    is cheap — one connection attempt, at once if refused and
    broker_socket_timeout_seconds if unreachable — and only the one caller
    holding the lock pays it, so pausing probes would save a couple of seconds
    for a request whose publish is about to fail against the same broker
    anyway. Not worth a second state machine that says "open" while the gate
    carries on deciding.
    """
    global _reading

    try:
        depth = measure_queue_depth(app, settings)
    except Exception as exc:
        _note_probe_failure(exc)
        return

    _note_probe_success()
    _reading = _Reading(depth, time.monotonic())

    # Here rather than in the caller on purpose: that rate-limits the warning
    # to once per cache TTL per process for free. The limit guard is not
    # redundant — a non-positive max_queue_depth means the gate is off, which
    # makes the threshold `depth >= 0` (always true) and the percentage a
    # division by zero. queue_is_saturated returns before reaching here in
    # that case, but _current_depth is callable on its own.
    limit = settings.max_queue_depth
    if limit > 0 and depth >= limit * settings.queue_depth_warn_ratio:
        logger.warning(
            f'Celery backpressure: queue depth {depth} is at '
            f'{depth / limit:.0%} of the limit {limit}. '
            f'New executions are refused with 429 at the limit.'
        )


_ungated_warned_at = float('-inf')


def _warn_ungated(reading: _Reading) -> None:
    """Say that the gate is off, at most once per staleness window.

    This is the state worth knowing about: work is being admitted with no
    depth check at all because nothing has been measured recently. It is
    reached on every request while it lasts, hence the rate limit. Two threads
    can both pass the check and both log; that is harmless.
    """
    global _ungated_warned_at

    now = time.monotonic()
    if now - _ungated_warned_at < _MAX_STALENESS_SECONDS:
        return
    _ungated_warned_at = now
    logger.warning(
        f'Celery backpressure: no queue-depth reading for {reading.age:.0f}s '
        f'(last was {reading.depth}); enqueueing ungated until a probe succeeds.'
    )


def _fresh_reading(settings: CeleryProducerSettings) -> Optional[_Reading]:
    """The reading, if it is inside the cache TTL."""
    reading = _reading
    if reading is not None and reading.age < settings.queue_depth_cache_ttl_seconds:
        return reading
    return None


def _current_reading(
    app: Celery, settings: CeleryProducerSettings
) -> Optional[_Reading]:
    """The reading admission should be decided on, or None if there is none.

    One rule: **take a new reading if the last one is older than the cache
    TTL; either way, answer from the newest reading there is, provided it is no
    more than _MAX_STALENESS_SECONDS old.** With no such reading the answer is
    None and the caller admits the work.

    A new reading can fail to arrive for two reasons — the probe failed, or
    another thread is mid-probe — and they are deliberately not distinguished
    in the answer. In both, the previous reading is the best evidence
    available, and the question of how long to go on believing it has the same
    answer. In particular a reading that was over the limit keeps refusing
    work: losing sight of a saturated queue does not drain it. Probe failures
    and the recovery are logged, and every refusal logs the age of the reading
    it was decided on.

    When the broker is actually down — as opposed to merely unprobeable —
    admitting enqueues nothing: `send_task` fails against the same broker, and
    create_and_enqueue_* marks the execution failed and raises. The gate is
    not what stands between a dead broker and new work; the dead broker is.
    Refusing instead (failing closed) would be worse, because a probe failure
    is not evidence of saturation: a Redis failover would then 429 all
    inference for as long as it lasted, and a probe that is merely broken
    would do so indefinitely. And under
    `maxmemory` + `noeviction`, the condition this gate exists for, reads are
    still served, so the probe keeps working when it matters.

    Runs on the asyncio default thread pool, which is shared with everything
    else that calls asyncio.to_thread — binary input uploads, every
    CacheManager call. So this must never park a thread waiting its turn: one
    caller probes, and the rest answer from the previous reading instead of
    queueing behind it. Parking would let a slow broker consume the pool and
    stall work that has nothing to do with the broker. The cost is that at
    process start, before any reading exists, the non-probing callers get None
    for the duration of that first probe.
    """
    fresh = _fresh_reading(settings)
    if fresh is not None:
        return fresh

    if _depth_lock.acquire(blocking=False):
        try:
            # Another thread may have refreshed between the check above and
            # taking the lock.
            if _fresh_reading(settings) is None:
                _refresh_reading(app, settings)
        finally:
            _depth_lock.release()

    reading = _reading
    if reading is None:
        return None
    if reading.age > _MAX_STALENESS_SECONDS:
        _warn_ungated(reading)
        return None
    return reading


def _current_depth(app: Celery, settings: CeleryProducerSettings) -> Optional[int]:
    """The depth from `_current_reading`, or None if there is no reading."""
    reading = _current_reading(app, settings)
    return None if reading is None else reading.depth


async def queue_is_saturated(
    app: Celery, settings: CeleryProducerSettings
) -> Tuple[bool, Optional[int]]:
    """Whether to refuse new work, plus the depth that decided it.

    Predicate form, for callers that want to answer 429 themselves rather than
    handle an exception.

    `app` is the publishing client, passed in so the gate measures the broker
    the work would actually go to — the containers build it from
    `config.celery.broker_url`, which need not match anything in this process's
    environment. `settings` comes from the same `[celery]` config section. The
    reading below is module-level and therefore shared across
    apps; that is correct only while every provider in a process points at one
    broker, which is true today (both AgentsContainer and TriggersContainer
    read the same config key).

    The depth may be up to queue_depth_cache_ttl_seconds stale, and each
    process caches independently, so a burst can overshoot the limit by
    roughly (processes x requests/sec x TTL) before anyone re-probes. That is
    intended — this bounds the queue, it does not meter it. Size the limit with
    headroom, or shorten the TTL to trade Redis round trips for a tighter
    bound. Staleness in the other direction costs at most one TTL of 429s
    after the queue drains, which is what Retry-After is for. When the broker
    cannot be probed, both bounds stretch to _MAX_STALENESS_SECONDS.
    """
    limit = settings.max_queue_depth
    if limit <= 0:
        # Disabled: no probe, so a deployment that does not want the gate pays
        # nothing for it.
        return False, None

    # measure_queue_depth is sync redis-py and both inference routes are async
    # on the main event loop, so this must not run inline.
    reading = await asyncio.to_thread(_current_reading, app, settings)
    if reading is None:
        return False, None

    saturated = reading.depth >= limit
    if saturated:
        # The age is the point: a refusal decided on a reading from 25s ago,
        # because the broker cannot currently be probed, is a weaker claim
        # than one decided on a reading from 200ms ago, and reads differently
        # here.
        logger.warning(
            f'Celery backpressure: refusing work, queue depth {reading.depth} '
            f'is at or over the limit {limit} '
            f'(reading {reading.age:.1f}s old)'
        )
    return saturated, reading.depth


async def ensure_queue_capacity(app: Celery, settings: CeleryProducerSettings) -> None:
    """Raise QueueBackpressure if the broker queue is at or over its limit."""
    saturated, depth = await queue_is_saturated(app, settings)
    if saturated:
        # depth is never None when saturated is True.
        raise QueueBackpressure(
            depth,
            settings.max_queue_depth,
            settings.backpressure_retry_after_seconds,
        )


def _reset_for_tests() -> None:
    """Clear the reading and the log throttles. Tests only."""
    global _reading, _ungated_warned_at
    global _probe_failing, _probe_failure_warned_at
    with _depth_lock:
        _reading = None
        _probe_failing = False
        _probe_failure_warned_at = float('-inf')
    _ungated_warned_at = float('-inf')
