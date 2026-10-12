"""Queue-depth admission control for the Celery broker.

No Redis server and no broker: the kombu channel is faked, so these assert the
gate's decisions and the shape of the call it makes to measure depth.
"""

import threading
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest
from celery import Celery

from agents_module.utils import celery_backpressure as bp
from agents_module.utils.celery_backpressure import QueueBackpressure
from agents_module.utils.celery_settings import CeleryProducerSettings

#: The exception a real kombu ConnectionPool raises when nothing is free, taken
#: off a real (lazily built, broker-less) app rather than imported from kombu —
#: which is a transitive dependency nothing else in the repo imports directly.
#: Using the genuine class is the point: it pins the translation in
#: measure_queue_depth to what production will actually raise.
POOL_LIMIT_EXCEEDED = Celery('probe-test', set_as_current=False).pool.LimitExceeded

# Mirrors kombu's queue_declare_ok_t, which is what the real queue_declare
# returns; the gate reads .message_count off it.
declare_ok = namedtuple(
    'queue_declare_ok_t', ('queue', 'message_count', 'consumer_count')
)

#: Stands in for the publishing app the gate is asked about. Tests that do
#: not patch measure_queue_depth build a real fake with fake_app() instead.
APP = MagicMock(name='celery_app')


class _Knobs:
    """The settings the gate is called with in the current test.

    The gate takes its settings as an argument (they come from config.ini via
    the container), so a test states the ones it cares about with `set` and
    passes `KNOBS.settings` wherever the gate is called.
    """

    def __init__(self):
        self._values = {}

    def reset(self):
        self._values = {}

    def set(self, **values):
        self._values.update(values)

    @property
    def settings(self):
        return CeleryProducerSettings(**self._values)


KNOBS = _Knobs()


@pytest.fixture(autouse=True)
def reset_gate():
    """Each test starts with no reading, unthrottled logs and default knobs."""
    bp._reset_for_tests()
    KNOBS.reset()
    yield
    bp._reset_for_tests()
    KNOBS.reset()


@pytest.fixture
def limit_10(monkeypatch):
    KNOBS.set(max_queue_depth=10)
    return 10


def fake_app(depth):
    """A Celery app stand-in whose channel reports `depth` ready messages."""
    app = MagicMock()
    channel = app.pool.acquire.return_value.default_channel
    channel.queue_declare.return_value = declare_ok('{celery}', depth, 0)
    return app, channel


# --- measurement ----------------------------------------------------------


def test_measures_depth_through_queue_declare():
    app, channel = fake_app(7)

    assert bp.measure_queue_depth(app, KNOBS.settings) == 7


def test_declares_the_hash_tagged_queue_non_passively():
    """The key is literally `{celery}`, and passive would raise when empty.

    kombu's Redis transport implements _has_queue as EXISTS over the priority
    keys, so queue_declare(passive=True) raises ChannelError on an empty queue
    - the normal case. Pin both the name and the absence of passive=True.
    """
    app, channel = fake_app(0)

    bp.measure_queue_depth(app, KNOBS.settings)

    channel.queue_declare.assert_called_once_with(queue='{celery}')
    assert 'passive' not in channel.queue_declare.call_args.kwargs


def test_acquires_with_a_timeout_and_releases():
    """An unbounded acquire would park the probe on a busy pool forever.

    app.connection_or_acquire() is pool.acquire(block=True) with timeout=None
    against a broker_pool_limit of 10, and publishes draw on the same pool, so
    the probe acquires with a timeout instead. `release` is the pool-aware
    closure Resource.acquire attaches, so this returns the connection rather
    than closing it.
    """
    app, _channel = fake_app(0)

    bp.measure_queue_depth(app, KNOBS.settings)

    assert app.pool.acquire.call_args.kwargs['block'] is True
    assert app.pool.acquire.call_args.kwargs['timeout'] > 0
    app.pool.acquire.return_value.release.assert_called_once()


def test_connection_is_released_even_when_the_probe_raises():
    app, channel = fake_app(0)
    channel.queue_declare.side_effect = OSError('broker down')

    with pytest.raises(OSError):
        bp.measure_queue_depth(app, KNOBS.settings)

    app.pool.acquire.return_value.release.assert_called_once()


def test_probe_makes_a_single_connection_attempt():
    """A probe that cannot connect must fail now, not after kombu's retries.

    `default_channel` on its own retries the connect with 2s then 4s sleeps
    under a 4s budget: measured, a probe against a refused port took 6s to
    fail. ensure_connection(max_retries=0) first makes that one attempt, and
    is a no-op once connected.
    """
    app, _channel = fake_app(0)
    conn = app.pool.acquire.return_value
    calls = []
    conn.ensure_connection.side_effect = lambda **kw: calls.append(('ensure', kw))
    type(conn).default_channel = property(
        lambda self: calls.append(('channel', None)) or _channel
    )

    bp.measure_queue_depth(app, KNOBS.settings)

    assert calls == [('ensure', {'max_retries': 0}), ('channel', None)]


def exhausted_app():
    """A Celery app stand-in whose connection pool has nothing free."""
    app = MagicMock()
    app.pool.LimitExceeded = POOL_LIMIT_EXCEEDED
    app.pool.acquire.side_effect = POOL_LIMIT_EXCEEDED(10)
    return app


def test_pool_exhaustion_is_translated_not_leaked():
    """An exhausted pool must be distinguishable from an unwell broker.

    kombu's ConnectionPool inherits Resource._add_when_empty, which raises
    self.LimitExceeded once _dirty is at the limit — so a probe that times out
    waiting for a connection surfaces as ConnectionLimitExceeded. Left as-is it
    would be logged as kombu's bare `LimitExceeded(10)`, which says nothing.
    """
    app = exhausted_app()

    with pytest.raises(bp.BrokerPoolExhausted):
        bp.measure_queue_depth(app, KNOBS.settings)

    # Nothing was acquired, so there is nothing to hand back.
    app.pool.acquire.return_value.release.assert_not_called()


# --- the decision ---------------------------------------------------------


async def test_admits_below_the_limit(limit_10):
    with patch.object(bp, 'measure_queue_depth', return_value=9):
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (False, 9)
        await bp.ensure_queue_capacity(APP, KNOBS.settings)  # does not raise


async def test_refuses_at_the_limit(limit_10):
    """The limit is inclusive: exactly MAX_QUEUE_DEPTH refuses."""
    with patch.object(bp, 'measure_queue_depth', return_value=10):
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (True, 10)

        with pytest.raises(QueueBackpressure) as exc_info:
            await bp.ensure_queue_capacity(APP, KNOBS.settings)

    assert exc_info.value.depth == 10
    assert exc_info.value.limit == 10
    assert exc_info.value.retry_after == 30


async def test_refuses_above_the_limit(limit_10):
    with patch.object(bp, 'measure_queue_depth', return_value=11):
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (True, 11)


async def test_backpressure_is_not_a_value_error(limit_10):
    """The async inference controllers answer 422 on ValueError.

    A transient capacity limit reported as a malformed request would be wrong
    and would stop clients retrying.
    """
    with patch.object(bp, 'measure_queue_depth', return_value=10):
        with pytest.raises(QueueBackpressure) as exc_info:
            await bp.ensure_queue_capacity(APP, KNOBS.settings)

    assert not isinstance(exc_info.value, ValueError)


# --- disabled -------------------------------------------------------------


@pytest.mark.parametrize('limit', [0, -1])
async def test_non_positive_limit_disables_the_gate_without_probing(monkeypatch, limit):
    """A deployment that does not want the gate must not pay for it."""
    KNOBS.set(max_queue_depth=limit)

    with patch.object(bp, 'measure_queue_depth') as probe:
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (False, None)
        await bp.ensure_queue_capacity(APP, KNOBS.settings)

    probe.assert_not_called()


@pytest.mark.parametrize('limit', [0, -1])
def test_disabled_limit_does_not_divide_by_zero(monkeypatch, limit):
    """_current_depth is reachable on its own, so the warn log must survive it.

    With a non-positive limit the watermark test `depth >= limit * ratio` is
    always true, which takes the percentage straight into a ZeroDivisionError.
    queue_is_saturated returns before this, but nothing stops another caller.
    """
    KNOBS.set(max_queue_depth=limit)

    with patch.object(bp, 'measure_queue_depth', return_value=5):
        assert bp._current_depth(APP, KNOBS.settings) == 5


# --- fail open ------------------------------------------------------------


async def test_probe_failure_fails_open(limit_10, caplog):
    """Could-not-measure is not evidence of saturation."""
    with patch.object(bp, 'measure_queue_depth', side_effect=OSError('broker down')):
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (False, None)
        await bp.ensure_queue_capacity(APP, KNOBS.settings)  # does not raise

    assert 'probe failed' in caplog.text


async def test_pool_exhaustion_is_just_another_failed_probe(limit_10, caplog):
    """Our own pool being busy says nothing about the queue, so: admit.

    And say why. The probe is the only acquirer with a timeout (send_task's is
    unbounded), so under concurrent publishing it is the one that loses.
    """
    with patch.object(
        bp,
        'measure_queue_depth',
        side_effect=bp.BrokerPoolExhausted('no broker connection free after 2.0s: 10'),
    ):
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (False, None)

    assert 'no broker connection free' in caplog.text

    # Nothing is paused by it: the very next call measures.
    with patch.object(bp, 'measure_queue_depth', return_value=12):
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (True, 12)


# --- no new reading: answer from the last one -----------------------------
#
# One rule covers a failed probe and a probe another thread has in flight:
# decide on the newest reading there is, while it is no more than
# _MAX_STALENESS_SECONDS old.


def blind_after(monkeypatch, depth):
    """Take one good reading of `depth`, then return a clock to move time."""
    KNOBS.set(queue_depth_cache_ttl_seconds=1.0)
    clock = [1000.0]
    monkeypatch.setattr(bp.time, 'monotonic', lambda: clock[0])
    with patch.object(bp, 'measure_queue_depth', return_value=depth):
        assert bp._current_depth(APP, KNOBS.settings) == depth
    return clock


async def test_a_failed_probe_does_not_forget_a_saturated_queue(limit_10, monkeypatch):
    """The request whose own probe fails gets the same answer as everyone else.

    It used to get None and be admitted, while callers a moment either side of
    it were refused on the very reading it had just discarded.
    """
    clock = blind_after(monkeypatch, depth=12)
    clock[0] += 5.0

    with patch.object(bp, 'measure_queue_depth', side_effect=OSError('down')) as probe:
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (True, 12)
        with pytest.raises(QueueBackpressure):
            await bp.ensure_queue_capacity(APP, KNOBS.settings)

    assert probe.call_count == 2  # still trying to measure each time


async def test_an_outage_keeps_refusing_a_saturated_queue(limit_10, monkeypatch):
    """Losing sight of a queue that was over the limit does not drain it.

    So the gate must not flip to admitting because the broker went quiet.
    """
    clock = blind_after(monkeypatch, depth=12)
    clock[0] += 2.0

    with patch.object(bp, 'measure_queue_depth', side_effect=OSError('down')) as probe:
        for _ in range(6):
            assert await bp.queue_is_saturated(APP, KNOBS.settings) == (True, 12)

    # No breaker: it goes on trying to measure, so it sees the recovery on
    # the first call after the broker returns.
    assert probe.call_count == 6


async def test_an_outage_keeps_admitting_below_the_limit(limit_10, monkeypatch):
    clock = blind_after(monkeypatch, depth=3)
    clock[0] += 2.0

    with patch.object(bp, 'measure_queue_depth', side_effect=OSError('down')):
        for _ in range(6):
            assert await bp.queue_is_saturated(APP, KNOBS.settings) == (False, 3)


async def test_a_reading_past_the_ceiling_stops_being_believed(
    limit_10, monkeypatch, caplog
):
    """...in either direction, which is what makes believing it safe.

    A broken probe in front of a healthy broker must not refuse work forever,
    so past the ceiling a saturated reading is dropped and work is admitted.
    """
    clock = blind_after(monkeypatch, depth=12)

    with patch.object(bp, 'measure_queue_depth', side_effect=OSError('down')):
        clock[0] += bp._MAX_STALENESS_SECONDS - 1.0
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (True, 12)
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (True, 12)

        clock[0] += 2.0  # now past the ceiling
        assert 'enqueueing ungated' not in caplog.text
        for _ in range(4):
            assert await bp.queue_is_saturated(APP, KNOBS.settings) == (False, None)
        await bp.ensure_queue_capacity(APP, KNOBS.settings)  # does not raise

    # The gate being off is said out loud, once per window rather than once
    # per request.
    assert caplog.text.count('enqueueing ungated') == 1
    assert 'last was 12' in caplog.text


async def test_an_outage_is_logged_once_and_so_is_the_recovery(
    limit_10, monkeypatch, caplog
):
    """During an outage the probe fails on nearly every request.

    One line per request would bury everything else, so the failure is logged
    once per staleness window and the recovery once.
    """
    clock = blind_after(monkeypatch, depth=12)
    clock[0] += 2.0

    with patch.object(bp, 'measure_queue_depth', side_effect=OSError('down')) as probe:
        for _ in range(8):
            await bp.queue_is_saturated(APP, KNOBS.settings)
        assert probe.call_count == 8
        assert caplog.text.count('probe failed') == 1

        clock[0] += bp._MAX_STALENESS_SECONDS + 1.0  # still down a window later
        await bp.queue_is_saturated(APP, KNOBS.settings)
        assert caplog.text.count('probe failed') == 2

    assert 'working again' not in caplog.text
    with patch.object(bp, 'measure_queue_depth', return_value=4):
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (False, 4)
        clock[0] += 2.0
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (False, 4)
    assert caplog.text.count('working again') == 1

    # A second outage is announced at once, not swallowed by the first's window.
    clock[0] += 2.0
    with patch.object(bp, 'measure_queue_depth', side_effect=OSError('down again')):
        await bp.queue_is_saturated(APP, KNOBS.settings)
    assert caplog.text.count('probe failed') == 3


async def test_a_refusal_logs_how_old_its_evidence_is(limit_10, monkeypatch, caplog):
    """A 429 decided on a 20s-old reading must not read like a live one."""
    clock = blind_after(monkeypatch, depth=12)
    assert 'reading 0.0s old' not in caplog.text  # nothing refused yet

    assert await bp.queue_is_saturated(APP, KNOBS.settings) == (True, 12)
    assert 'reading 0.0s old' in caplog.text

    clock[0] += 20.0
    with patch.object(bp, 'measure_queue_depth', side_effect=OSError('down')):
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (True, 12)
    assert 'reading 20.0s old' in caplog.text


# --- the TTL cache --------------------------------------------------------


async def test_depth_is_reused_within_the_ttl(limit_10, monkeypatch):
    KNOBS.set(queue_depth_cache_ttl_seconds=1.0)
    clock = [1000.0]
    monkeypatch.setattr(bp.time, 'monotonic', lambda: clock[0])

    with patch.object(bp, 'measure_queue_depth', return_value=3) as probe:
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (False, 3)
        clock[0] += 0.5
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (False, 3)

    assert probe.call_count == 1


async def test_depth_is_remeasured_after_the_ttl(limit_10, monkeypatch):
    KNOBS.set(queue_depth_cache_ttl_seconds=1.0)
    clock = [1000.0]
    monkeypatch.setattr(bp.time, 'monotonic', lambda: clock[0])

    with patch.object(bp, 'measure_queue_depth', side_effect=[3, 12]) as probe:
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (False, 3)
        clock[0] += 1.5
        assert await bp.queue_is_saturated(APP, KNOBS.settings) == (True, 12)

    assert probe.call_count == 2


class HeldProbe:
    """A probe stuck in flight, so the contended path can be tested exactly.

    Sleeping and racing threads would make the assertions depend on 8 thread
    starts landing inside a sleep window, which is not something a loaded CI
    runner guarantees. This blocks the probe until the test releases it, so
    "another caller is probing right now" is a fact rather than a hope.
    """

    def __init__(self, depth):
        self.depth = depth
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=5), 'probe was never released'
        return self.depth

    def run_in_thread(self):
        """Start a prober and return once it is actually inside the probe."""
        result = []
        thread = threading.Thread(
            target=lambda: result.append(bp._current_depth(APP, KNOBS.settings))
        )
        thread.start()
        assert self.started.wait(timeout=5), 'probe never started'
        return thread, result

    def finish(self, thread):
        self.release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_callers_do_not_park_behind_an_in_flight_probe(limit_10, monkeypatch):
    """One caller probes; the rest must not queue behind it.

    _current_depth runs on the asyncio default thread pool, shared with binary
    input uploads and every CacheManager call, so parking here would let a slow
    broker starve work that has nothing to do with the broker.
    """
    KNOBS.set(queue_depth_cache_ttl_seconds=60.0)
    probe = HeldProbe(4)

    with patch.object(bp, 'measure_queue_depth', side_effect=probe):
        thread, probed = probe.run_in_thread()

        # The probe is in flight and holding the lock. These return at once
        # instead of blocking, and on a cold cache they have nothing to serve,
        # so they fail open.
        assert [bp._current_depth(APP, KNOBS.settings) for _ in range(7)] == [None] * 7

        probe.finish(thread)

    assert probe.calls == 1
    assert probed == [4]


def test_in_flight_probe_serves_the_previous_value(limit_10, monkeypatch):
    """Once there is a previous value, contended callers reuse it."""
    KNOBS.set(queue_depth_cache_ttl_seconds=0.0)

    with patch.object(bp, 'measure_queue_depth', return_value=4):
        assert bp._current_depth(APP, KNOBS.settings) == 4

    probe = HeldProbe(9)
    with patch.object(bp, 'measure_queue_depth', side_effect=probe):
        thread, probed = probe.run_in_thread()

        assert [bp._current_depth(APP, KNOBS.settings) for _ in range(3)] == [4, 4, 4]

        probe.finish(thread)

    assert probed == [9]


def test_a_wedged_probe_stops_being_believed(limit_10, monkeypatch):
    """A probe that never returns must not freeze the gate forever.

    A probe that hangs neither succeeds nor fails, so nothing replaces the
    last reading and nothing is logged. Without a staleness ceiling it
    would be served for the life of the process — the gate silently off if it
    was under the limit, or a permanent 429 if it was over.
    """
    KNOBS.set(queue_depth_cache_ttl_seconds=0.0)
    monkeypatch.setattr(bp, '_MAX_STALENESS_SECONDS', 30.0)

    clock = [1000.0]
    monkeypatch.setattr(bp.time, 'monotonic', lambda: clock[0])

    with patch.object(bp, 'measure_queue_depth', return_value=99):
        assert bp._current_depth(APP, KNOBS.settings) == 99

    probe = HeldProbe(1)
    with patch.object(bp, 'measure_queue_depth', side_effect=probe):
        thread, _probed = probe.run_in_thread()

        # Still within the ceiling: the stale reading is better than nothing.
        clock[0] += 29.0
        assert bp._current_depth(APP, KNOBS.settings) == 99

        # Past it: stop refusing work on the strength of a reading nobody can
        # refresh.
        clock[0] += 2.0
        assert bp._current_depth(APP, KNOBS.settings) is None

        probe.finish(thread)


# --- the warn watermark ---------------------------------------------------


async def test_warns_at_the_watermark(limit_10, monkeypatch, caplog):
    KNOBS.set(queue_depth_warn_ratio=0.8)

    with patch.object(bp, 'measure_queue_depth', return_value=8):
        await bp.queue_is_saturated(APP, KNOBS.settings)

    assert 'queue depth 8' in caplog.text


async def test_does_not_warn_below_the_watermark(limit_10, monkeypatch, caplog):
    KNOBS.set(queue_depth_warn_ratio=0.8)

    with patch.object(bp, 'measure_queue_depth', return_value=7):
        await bp.queue_is_saturated(APP, KNOBS.settings)

    assert 'queue depth' not in caplog.text


# --- settings from config.ini ---------------------------------------------


def test_config_strings_are_coerced():
    """config.ini hands every value over as a string."""
    settings = CeleryProducerSettings(
        max_queue_depth='250',
        queue_depth_cache_ttl_seconds='0.5',
        queue_depth_warn_ratio='0.9',
        backpressure_retry_after_seconds='45',
        broker_socket_timeout_seconds='3',
    )

    assert settings.max_queue_depth == 250
    assert settings.queue_depth_cache_ttl_seconds == 0.5
    assert settings.queue_depth_warn_ratio == 0.9
    assert settings.backpressure_retry_after_seconds == 45
    assert settings.broker_socket_timeout_seconds == 3.0


@pytest.mark.parametrize('raw', [None, '', '   ', 'not-a-number'])
def test_unusable_config_values_fall_back_to_the_default(raw):
    """A typo must not crash the server, nor silently disable the gate.

    None is what the container passes for a key the ini does not define, as in
    the celery worker, whose config.ini carries only the socket timeout.
    """
    settings = CeleryProducerSettings(
        max_queue_depth=raw,
        queue_depth_cache_ttl_seconds=raw,
        queue_depth_warn_ratio=raw,
        backpressure_retry_after_seconds=raw,
        broker_socket_timeout_seconds=raw,
    )

    assert settings == CeleryProducerSettings()
    assert settings.max_queue_depth == 10000


async def test_the_configured_retry_after_reaches_the_exception(limit_10):
    KNOBS.set(backpressure_retry_after_seconds='75')

    with patch.object(bp, 'measure_queue_depth', return_value=10):
        with pytest.raises(QueueBackpressure) as exc_info:
            await bp.ensure_queue_capacity(APP, KNOBS.settings)

    assert exc_info.value.retry_after == 75


def test_the_configured_socket_timeout_bounds_the_pool_wait():
    app, _channel = fake_app(0)
    KNOBS.set(broker_socket_timeout_seconds='0.25')

    bp.measure_queue_depth(app, KNOBS.settings)

    assert app.pool.acquire.call_args.kwargs['timeout'] == 0.25
