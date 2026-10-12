from celery import Celery

from agents_module.utils.celery_settings import CeleryProducerSettings

#: The literal Redis key the worker's queue lives under. The braces are a Redis
#: Cluster hash tag, not a placeholder — they keep the queue and its
#: `{celery}.unacked*` bookkeeping in one hash slot so they stay on one shard.
#: Must match `task_default_queue` in celery_worker/celery_app.py. Note this
#: means the queue is `LLEN '{celery}'` from redis-cli; `LLEN celery` reads a
#: key that does not exist and answers 0.
CELERY_QUEUE_NAME = '{celery}'


def build_celery_client(
    broker_url: str, settings: CeleryProducerSettings | None = None
) -> Celery:
    """The Celery app used to publish tasks.

    Wired as a `providers.Singleton` in AgentsContainer and TriggersContainer,
    which is what keeps one app — and so one broker connection pool — per
    process. Building one per call would mean a new pool on every inference
    request.
    """
    # set_as_current=False is load-bearing, not tidiness. Celery() defaults it
    # to True, and TriggerEventProcessor reaches a publisher from *inside* the
    # worker (trigger_event_task -> _dispatch_inference ->
    # create_and_enqueue_agent). With the default, every trigger event replaced
    # the worker's own app as the current app with this one — which has no
    # result backend, no task `include`, and none of the `unacked_key`
    # transport overrides.
    app = Celery('async_executor', broker=broker_url, set_as_current=False)
    app.conf.task_default_queue = CELERY_QUEUE_NAME
    # Unset, redis-py blocks forever, so a wedged broker would hang the publish
    # and the queue-depth probe rather than failing them. Producer-side only:
    # the worker must not inherit these, because its BRPOP depends on long
    # blocking reads (brpop_timeout).
    timeout = (settings or CeleryProducerSettings()).broker_socket_timeout_seconds
    app.conf.broker_transport_options = {
        'socket_timeout': timeout,
        'socket_connect_timeout': timeout,
    }
    return app
