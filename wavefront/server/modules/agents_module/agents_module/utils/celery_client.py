from celery import Celery

from agents_module.utils.celery_settings import CeleryProducerSettings

# Literal key; the braces are a Redis Cluster hash tag. Must match
# task_default_queue in celery_worker/celery_app.py.
CELERY_QUEUE_NAME = '{celery}'


def build_celery_client(
    broker_url: str, settings: CeleryProducerSettings | None = None
) -> Celery:
    # set_as_current=False: this app is also built inside the worker, and the
    # default would replace the worker's own app.
    app = Celery('async_executor', broker=broker_url, set_as_current=False)
    app.conf.task_default_queue = CELERY_QUEUE_NAME
    timeout = (settings or CeleryProducerSettings()).broker_socket_timeout_seconds
    app.conf.broker_transport_options = {
        'socket_timeout': timeout,
        'socket_connect_timeout': timeout,
    }
    return app
