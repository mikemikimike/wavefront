"""A refused enqueue must cost nothing.

create_and_enqueue_* uploads binary inputs to object storage and inserts a
`pending` row before it publishes, so the guard has to sit ahead of both -
otherwise a 429 leaves an orphan row and stored blobs behind for a request
that never ran. No DB and no cloud storage here: both are MagicMock, which is
the point - the assertions are that neither was touched.
"""

import base64
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents_module.services import async_agentic_execution_service as module
from agents_module.services.async_agentic_execution_service import (
    AsyncAgenticExecutionService,
)
from agents_module.utils.celery_backpressure import QueueBackpressure


@pytest.fixture
def uuid4():
    return uuid.uuid4()


@pytest.fixture
def service():
    svc = AsyncAgenticExecutionService(
        async_agentic_execution_repository=MagicMock(),
        cloud_storage_manager=MagicMock(),
        cache_manager=MagicMock(),
        celery_client=MagicMock(),
        executions_bucket='test-bucket',
    )
    svc.repo.create = AsyncMock()
    svc.repo.find_one_and_update = AsyncMock()
    return svc


@pytest.fixture
def saturated():
    """The queue is over its limit."""
    with patch.object(
        module,
        'ensure_queue_capacity',
        AsyncMock(side_effect=QueueBackpressure(10000, 10000)),
    ) as guard:
        yield guard


@pytest.fixture
def has_capacity():
    with patch.object(module, 'ensure_queue_capacity', AsyncMock()) as guard:
        yield guard


# A real binary input, so the blob assertions below mean something. A plain
# string would make them vacuous: pre_save_binary_inputs returns at its
# `isinstance(inputs, str)` short-circuit and never reaches cloud storage, so
# save_small_file would go uncalled whether the guard ran or not.
PDF_B64 = base64.b64encode(b'%PDF-1.4 fake').decode('utf-8')

BINARY_INPUTS = [
    {
        'role': 'user',
        'content': {
            'document_base64': PDF_B64,
            'mime_type': 'application/pdf',
            'file_name': 'doc.pdf',
        },
    }
]

AGENT_KWARGS = dict(
    inputs=BINARY_INPUTS,
    variables=None,
    output_json_enabled=False,
    access_token=None,
    app_key=None,
)

WORKFLOW_KWARGS = dict(
    workflow_name='wf',
    namespace='default',
    inputs=BINARY_INPUTS,
    variables=None,
    output_json_enabled=False,
    access_token=None,
    app_key=None,
)


async def test_agent_enqueue_refused_when_saturated(service, saturated, uuid4):
    with pytest.raises(QueueBackpressure):
        await service.create_and_enqueue_agent(agent_id=uuid4, **AGENT_KWARGS)


async def test_agent_refusal_writes_no_blob_and_no_row(service, saturated, uuid4):
    with pytest.raises(QueueBackpressure):
        await service.create_and_enqueue_agent(agent_id=uuid4, **AGENT_KWARGS)

    service.cloud_storage.save_small_file.assert_not_called()
    service.repo.create.assert_not_called()
    service.repo.find_one_and_update.assert_not_called()


async def test_workflow_refusal_writes_no_blob_and_no_row(service, saturated, uuid4):
    with pytest.raises(QueueBackpressure):
        await service.create_and_enqueue_workflow(workflow_id=uuid4, **WORKFLOW_KWARGS)

    service.cloud_storage.save_small_file.assert_not_called()
    service.repo.create.assert_not_called()


async def test_refusal_is_not_a_value_error(service, saturated, uuid4):
    """The controllers map ValueError to 422; this must not land there."""
    with pytest.raises(QueueBackpressure) as exc_info:
        await service.create_and_enqueue_agent(agent_id=uuid4, **AGENT_KWARGS)

    assert not isinstance(exc_info.value, ValueError)


async def test_worker_internal_callers_skip_the_guard(service, saturated, uuid4):
    """enforce_queue_limit=False is how trigger event dispatch stays alive.

    That path runs inside the worker on an event already gated at the webhook.
    _handle_single_event would catch a refusal and mark the trigger event row
    failed with processed_at set, and CELERY_TASK_MAX_RETRIES defaults to 0 -
    so refusing there loses the work instead of deferring it.
    """
    service.celery.send_task.return_value.id = 'task-1'

    await service.create_and_enqueue_agent(
        agent_id=uuid4, enforce_queue_limit=False, **AGENT_KWARGS
    )

    saturated.assert_not_called()
    service.repo.create.assert_called_once()


async def test_enqueue_proceeds_when_there_is_capacity(service, has_capacity, uuid4):
    """The control for the assertions above.

    Same inputs, capacity available: the blob *is* written and the row *is*
    inserted. Without this, `assert_not_called` on the refusal path could pass
    for reasons unrelated to the guard.
    """
    service.celery.send_task.return_value.id = 'task-1'

    result = await service.create_and_enqueue_agent(agent_id=uuid4, **AGENT_KWARGS)

    has_capacity.assert_awaited_once()
    service.cloud_storage.save_small_file.assert_called_once()
    service.repo.create.assert_called_once()
    assert result.status == 'pending'


async def test_service_passes_its_configured_settings_to_the_gate(uuid4):
    """The limit that applies is the one from config.ini, not a default."""
    from agents_module.utils.celery_settings import CeleryProducerSettings

    settings = CeleryProducerSettings(max_queue_depth='7')
    svc = AsyncAgenticExecutionService(
        async_agentic_execution_repository=MagicMock(),
        cloud_storage_manager=MagicMock(),
        cache_manager=MagicMock(),
        celery_client=MagicMock(),
        executions_bucket='test-bucket',
        celery_settings=settings,
    )

    with patch.object(
        module, 'queue_is_saturated', AsyncMock(return_value=(True, 7))
    ) as predicate:
        assert await svc.queue_is_saturated() is True
    predicate.assert_awaited_once_with(svc.celery, settings)

    with patch.object(module, 'ensure_queue_capacity', AsyncMock()) as guard:
        svc.repo.create = AsyncMock()
        svc.repo.find_one_and_update = AsyncMock()
        await svc.create_and_enqueue_agent(agent_id=uuid4, **AGENT_KWARGS)
    guard.assert_awaited_once_with(svc.celery, settings)
