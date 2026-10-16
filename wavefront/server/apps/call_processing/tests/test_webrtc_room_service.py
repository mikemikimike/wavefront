"""Join-token / room lifecycle against a real Redis (skipped when none is reachable)."""

import threading
import uuid

import pytest
from redis import Redis

from call_processing.cache.cache_manager import CacheManager
from call_processing.services.webrtc_room_service import (
    ERR_INVALID_JOIN_TOKEN,
    ERR_ROOM_ALREADY_JOINED,
    ERR_ROOM_EXPIRED,
    ROOM_STATUS_JOINED,
    ROOM_STATUS_PENDING,
    WebRTCRoomError,
    WebRTCRoomService,
    hash_join_token,
)


@pytest.fixture
def cache_manager():
    try:
        Redis(socket_connect_timeout=1).ping()
    except Exception:
        pytest.skip('Redis is not reachable')
    manager = CacheManager(namespace=f'test-webrtc-{uuid.uuid4().hex}')
    yield manager
    for key in manager.redis.keys(f'{manager.namespace}/*'):
        manager.redis.delete(key)
    manager.close()


@pytest.fixture
def rooms(cache_manager):
    return WebRTCRoomService(cache_manager, room_ttl_secs=600, max_session_secs=1800)


AGENT_ID = str(uuid.uuid4())


def _code(exc_info) -> str:
    return exc_info.value.code


def test_create_room_stores_only_the_token_hash(rooms):
    room = rooms.create_room(AGENT_ID)

    record = rooms.get_room(room.room_id)
    assert record['status'] == ROOM_STATUS_PENDING
    assert record['voice_agent_id'] == AGENT_ID
    assert record['join_token_hash'] == hash_join_token(room.join_token)
    assert room.join_token not in str(record)
    assert (
        0
        < rooms.cache_manager.redis.ttl(
            f'{rooms.cache_manager.namespace}/webrtc_room:{room.room_id}'
        )
        <= 600
    )


def test_consume_marks_joined_and_clears_the_token(rooms):
    room = rooms.create_room(AGENT_ID)

    rooms.consume_join_token(room.room_id, AGENT_ID, room.join_token)

    record = rooms.get_room(room.room_id)
    assert record['status'] == ROOM_STATUS_JOINED
    assert 'join_token_hash' not in record
    ttl = rooms.cache_manager.redis.ttl(
        f'{rooms.cache_manager.namespace}/webrtc_room:{room.room_id}'
    )
    assert 600 < ttl <= 1800 + 60


def test_second_join_is_rejected(rooms):
    room = rooms.create_room(AGENT_ID)
    rooms.consume_join_token(room.room_id, AGENT_ID, room.join_token)

    with pytest.raises(WebRTCRoomError) as exc_info:
        rooms.consume_join_token(room.room_id, AGENT_ID, room.join_token)

    assert _code(exc_info) == ERR_ROOM_ALREADY_JOINED
    assert exc_info.value.status_code == 409


def test_wrong_token_or_agent_is_rejected_and_does_not_burn_the_room(rooms):
    room = rooms.create_room(AGENT_ID)

    for agent_id, token in (
        (AGENT_ID, 'wrong'),
        (str(uuid.uuid4()), room.join_token),
        (AGENT_ID, None),
    ):
        with pytest.raises(WebRTCRoomError) as exc_info:
            rooms.consume_join_token(room.room_id, agent_id, token)
        assert _code(exc_info) == ERR_INVALID_JOIN_TOKEN
        assert exc_info.value.status_code == 401

    # the real token still works
    rooms.consume_join_token(room.room_id, AGENT_ID, room.join_token)


def test_unknown_or_ended_room_is_expired(rooms):
    with pytest.raises(WebRTCRoomError) as exc_info:
        rooms.consume_join_token(uuid.uuid4().hex, AGENT_ID, 'token')
    assert _code(exc_info) == ERR_ROOM_EXPIRED
    assert exc_info.value.status_code == 410

    room = rooms.create_room(AGENT_ID)
    rooms.end_room(room.room_id)
    with pytest.raises(WebRTCRoomError) as exc_info:
        rooms.consume_join_token(room.room_id, AGENT_ID, room.join_token)
    assert _code(exc_info) == ERR_ROOM_EXPIRED


def test_concurrent_offers_only_one_wins(rooms):
    room = rooms.create_room(AGENT_ID)
    outcomes = []

    def join():
        try:
            rooms.consume_join_token(room.room_id, AGENT_ID, room.join_token)
            outcomes.append('ok')
        except WebRTCRoomError as e:
            outcomes.append(e.code)

    threads = [threading.Thread(target=join) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count('ok') == 1
    assert outcomes.count(ERR_ROOM_ALREADY_JOINED) == 7


def test_attach_connection_keeps_ttl(rooms):
    room = rooms.create_room(AGENT_ID)
    rooms.consume_join_token(room.room_id, AGENT_ID, room.join_token)

    rooms.attach_connection(room.room_id, 'pc-1', 'instance-a')

    record = rooms.get_room(room.room_id)
    assert (record['pc_id'], record['instance_id']) == ('pc-1', 'instance-a')
    assert (
        rooms.cache_manager.redis.ttl(
            f'{rooms.cache_manager.namespace}/webrtc_room:{room.room_id}'
        )
        > 0
    )
