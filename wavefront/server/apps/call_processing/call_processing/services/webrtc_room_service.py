"""Redis-backed registry of browser (WebRTC) rooms.

A room is created by floware for an active voice agent. The browser then joins
it with a one-time ``join_token``. Only the SHA-256 hash of the token is stored;
the plaintext is returned once, on creation.

Room record (``webrtc_room:{room_id}``), JSON:
    voice_agent_id, join_token_hash, status (pending | joined | ended),
    created_at, joined_at, pc_id, instance_id

Peer connections themselves are NOT stored here (aiortc state is in-process);
``pc_id`` / ``instance_id`` are kept for diagnostics and sticky-routing errors.
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from call_processing.cache.cache_manager import CacheManager
from call_processing.log.logger import logger

ROOM_STATUS_PENDING = 'pending'
ROOM_STATUS_JOINED = 'joined'
ROOM_STATUS_ENDED = 'ended'

# Stable error codes surfaced to the browser client
ERR_INVALID_JOIN_TOKEN = 'invalid_join_token'
ERR_ROOM_EXPIRED = 'room_expired'
ERR_ROOM_ALREADY_JOINED = 'room_already_joined'
ERR_STICKY_REQUIRED = 'sticky_required'

# Consume the join token: validate, flip pending -> joined, drop the token hash.
# Runs atomically so two concurrent offers cannot both win.
#
# KEYS[1] room key
# ARGV[1] token hash, ARGV[2] voice_agent_id, ARGV[3] joined_at, ARGV[4] joined TTL
# Returns {'ok'} or {'err', <code>}
_CONSUME_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then
    return {'err', 'room_expired'}
end
local room = cjson.decode(raw)
if room['status'] == 'ended' then
    return {'err', 'room_expired'}
end
if room['status'] ~= 'pending' then
    return {'err', 'room_already_joined'}
end
if room['voice_agent_id'] ~= ARGV[2] then
    return {'err', 'invalid_join_token'}
end
if (not room['join_token_hash']) or room['join_token_hash'] ~= ARGV[1] then
    return {'err', 'invalid_join_token'}
end
room['status'] = 'joined'
room['joined_at'] = ARGV[3]
room['join_token_hash'] = nil
redis.call('SET', KEYS[1], cjson.encode(room), 'EX', tonumber(ARGV[4]))
return {'ok'}
"""

# Attach pc_id / instance_id to a joined room, keeping its TTL.
# ARGV[1] pc_id, ARGV[2] instance_id
_ATTACH_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then
    return 0
end
local room = cjson.decode(raw)
room['pc_id'] = ARGV[1]
room['instance_id'] = ARGV[2]
redis.call('SET', KEYS[1], cjson.encode(room), 'KEEPTTL')
return 1
"""


class WebRTCRoomError(Exception):
    """Raised when a room cannot be joined. ``code`` is a stable ERR_* value."""

    def __init__(self, code: str, status_code: int, message: str):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message


_ERROR_STATUS = {
    ERR_INVALID_JOIN_TOKEN: (401, 'The join token is invalid'),
    ERR_ROOM_EXPIRED: (410, 'The room has expired or has ended'),
    ERR_ROOM_ALREADY_JOINED: (409, 'The room has already been joined'),
}


def room_error(code: str) -> WebRTCRoomError:
    status_code, message = _ERROR_STATUS[code]
    return WebRTCRoomError(code, status_code, message)


@dataclass(frozen=True)
class CreatedRoom:
    room_id: str
    voice_agent_id: str
    join_token: str


def hash_join_token(join_token: str) -> str:
    return hashlib.sha256(join_token.encode('utf-8')).hexdigest()


class WebRTCRoomService:
    def __init__(
        self,
        cache_manager: CacheManager,
        room_ttl_secs: int,
        max_session_secs: int,
    ):
        self.cache_manager = cache_manager
        self.room_ttl_secs = room_ttl_secs
        self.max_session_secs = max_session_secs

    @staticmethod
    def _key(room_id: str) -> str:
        return f'webrtc_room:{room_id}'

    def create_room(self, voice_agent_id: str) -> CreatedRoom:
        """Create a pending room and mint its one-time join token."""
        room_id = uuid.uuid4().hex
        join_token = secrets.token_urlsafe(32)
        record = {
            'voice_agent_id': voice_agent_id,
            'join_token_hash': hash_join_token(join_token),
            'status': ROOM_STATUS_PENDING,
            'created_at': time.time(),
            'joined_at': None,
            'pc_id': None,
            'instance_id': None,
        }
        if not self.cache_manager.set_json(
            self._key(room_id), record, expiry=self.room_ttl_secs
        ):
            raise RuntimeError('Failed to store WebRTC room')
        logger.info(f'Created WebRTC room {room_id} for agent {voice_agent_id}')
        return CreatedRoom(
            room_id=room_id, voice_agent_id=voice_agent_id, join_token=join_token
        )

    def consume_join_token(
        self, room_id: str, voice_agent_id: str, join_token: Optional[str]
    ) -> None:
        """Validate the token and atomically mark the room ``joined``.

        Raises ``WebRTCRoomError`` (invalid_join_token | room_expired |
        room_already_joined). A wrong token never consumes the real one.
        """
        if not room_id or not voice_agent_id or not join_token:
            raise room_error(ERR_INVALID_JOIN_TOKEN)

        result = self.cache_manager.run_script(
            _CONSUME_SCRIPT,
            keys=[self._key(room_id)],
            args=[
                hash_join_token(join_token),
                voice_agent_id,
                str(time.time()),
                # Joined rooms live for the max session plus a small buffer
                self.max_session_secs + 60,
            ],
        )
        if not result or result[0] != 'ok':
            code = result[1] if result and len(result) > 1 else ERR_ROOM_EXPIRED
            raise room_error(code if code in _ERROR_STATUS else ERR_ROOM_EXPIRED)

    def attach_connection(self, room_id: str, pc_id: str, instance_id: str) -> None:
        self.cache_manager.run_script(
            _ATTACH_SCRIPT, keys=[self._key(room_id)], args=[pc_id, instance_id]
        )

    def get_room(self, room_id: str) -> Optional[dict]:
        return self.cache_manager.get_json(self._key(room_id))

    def end_room(self, room_id: str) -> None:
        """Remove the room; later offers see ``room_expired``."""
        try:
            self.cache_manager.remove(self._key(room_id))
        except Exception as e:  # cleanup must never raise
            logger.warning(f'Failed to remove WebRTC room {room_id}: {e}')
