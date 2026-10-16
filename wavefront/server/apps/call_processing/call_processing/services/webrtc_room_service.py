"""Redis-backed registry of browser (WebRTC) rooms.

A room is created by floware for an active voice agent. The browser then joins
it with a one-time ``join_token``. Only the SHA-256 hash of the token is stored;
the plaintext is returned once, on creation.

Room record (``webrtc_room:{room_id}``), JSON:
    voice_agent_id, join_token_hash, status (pending | joined | ended),
    created_at, joined_at, pc_id, instance_id,
    allowed_agent_ids   agents the browser may hot-swap to (always includes
                        ``voice_agent_id``); set by floware, never by the browser
    active_voice_agent_id  the agent currently serving the room
    swap_count          number of completed agent swaps
    room_context_handles_language  true when the app drives the language via
                        ``set_context.language`` (no language rules / tool for
                        the agent); absent or false keeps native switching

Peer connections themselves are NOT stored here (aiortc state is in-process);
``pc_id`` / ``instance_id`` are kept for diagnostics and sticky-routing errors.
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

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

# Stable error codes for a ``switch_agent`` request (sent as ``agent_switch_failed``)
ERR_AGENT_NOT_ALLOWED = 'agent_not_allowed'
ERR_AGENT_UNAVAILABLE = 'agent_unavailable'
ERR_SWAP_IN_PROGRESS = 'swap_in_progress'
ERR_SWAP_LIMIT_REACHED = 'swap_limit_reached'
ERR_ALREADY_ACTIVE = 'already_active'
ERR_SWAP_FAILED = 'swap_failed'
ERR_INVALID_REQUEST = 'invalid_request'

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

# Record a completed agent swap, keeping the room TTL. Refuses agents that are not
# on the room's allowlist, so the record can never point outside it.
# ARGV[1] agent_id
# Returns {'ok', <swap_count>} or {'err', <code>}
_RECORD_SWAP_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then
    return {'err', 'room_expired'}
end
local room = cjson.decode(raw)
if room['status'] ~= 'joined' then
    return {'err', 'room_expired'}
end
local allowed = false
if type(room['allowed_agent_ids']) == 'table' then
    for _, id in ipairs(room['allowed_agent_ids']) do
        if id == ARGV[1] then
            allowed = true
        end
    end
end
if not allowed then
    return {'err', 'agent_not_allowed'}
end
room['active_voice_agent_id'] = ARGV[1]
room['swap_count'] = (tonumber(room['swap_count']) or 0) + 1
redis.call('SET', KEYS[1], cjson.encode(room), 'KEEPTTL')
return {'ok', room['swap_count']}
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

    def create_room(
        self,
        voice_agent_id: str,
        allowed_agent_ids: Optional[List[str]] = None,
        room_context_handles_language: bool = False,
    ) -> CreatedRoom:
        """Create a pending room and mint its one-time join token.

        ``allowed_agent_ids`` is the set of agents the browser may hot-swap to.
        The initial agent is always included; ``voice_agent_id`` itself never
        changes (the join-token consume script depends on it).

        ``room_context_handles_language`` makes the app (``set_context.language``)
        the only thing that changes the language, for every agent in this room.
        """
        room_id = uuid.uuid4().hex
        join_token = secrets.token_urlsafe(32)
        allowed = [voice_agent_id]
        for agent_id in allowed_agent_ids or []:
            if agent_id and agent_id not in allowed:
                allowed.append(agent_id)
        record = {
            'voice_agent_id': voice_agent_id,
            'allowed_agent_ids': allowed,
            'active_voice_agent_id': voice_agent_id,
            'swap_count': 0,
            'room_context_handles_language': bool(room_context_handles_language),
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

    def record_swap(self, room_id: str, agent_id: str) -> Optional[int]:
        """Record that the room now runs ``agent_id``. Returns the new swap count.

        Returns None if the room is gone or ``agent_id`` is not allowed. Never
        raises: the live session must not fail because bookkeeping did.
        """
        try:
            result = self.cache_manager.run_script(
                _RECORD_SWAP_SCRIPT, keys=[self._key(room_id)], args=[agent_id]
            )
        except Exception as e:
            logger.warning(f'Failed to record swap for room {room_id}: {e}')
            return None
        if not result or result[0] != 'ok':
            return None
        return int(result[1])

    def get_allowed_agent_ids(self, room_id: str) -> List[str]:
        """The room's swap allowlist (empty if the room is gone)."""
        room = self.get_room(room_id) or {}
        allowed = room.get('allowed_agent_ids')
        return [str(a) for a in allowed] if isinstance(allowed, list) else []

    def get_room_context_handles_language(self, room_id: str) -> bool:
        """Whether the app drives this room's language (False if unset/gone)."""
        room = self.get_room(room_id) or {}
        return room.get('room_context_handles_language') is True

    def get_room(self, room_id: str) -> Optional[dict]:
        return self.cache_manager.get_json(self._key(room_id))

    def end_room(self, room_id: str) -> None:
        """Remove the room; later offers see ``room_expired``."""
        try:
            self.cache_manager.remove(self._key(room_id))
        except Exception as e:  # cleanup must never raise
            logger.warning(f'Failed to remove WebRTC room {room_id}: {e}')
