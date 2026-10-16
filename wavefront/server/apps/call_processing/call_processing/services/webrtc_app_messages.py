"""Browser (WebRTC) app messages for a live voice session.

The browser talks to the running pipeline over Pipecat's RTVI data channel
(``PipelineTask`` enables RTVI by default):

Inbound (browser -> pipeline), via ``client.sendClientMessage(type, data)``:
    ``set_context``  {step, field?, language?, instruction?, allowedTools?}
        Swap the step overlay on the live system prompt, restrict the tools the
        LLM may call, and have the agent open the new step.
    ``interrupt``    {}  stop the bot's current speech (orb tap).

Outbound (pipeline -> browser), via ``RTVIProcessor.send_server_message``:
    ``agent_state``  {state: listening | thinking | speaking}
    ``caption``      {role: 'agent', text, final}  what the bot is saying, as spoken
    ``transcript``   {role: 'user', text, final}   the user's finished utterance

One room is one pipeline: pages only send ``set_context``; nothing here creates
or restarts a conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMMessagesAppendFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import NOT_GIVEN
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import (
    TextPartForConcatenation,
    concatenate_aggregated_text,
    match_endofsentence,
)

from call_processing.constants.language_config import (
    LANGUAGE_DISPLAY_NAMES,
    LANGUAGE_INSTRUCTIONS,
)
from call_processing.log.logger import logger
from call_processing.services.stt_service import STTServiceFactory
from call_processing.services.tts_service import TTSServiceFactory

# Inbound message types
MSG_SET_CONTEXT = 'set_context'
MSG_INTERRUPT = 'interrupt'

# Outbound message types
MSG_AGENT_STATE = 'agent_state'
MSG_CAPTION = 'caption'
MSG_TRANSCRIPT = 'transcript'

STATE_LISTENING = 'listening'
STATE_THINKING = 'thinking'
STATE_SPEAKING = 'speaking'

OVERLAY_START = '<<APP_STEP_CONTEXT>>'
OVERLAY_END = '<<END_APP_STEP_CONTEXT>>'
_OVERLAY_RE = re.compile(
    re.escape(OVERLAY_START) + r'.*?' + re.escape(OVERLAY_END), re.S
)

# Added to the conversation so the agent opens the step it was just moved to
STEP_OPENER = '[The app moved to a new step. Follow the current step instructions and begin it now.]'


# --------------------------------------------------------------------- parsing


@dataclass(frozen=True)
class SetContext:
    step: str
    field: Optional[str] = None
    language: Optional[str] = None
    instruction: Optional[str] = None
    # None = leave the tool set unchanged; empty = no agent tools for this step
    allowed_tools: Optional[Tuple[str, ...]] = None


def _clean_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def parse_set_context(data: Any) -> Optional[SetContext]:
    """Validate a ``set_context`` payload. Returns None when it is unusable."""
    if not isinstance(data, dict):
        return None
    step = _clean_str(data.get('step'))
    if step is None:
        return None

    raw_tools = data.get('allowedTools', data.get('allowed_tools'))
    allowed: Optional[Tuple[str, ...]] = None
    if isinstance(raw_tools, (list, tuple)):
        allowed = tuple(t for t in raw_tools if isinstance(t, str))

    return SetContext(
        step=step,
        field=_clean_str(data.get('field')),
        language=_clean_str(data.get('language')),
        instruction=_clean_str(data.get('instruction')),
        allowed_tools=allowed,
    )


# ------------------------------------------------------------- prompt / tools


def build_step_overlay(ctx: SetContext) -> str:
    """The marked block appended to the system prompt for the current step."""
    where = f'{ctx.step}/{ctx.field}' if ctx.field else ctx.step
    lines = [f'CURRENT APP STEP: {where}']
    if ctx.language:
        name = LANGUAGE_DISPLAY_NAMES.get(ctx.language, ctx.language)
        lines.append(f'Speak only in {name} (language code: {ctx.language}).')
    lines.append(
        ctx.instruction
        or f'You are on step {where}. Only handle this step; do not move on by yourself.'
    )
    return f'{OVERLAY_START}\n' + '\n'.join(lines) + f'\n{OVERLAY_END}'


def strip_step_overlay(content: str) -> str:
    return _OVERLAY_RE.sub('', content).rstrip()


def apply_step_overlay(content: str, overlay: str) -> str:
    """Replace any previous step overlay in ``content`` with ``overlay``."""
    return f'{strip_step_overlay(content)}\n\n{overlay}'


def filter_function_schemas(
    schemas: Sequence[FunctionSchema],
    allowed: Optional[Sequence[str]],
    always_allowed: Sequence[str] = (),
) -> List[FunctionSchema]:
    """Subset of ``schemas`` the LLM may call.

    ``allowed=None`` keeps everything. ``always_allowed`` names (built-ins such
    as language switching) survive an empty allowlist.
    """
    if allowed is None:
        return list(schemas)
    permitted = set(allowed) | set(always_allowed)
    return [s for s in schemas if s.name in permitted]


# --------------------------------------------------------------------- captions


class CaptionAssembler:
    """Turns spoken-text frames into live subtitle updates.

    ``feed`` returns ``(text, final)`` after every frame: the current sentence
    so far (``final=False``), then the completed sentence (``final=True``). The
    next sentence starts fresh, so each caption replaces the previous one.
    """

    def __init__(self) -> None:
        self._parts: List[TextPartForConcatenation] = []

    def feed(
        self, text: str, includes_inter_frame_spaces: bool
    ) -> List[Tuple[str, bool]]:
        if not text:
            return []
        self._parts.append(TextPartForConcatenation(text, includes_inter_frame_spaces))
        joined = concatenate_aggregated_text(self._parts).strip()
        if not joined:
            return []

        updates: List[Tuple[str, bool]] = []
        while joined:
            end = match_endofsentence(joined)
            if not end:
                updates.append((joined, False))
                self._parts = [TextPartForConcatenation(joined, True)]
                return updates
            sentence, joined = joined[:end].strip(), joined[end:].strip()
            if sentence:
                updates.append((sentence, True))

        self._parts = []
        return updates

    def flush(self) -> Optional[Tuple[str, bool]]:
        """Complete whatever is buffered (bot stopped speaking)."""
        if not self._parts:
            return None
        text = concatenate_aggregated_text(self._parts).strip()
        self._parts = []
        return (text, True) if text else None

    def reset(self) -> None:
        self._parts = []


class BotCaptionProcessor(FrameProcessor):
    """Place after ``transport.output()``: emits captions as the bot speaks."""

    def __init__(self, messenger: 'BrowserSessionMessenger', **kwargs):
        super().__init__(**kwargs)
        self._messenger = messenger
        self._assembler = CaptionAssembler()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSTextFrame):
                for text, final in self._assembler.feed(
                    frame.text, frame.includes_inter_frame_spaces
                ):
                    await self._messenger.emit_caption(text, final)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                flushed = self._assembler.flush()
                if flushed:
                    await self._messenger.emit_caption(*flushed)
            elif isinstance(frame, InterruptionFrame):
                self._assembler.reset()

        await self.push_frame(frame, direction)


# -------------------------------------------------------------------- messenger


class BrowserSessionMessenger:
    """Handles browser messages for one live pipeline and emits UI events."""

    def __init__(
        self,
        *,
        context: Any,
        function_schemas: Sequence[FunctionSchema],
        language_state: Dict[str, Any],
        always_allowed_tools: Sequence[str] = (),
        supported_languages: Sequence[str] = (),
        tts_provider: str = '',
        stt_provider: str = '',
        tts_voice_ids: Optional[Dict[str, str]] = None,
    ):
        self._task: Any = None  # set via bind_task once the PipelineTask exists
        self._context = context
        self._function_schemas = list(function_schemas)
        self._language_state = language_state
        self._always_allowed = tuple(always_allowed_tools)
        self._supported_languages = list(supported_languages)
        self._tts_provider = tts_provider
        self._stt_provider = stt_provider
        self._tts_voice_ids = tts_voice_ids or {}
        self._last_context_key: Optional[tuple] = None
        self._last_state: Optional[str] = None

    def bind_task(self, task: Any) -> None:
        """The messenger is needed to build the pipeline; the task comes after."""
        self._task = task

    # ------------------------------------------------------------- outbound

    async def _send(self, payload: Dict[str, Any]) -> None:
        if self._task is None:
            return
        try:
            await self._task.rtvi.send_server_message(payload)
        except Exception as e:  # UI events must never break the call
            logger.warning(f"Failed to send '{payload.get('type')}' to browser: {e}")

    async def emit_agent_state(self, state: str) -> None:
        if state == self._last_state:
            return
        self._last_state = state
        await self._send({'type': MSG_AGENT_STATE, 'state': state})

    async def emit_caption(self, text: str, final: bool) -> None:
        await self._send(
            {'type': MSG_CAPTION, 'role': 'agent', 'text': text, 'final': final}
        )

    async def emit_transcript(self, text: str) -> None:
        if text and text.strip():
            await self._send(
                {'type': MSG_TRANSCRIPT, 'role': 'user', 'text': text, 'final': True}
            )

    # -------------------------------------------------------------- inbound

    async def handle_client_message(self, message: Any) -> None:
        """RTVI ``on_client_message`` handler (``message.type`` / ``message.data``)."""
        msg_type = getattr(message, 'type', None)
        data = getattr(message, 'data', None)
        if self._task is None:
            return
        try:
            if msg_type == MSG_SET_CONTEXT:
                ctx = parse_set_context(data)
                if ctx is None:
                    logger.warning(f'Ignoring invalid set_context payload: {data!r}')
                    return
                await self.apply_context(ctx)
            elif msg_type == MSG_INTERRUPT:
                await self._interrupt()
        except Exception as e:
            logger.error(
                f'Error handling browser message {msg_type}: {e}', exc_info=True
            )

    async def _interrupt(self) -> None:
        await self._task.rtvi.interrupt_bot()

    async def apply_context(self, ctx: SetContext) -> None:
        key = (ctx.step, ctx.field, ctx.language, ctx.instruction, ctx.allowed_tools)
        if key == self._last_context_key:
            return  # duplicate (e.g. effect re-run); nothing to change
        first = self._last_context_key is None
        self._last_context_key = key

        overlay = build_step_overlay(ctx)
        await self._apply_prompt(ctx, overlay)
        self._apply_tools(ctx)
        logger.info(
            f'Browser set_context step={ctx.step} field={ctx.field} '
            f'language={ctx.language} tools={ctx.allowed_tools}'
        )

        # The first context lets the welcome message finish; later ones cut the
        # previous step's speech off before the agent opens the new step.
        if not first:
            await self._interrupt()
        await self._task.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{'role': 'user', 'content': STEP_OPENER}], run_llm=True
            )
        )

    async def _apply_prompt(self, ctx: SetContext, overlay: str) -> None:
        messages = list(self._context.get_messages())
        original = self._language_state.get('original_system_prompt')

        if original:
            # Multi-language agent: the language tool rebuilds the system prompt
            # from ``original_system_prompt``, so the overlay must live there too.
            new_original = apply_step_overlay(original, overlay)
            self._language_state['original_system_prompt'] = new_original
            language = self._resolve_language(ctx.language)
            instruction = LANGUAGE_INSTRUCTIONS.get(
                language, LANGUAGE_INSTRUCTIONS.get('en', 'Respond in English.')
            )
            content = f'{instruction}\n\n{new_original}'
            await self._switch_services_language(language)
        else:
            current = ''
            if messages and messages[0].get('role') == 'system':
                existing = messages[0].get('content')
                current = existing if isinstance(existing, str) else ''
            content = apply_step_overlay(current, overlay)

        system_message = {'role': 'system', 'content': content}
        if messages and messages[0].get('role') == 'system':
            messages[0] = system_message
        else:
            messages.insert(0, system_message)
        self._context.set_messages(messages)

    def _resolve_language(self, requested: Optional[str]) -> str:
        current = self._language_state.get('current_language', 'en')
        if requested and requested in self._supported_languages:
            return requested
        return current

    async def _switch_services_language(self, language: str) -> None:
        """Move TTS/STT to ``language`` (multi-language agents only)."""
        current = self._language_state.get('current_language')
        if language == current:
            return
        tts_frame = TTSServiceFactory.create_language_update_frame(
            self._tts_provider, language, self._tts_voice_ids.get(language)
        )
        if tts_frame:
            await self._task.queue_frame(tts_frame)
        stt_frame = STTServiceFactory.create_language_update_frame(
            self._stt_provider, language
        )
        if stt_frame:
            await self._task.queue_frame(stt_frame)
        self._language_state['current_language'] = language
        self._language_state['switch_count'] = (
            self._language_state.get('switch_count', 0) + 1
        )

    def _apply_tools(self, ctx: SetContext) -> None:
        if ctx.allowed_tools is None:
            return
        subset = filter_function_schemas(
            self._function_schemas, ctx.allowed_tools, self._always_allowed
        )
        # An empty tools list is rejected by some LLM APIs; NOT_GIVEN disables tools
        self._context.set_tools(
            ToolsSchema(standard_tools=subset) if subset else NOT_GIVEN
        )
