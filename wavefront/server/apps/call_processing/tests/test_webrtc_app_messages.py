"""Browser app messages: set_context parsing, step overlay, tool allowlist, captions."""

from types import SimpleNamespace

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext

from call_processing.services.webrtc_app_messages import (
    OVERLAY_END,
    OVERLAY_START,
    CaptionAssembler,
    BrowserSessionMessenger,
    SetContext,
    apply_step_overlay,
    build_step_overlay,
    filter_function_schemas,
    parse_set_context,
    parse_switch_agent,
    strip_step_overlay,
)


def _schema(name: str) -> FunctionSchema:
    return FunctionSchema(name=name, description=name, properties={}, required=[])


# ------------------------------------------------------------------- parsing


def test_parse_set_context_full_payload():
    ctx = parse_set_context(
        {
            'type': 'set_context',
            'step': 'identity',
            'field': 'name',
            'language': 'hi',
            'instruction': 'Ask for the name.',
            'allowedTools': ['verify_identity_field'],
        }
    )
    assert ctx == SetContext(
        step='identity',
        field='name',
        language='hi',
        instruction='Ask for the name.',
        allowed_tools=('verify_identity_field',),
    )


def test_parse_set_context_requires_step():
    assert parse_set_context({'field': 'name'}) is None
    assert parse_set_context({'step': '  '}) is None
    assert parse_set_context(None) is None
    assert parse_set_context('identity') is None


def test_parse_set_context_tool_allowlist_semantics():
    assert parse_set_context({'step': 'a'}).allowed_tools is None
    assert parse_set_context({'step': 'a', 'allowedTools': []}).allowed_tools == ()
    assert parse_set_context({'step': 'a', 'allowedTools': 'x'}).allowed_tools is None
    assert parse_set_context(
        {'step': 'a', 'allowedTools': ['x', 3, None]}
    ).allowed_tools == ('x',)


# --------------------------------------------------------------- tool filter


def test_filter_function_schemas():
    schemas = [_schema('a'), _schema('b'), _schema('detect_and_switch_language')]

    assert [s.name for s in filter_function_schemas(schemas, None)] == [
        'a',
        'b',
        'detect_and_switch_language',
    ]
    assert [s.name for s in filter_function_schemas(schemas, ['b'])] == ['b']
    assert filter_function_schemas(schemas, []) == []
    assert [
        s.name
        for s in filter_function_schemas(
            schemas, [], always_allowed=['detect_and_switch_language']
        )
    ] == ['detect_and_switch_language']


# ------------------------------------------------------------------- overlay


def test_overlay_uses_instruction_and_language():
    overlay = build_step_overlay(
        SetContext(step='identity', field='dob', language='hi', instruction='Ask DOB.')
    )
    assert overlay.startswith(OVERLAY_START) and overlay.endswith(OVERLAY_END)
    assert 'identity/dob' in overlay
    assert 'Hindi' in overlay
    assert 'Ask DOB.' in overlay


def test_overlay_falls_back_without_instruction():
    overlay = build_step_overlay(SetContext(step='face'))
    assert 'step face' in overlay


def test_apply_overlay_replaces_previous_overlay():
    base = 'You are a KYC agent.'
    first = apply_step_overlay(base, build_step_overlay(SetContext(step='language')))
    second = apply_step_overlay(first, build_step_overlay(SetContext(step='identity')))

    assert second.startswith(base)
    assert second.count(OVERLAY_START) == 1
    assert 'identity' in second and 'language' not in second
    assert strip_step_overlay(second) == base


# ------------------------------------------------------------------ captions


def test_caption_assembler_grows_then_finalizes_each_sentence():
    assembler = CaptionAssembler()

    assert assembler.feed('Hello', False) == [('Hello', False)]
    assert assembler.feed('there.', False) == [('Hello there.', True)]
    # The next sentence starts fresh (replaces the previous caption in the UI)
    assert assembler.feed('How are', False) == [('How are', False)]
    assert assembler.flush() == ('How are', True)
    assert assembler.flush() is None


def test_caption_assembler_reset_drops_buffer():
    assembler = CaptionAssembler()
    assembler.feed('Hello', False)
    assembler.reset()
    assert assembler.flush() is None


# ----------------------------------------------------------------- messenger


class FakeRTVI:
    def __init__(self):
        self.sent = []
        self.interrupts = 0

    async def send_server_message(self, data):
        self.sent.append(data)

    async def interrupt_bot(self):
        self.interrupts += 1


class FakeTask:
    def __init__(self):
        self.rtvi = FakeRTVI()
        self.frames = []

    async def queue_frame(self, frame):
        self.frames.append(frame)


def _messenger(tools=('tool_a', 'tool_b'), language_state=None, **kwargs):
    context = LLMContext(
        [{'role': 'system', 'content': 'BASE PROMPT'}],
    )
    messenger = BrowserSessionMessenger(
        context=context,
        function_schemas=[_schema(n) for n in tools],
        language_state=language_state
        if language_state is not None
        else {
            'current_language': 'en',
            'switch_count': 0,
            'original_system_prompt': '',
        },
        **kwargs,
    )
    task = FakeTask()
    messenger.bind_task(task)
    return messenger, context, task


def _msg(type_, data=None):
    return SimpleNamespace(type=type_, data=data)


async def test_set_context_overlays_prompt_and_restricts_tools():
    messenger, context, task = _messenger()

    await messenger.handle_client_message(
        _msg(
            'set_context',
            {
                'step': 'identity',
                'field': 'name',
                'language': 'hi',
                'instruction': 'Ask for the name only.',
                'allowedTools': ['tool_b'],
            },
        )
    )

    system = context.get_messages()[0]['content']
    assert system.startswith('BASE PROMPT')
    assert 'Ask for the name only.' in system
    assert [t.name for t in context.tools.standard_tools] == ['tool_b']
    # First context: no interruption, but the agent is told to open the step
    assert task.rtvi.interrupts == 0
    assert isinstance(task.frames[-1], LLMMessagesAppendFrame)
    assert task.frames[-1].run_llm is True


async def test_empty_allowlist_disables_tools_and_missing_allowlist_keeps_them():
    messenger, context, _ = _messenger()

    await messenger.handle_client_message(
        _msg('set_context', {'step': 'face', 'allowedTools': []})
    )
    assert context.tools is NOT_GIVEN

    await messenger.handle_client_message(_msg('set_context', {'step': 'house'}))
    # No allowlist -> tool set is left as the previous step set it
    assert context.tools is NOT_GIVEN


async def test_later_context_swaps_overlay_and_interrupts_bot():
    messenger, context, task = _messenger()

    await messenger.handle_client_message(
        _msg('set_context', {'step': 'language', 'instruction': 'Ask language.'})
    )
    await messenger.handle_client_message(
        _msg('set_context', {'step': 'identity', 'instruction': 'Ask name.'})
    )

    system = context.get_messages()[0]['content']
    assert 'Ask name.' in system and 'Ask language.' not in system
    assert system.count(OVERLAY_START) == 1
    assert task.rtvi.interrupts == 1


async def test_duplicate_context_is_ignored():
    messenger, _, task = _messenger()
    payload = {'step': 'language', 'instruction': 'Ask language.'}

    await messenger.handle_client_message(_msg('set_context', payload))
    await messenger.handle_client_message(_msg('set_context', payload))

    assert len(task.frames) == 1


async def test_invalid_context_changes_nothing():
    messenger, context, task = _messenger()

    await messenger.handle_client_message(_msg('set_context', {'field': 'name'}))

    assert context.get_messages()[0]['content'] == 'BASE PROMPT'
    assert task.frames == []


async def test_interrupt_message_interrupts_bot():
    messenger, _, task = _messenger()

    await messenger.handle_client_message(_msg('interrupt'))

    assert task.rtvi.interrupts == 1


async def test_multi_language_overlay_survives_in_original_prompt():
    state = {
        'current_language': 'en',
        'switch_count': 0,
        'original_system_prompt': 'BASE PROMPT',
    }
    messenger, context, _ = _messenger(
        language_state=state,
        supported_languages=['en', 'hi'],
        tts_provider='unknown-provider',
        stt_provider='unknown-provider',
    )

    await messenger.handle_client_message(
        _msg('set_context', {'step': 'identity', 'language': 'hi', 'instruction': 'X.'})
    )

    system = context.get_messages()[0]['content']
    assert system.startswith('Respond in Hindi')
    assert OVERLAY_START in state['original_system_prompt']
    assert state['current_language'] == 'hi'


async def test_room_context_language_adds_no_language_instruction_prefix():
    state = {
        'current_language': 'en',
        'switch_count': 0,
        'original_system_prompt': 'BASE PROMPT',
    }
    messenger, context, task = _messenger(
        language_state=state,
        supported_languages=['en', 'hi'],
        tts_provider='unknown-provider',
        stt_provider='unknown-provider',
        room_context_language=True,
    )

    await messenger.handle_client_message(
        _msg('set_context', {'step': 'identity', 'language': 'hi', 'instruction': 'X.'})
    )

    system = context.get_messages()[0]['content']
    assert system.startswith('BASE PROMPT')
    assert 'Respond in' not in system
    assert 'language code: hi' in system
    assert state['current_language'] == 'hi'  # STT/TTS still follow the app


async def test_agent_state_dedupes_and_events_use_server_messages():
    messenger, _, task = _messenger()

    await messenger.emit_agent_state('speaking')
    await messenger.emit_agent_state('speaking')
    await messenger.emit_caption('Hello', False)
    await messenger.emit_transcript('my name is Asha')
    await messenger.emit_transcript('   ')

    assert task.rtvi.sent == [
        {'type': 'agent_state', 'state': 'speaking'},
        {'type': 'caption', 'role': 'agent', 'text': 'Hello', 'final': False},
        {
            'type': 'transcript',
            'role': 'user',
            'text': 'my name is Asha',
            'final': True,
        },
    ]


# --------------------------------------------------------------- switch_agent


def test_parse_switch_agent():
    assert parse_switch_agent({'agent_id': ' abc '}) == 'abc'
    assert parse_switch_agent({'agentId': 'abc'}) == 'abc'
    for bad in (
        None,
        'abc',
        [],
        {},
        {'agent_id': ''},
        {'agent_id': '  '},
        {'agent_id': 5},
    ):
        assert parse_switch_agent(bad) is None


async def test_switch_agent_is_delegated_to_the_handler():
    messenger, _, task = _messenger()
    received = []

    async def handler(agent_id):
        received.append(agent_id)

    messenger.set_switch_handler(handler)
    await messenger.handle_client_message(_msg('switch_agent', {'agent_id': 'a-1'}))

    assert received == ['a-1']
    assert task.rtvi.sent == []


async def test_switch_agent_bad_payload_is_rejected_without_calling_handler():
    messenger, _, task = _messenger()
    received = []

    async def handler(agent_id):
        received.append(agent_id)

    messenger.set_switch_handler(handler)
    await messenger.handle_client_message(_msg('switch_agent', {'agent_id': ''}))
    await messenger.handle_client_message(_msg('switch_agent', 'nope'))

    assert received == []
    assert [m['code'] for m in task.rtvi.sent] == ['invalid_request'] * 2
    assert all(m['type'] == 'agent_switch_failed' for m in task.rtvi.sent)


async def test_switch_agent_without_handler_fails_cleanly():
    messenger, _, task = _messenger()

    await messenger.handle_client_message(_msg('switch_agent', {'agent_id': 'a-1'}))

    assert task.rtvi.sent[0]['type'] == 'agent_switch_failed'
    assert task.rtvi.sent[0]['code'] == 'swap_failed'


async def test_switch_events_payloads():
    messenger, _, task = _messenger()

    await messenger.emit_agent_switching('a-2')
    await messenger.emit_agent_changed('a-2', 'Second')

    assert task.rtvi.sent == [
        {'type': 'agent_switching', 'agent_id': 'a-2'},
        {'type': 'agent_changed', 'agent_id': 'a-2', 'agent_name': 'Second'},
    ]


async def test_language_changed_event_payload():
    messenger, _, task = _messenger()

    await messenger.emit_language_changed('hi')

    assert task.rtvi.sent == [{'type': 'language_changed', 'language': 'hi'}]


async def test_set_context_during_swap_is_deferred_and_applied_to_the_new_context():
    messenger, old_context, task = _messenger()
    await messenger.handle_client_message(
        _msg('set_context', {'step': 'language', 'instruction': 'Ask language.'})
    )
    assert messenger.will_reopen_step()

    messenger.begin_swap()
    await messenger.handle_client_message(
        _msg('set_context', {'step': 'identity', 'instruction': 'Ask name.'})
    )
    # The outgoing context is untouched while swapping
    assert 'Ask name.' not in old_context.get_messages()[0]['content']

    new_context = LLMContext([{'role': 'system', 'content': 'NEW PROMPT'}])
    messenger.rebind(
        context=new_context,
        function_schemas=[_schema('tool_a')],
        language_state={
            'current_language': 'en',
            'switch_count': 0,
            'original_system_prompt': '',
        },
    )
    await messenger.finish_swap()

    system = new_context.get_messages()[0]['content']
    assert system.startswith('NEW PROMPT') and 'Ask name.' in system
    assert 'Ask language.' not in system
    assert 'Ask name.' not in old_context.get_messages()[0]['content']


async def test_finish_swap_reapplies_the_last_step_on_the_new_context():
    messenger, _, _ = _messenger()
    await messenger.handle_client_message(
        _msg('set_context', {'step': 'identity', 'instruction': 'Ask name.'})
    )

    messenger.begin_swap()
    new_context = LLMContext([{'role': 'system', 'content': 'NEW PROMPT'}])
    messenger.rebind(
        context=new_context,
        function_schemas=[_schema('tool_a')],
        language_state={
            'current_language': 'en',
            'switch_count': 0,
            'original_system_prompt': '',
        },
    )
    await messenger.finish_swap()

    assert 'Ask name.' in new_context.get_messages()[0]['content']


async def test_finish_swap_without_any_step_does_nothing():
    messenger, _, task = _messenger()
    messenger.begin_swap()
    assert not messenger.will_reopen_step()

    await messenger.finish_swap()

    assert task.frames == []
