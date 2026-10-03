"""Exercise the actual Agents SDK runner with a deterministic fake model."""

import json
import pytest

from fakes import FakeClient, FakeModel, text_response, tool_call, tool_response
from support_agent.agent import AgentLoopError, SupportAgent
from support_agent.config import PROJECT_ROOT
from support_agent.policy import DEFAULT_POLICY
from support_agent.prompt import EARLY_RISERS_PROMPT, SYSTEM_PROMPT
from support_agent.tools import HotelTools


def make_agent(responses, max_tool_rounds=5):
    client = FakeClient(responses)
    agent = SupportAgent(client=client, model='openai-test-model',
                         tools=HotelTools(PROJECT_ROOT / 'data', 'a-test-secret-value'),
                         sdk_model=FakeModel(client), max_tool_rounds=max_tool_rounds)
    return agent, client


def test_continues_customer_history_and_passes_model_settings():
    agent, client = make_agent([text_response('Welcome!', 'r1'), text_response('Enjoy your stay!', 'r2')])
    assert agent.reply('Hello') == 'Welcome!'
    assert agent.reply('Thanks') == 'Enjoy your stay!'
    first, second = client.responses.requests
    assert first['instructions'] == SYSTEM_PROMPT
    assert first['reasoning'] == {'effort': 'medium'}
    assert first['store'] is False
    assert first['input'] == [{'role': 'user', 'content': 'Hello'}]
    assert second['input'][-1] == {'role': 'user', 'content': 'Thanks'}
    assert any('Welcome!' in json.dumps(item) for item in second['input'])


def test_promotion_tools_remain_hidden_until_guest_mentions_promotion():
    agent, client = make_agent([text_response('Hello'), text_response('Here are the hours'), text_response('Let me check')])
    agent.reply('Hello')
    assert agent.last_trace['early_risers_guard'] == 'hidden'
    agent.reply('How does Early Risers work?')
    agent.reply('Can I get a code?')
    assert {t['name'] for t in client.responses.requests[0]['tools']} == {'lookup_reservation', 'get_available_rooms'}
    assert EARLY_RISERS_PROMPT not in client.responses.requests[0]['instructions']
    for request in client.responses.requests[1:]:
        assert EARLY_RISERS_PROMPT in request['instructions']
        assert 'create_early_risers_code' in {t['name'] for t in request['tools']}


def test_sdk_dispatches_parallel_tools_and_returns_correlated_results():
    args = {'email': 'morgan.lee@example.com', 'reservation_number': '#H002'}
    agent, client = make_agent([
        tool_response('r1', tool_call('lookup_reservation', 'c1', args), tool_call('get_available_rooms', 'c2')),
        text_response('Your reservation is confirmed.', 'r2')])
    assert agent.reply('Check my reservation and rooms') == 'Your reservation is confirmed.'
    inputs = client.responses.requests[1]['input']
    outputs = [item for item in inputs if item.get('type') == 'function_call_output']
    assert {item['call_id'] for item in outputs} == {'c1', 'c2'}
    assert json.loads(outputs[0]['output'])['found'] is True
    assert agent.last_trace['tool_rounds'] == 1
    assert [t['outcome'] for t in agent.last_trace['tools']] == ['found', 'completed']
    trace = json.dumps(agent.last_trace)
    assert 'morgan.lee@example.com' not in trace
    assert '#H002' not in trace


def test_rejected_tool_result_is_available_to_model():
    agent, client = make_agent([tool_response('r1', tool_call('lookup_reservation', 'c1', {'email': 'bad', 'reservation_number': '#H001'})), text_response('Could you double-check both details?')])
    agent.reply('Check it')
    outputs = [i for i in client.responses.requests[1]['input'] if i.get('type') == 'function_call_output']
    assert json.loads(outputs[0]['output'])['ok'] is False
    assert agent.last_trace['tools'][0]['outcome'] == 'rejected'


def test_tool_round_limit_rolls_back_the_failed_turn():
    agent, client = make_agent([text_response('Welcome', 'r0'), tool_response('r1', tool_call('get_available_rooms', 'c1')), tool_response('r2', tool_call('get_available_rooms', 'c2')), text_response('Try again', 'r3')], max_tool_rounds=1)
    agent.reply('Hello')
    with pytest.raises(AgentLoopError, match='tool-call limit'):
        agent.reply('Keep calling tools')
    assert agent.last_trace['outcome'] == 'failed'
    assert len(agent.last_trace['tools']) == 1
    assert agent.last_trace['tool_rounds'] == 1
    assert agent.reply('Retry') == 'Try again'
    inputs = client.responses.requests[-1]['input']
    assert 'Keep calling tools' not in json.dumps(inputs)
    assert 'Welcome' in json.dumps(inputs)


@pytest.mark.parametrize('response, message', [(text_response('partial', status='incomplete'), 'max_output_tokens'), (text_response('', refusal=True), 'declined'), (text_response(''), 'no text')])
def test_rejects_incomplete_refused_and_empty_responses(response, message):
    agent, client = make_agent([response, text_response('Hello')])
    with pytest.raises(AgentLoopError, match=message):
        agent.reply('Tell me about Early Risers')
    agent.reply('Hello')
    assert EARLY_RISERS_PROMPT not in client.responses.requests[-1]['instructions']
    assert client.responses.requests[-1]['input'] == [{'role': 'user', 'content': 'Hello'}]


def test_policy_rewrite_has_no_tools_and_preserves_promotion_guidance():
    agent, client = make_agent([text_response("I'll contact a manager and refund your booking.", 'r1'), text_response('I can check your reservation.', 'r2'), text_response('Welcome back', 'r3')])
    assert agent.reply('Early Risers and refund help please') == 'I can check your reservation.'
    rewrite = client.responses.requests[1]
    assert rewrite['tool_choice'] == 'none'
    assert rewrite['tools'] == []
    assert EARLY_RISERS_PROMPT in rewrite['instructions']
    assert agent.last_trace['response_policy'] == 'rewritten'
    agent.reply('Thanks')
    history = json.dumps(client.responses.requests[-1]['input'])
    assert 'contact a manager' not in history
    assert 'I can check your reservation.' in history


def test_safe_fallback_replaces_both_unsafe_drafts():
    agent, client = make_agent([text_response('I will cancel your booking.', 'r1'), text_response("I've contacted staff.", 'r2'), text_response('Hello', 'r3')])
    assert agent.reply('Cancel it') == DEFAULT_POLICY.fallback
    assert agent.last_trace['response_policy'] == 'fallback'
    agent.reply('Hello')
    assert client.responses.requests[-1]['input'] == [{'role': 'user', 'content': 'Cancel it'}, {'role': 'assistant', 'content': DEFAULT_POLICY.fallback}, {'role': 'user', 'content': 'Hello'}]


def test_trace_returns_detached_tool_records():
    agent, _ = make_agent([tool_response('r1', tool_call('get_available_rooms', 'c1')), text_response('Here are the rooms')])
    agent.reply('Rooms please')
    trace = agent.last_trace
    trace['tools'][0]['name'] = 'changed'
    assert agent.last_trace['tools'][0]['name'] == 'get_available_rooms'
