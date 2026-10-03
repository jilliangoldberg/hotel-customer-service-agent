"""Verify real SDK/API serialization against an in-memory HTTP transport."""

import json
from concurrent.futures import ThreadPoolExecutor

import httpx2
from openai import AsyncOpenAI, OpenAI
import pytest

from support_agent import agent as agent_module
from support_agent.agent import AgentLoopError, SupportAgent
from support_agent.config import PROJECT_ROOT
from support_agent.tools import HotelTools


def response_payload(output, status='completed'):
    return {'id': 'resp_test', 'object': 'response', 'created_at': 1,
            'model': 'gpt-5.4', 'status': status, 'output': output,
            'incomplete_details': {'reason': 'max_output_tokens'} if status == 'incomplete' else None}


def message(text):
    return {'id': 'msg_test', 'type': 'message', 'role': 'assistant',
            'status': 'completed', 'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}


def build_with_transport(monkeypatch, payloads):
    requests = []
    responses = iter(payloads)
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=next(responses))
    def async_client(**kwargs):
        return AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr(agent_module, 'AsyncOpenAI', async_client)
    client = OpenAI(api_key='test-key')
    agent = SupportAgent(client=client, model='gpt-5.4',
                         tools=HotelTools(PROJECT_ROOT / 'data', 'a-test-secret-value'))
    return agent, requests, client


def test_real_sdk_sends_openai_function_schemas_and_correlated_results(monkeypatch):
    args = {'email': 'morgan.lee@example.com', 'reservation_number': '#H002'}
    agent, requests, client = build_with_transport(monkeypatch, [
        response_payload([{'id': 'fc_test', 'type': 'function_call', 'call_id': 'call_test',
                           'name': 'lookup_reservation', 'arguments': json.dumps(args)}]),
        response_payload([message('Your reservation is confirmed.')]),
        response_payload([message('Enjoy your stay!')]),
    ])
    try:
        assert agent.reply('Check my reservation') == 'Your reservation is confirmed.'
        # The same conversation can move between Flask request threads.
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(agent.reply, 'Thanks').result() == 'Enjoy your stay!'
    finally:
        client.close()
    first = requests[0]
    assert first['model'] == 'gpt-5.4'
    assert first['reasoning'] == {'effort': 'medium'}
    assert first['store'] is False
    assert all(t['type'] == 'function' and t['strict'] for t in first['tools'])
    result = next(i for i in requests[1]['input'] if i.get('type') == 'function_call_output')
    assert result['call_id'] == 'call_test'
    assert json.loads(result['output'])['reservation_number'] == '#H002'


def test_raw_incomplete_status_is_rejected_even_when_message_looks_complete(monkeypatch):
    agent, _, client = build_with_transport(monkeypatch, [response_payload([message('Partial reply')], 'incomplete')])
    try:
        with pytest.raises(AgentLoopError, match='max_output_tokens'):
            agent.reply('Hello')
        assert agent._messages == []
    finally:
        client.close()
