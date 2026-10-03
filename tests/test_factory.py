"""Verify the shared composition root and configured OpenAI model."""

from fakes import FakeClient, FakeModel, text_response
from support_agent import factory
from support_agent.agent import SupportAgent
from support_agent.config import PROJECT_ROOT, Settings


def test_build_agent_composes_configured_model_and_registered_tools(monkeypatch):
    client = FakeClient([text_response('Hello from Trailhead Hotel.')])
    keys = []
    def make_client(**kwargs):
        keys.append(kwargs['api_key'])
        return client
    monkeypatch.setattr(factory, 'OpenAI', make_client)
    def make_agent(**kwargs):
        model = FakeModel(client)
        model.name = kwargs['model']
        return SupportAgent(**kwargs, sdk_model=model)
    monkeypatch.setattr(factory, 'SupportAgent', make_agent)
    settings = Settings(api_key='test-key', agent_model='agent-model',
                        eval_model='judge-model', agent_effort='low',
                        promotion_secret='a-valid-promotion-secret', data_dir=PROJECT_ROOT / 'data')
    agent = factory.build_agent(settings)
    assert agent.reply('Hello') == 'Hello from Trailhead Hotel.'
    assert keys == ['test-key']
    request = client.responses.requests[0]
    assert request['model'] == 'agent-model'
    assert request['reasoning'] == {'effort': 'low'}
    assert {t['name'] for t in request['tools']} == {'lookup_reservation', 'get_available_rooms'}
