"""Typed fake Responses API results consumed by the real Agents SDK runner."""

import copy
import json
from types import SimpleNamespace
from typing import Any

from agents import Model, ModelResponse
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage
from openai.types.responses import ResponseOutputText, ResponseOutputRefusal

from support_agent.agent import ensure_complete


class FakeResponses:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(copy.deepcopy(request))
        return next(self._responses)


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = FakeResponses(responses)


class FakeModel(Model):
    def __init__(self, client: FakeClient) -> None:
        self.client = client

    async def get_response(self, system_instructions, input, model_settings,
                           tools, output_schema, handoffs, tracing, **kwargs):
        response = self.client.responses.create(
            model=getattr(self, 'name', 'openai-test-model'),
            instructions=system_instructions, input=input,
            tools=[{'name': t.name, 'parameters': t.params_json_schema} for t in tools],
            reasoning=model_settings.reasoning.model_dump(exclude_none=True),
            tool_choice=model_settings.tool_choice,
            store=model_settings.store,
        )
        ensure_complete(response)
        return ModelResponse(output=response.output, usage=Usage(), response_id=response.id)

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError('Streaming is not used in these tests.')
        yield


def text_response(text: str, response_id: str = 'response',
                  status: str = 'completed', refusal: bool = False) -> Any:
    block = (ResponseOutputRefusal(type='refusal', refusal='Declined') if refusal
             else ResponseOutputText(type='output_text', text=text, annotations=[]))
    return SimpleNamespace(
        id=response_id, status=status,
        incomplete_details=SimpleNamespace(reason='max_output_tokens'),
        output=[ResponseOutputMessage(id=response_id, type='message',
                role='assistant', status='completed', content=[block])],
        output_text=text,
    )


def tool_call(name: str, call_id: str, arguments: dict[str, Any] | None = None):
    return ResponseFunctionToolCall(
        type='function_call', id=call_id, call_id=call_id,
        name=name, arguments=json.dumps(arguments or {}),
    )


def tool_response(response_id: str, *calls: Any) -> Any:
    return SimpleNamespace(id=response_id, status='completed',
                           incomplete_details=None, output=list(calls), output_text='')
