"""Tests for the agent loop using a fake OpenAI client."""

from types import SimpleNamespace
from typing import Any

import pytest

from sierra_agent.agent import AgentLoopError, SierraAgent


class FakeResponses:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        return next(self._responses)


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = FakeResponses(responses)


class FakeTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def execute(self, name: str, arguments: str) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"ok": True, "tool": name}


def text_response(response_id: str, text: str) -> Any:
    return SimpleNamespace(id=response_id, output=[], output_text=text)


def tool_response(response_id: str, *calls: Any) -> Any:
    return SimpleNamespace(id=response_id, output=list(calls), output_text="")


def tool_call(name: str, call_id: str, arguments: str = "{}") -> Any:
    return SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


def make_agent(
    responses: list[Any],
    max_tool_rounds: int = 5,
) -> tuple[SierraAgent, FakeClient, FakeTools]:
    client = FakeClient(responses)
    tools = FakeTools()
    agent = SierraAgent(
        client=client,  # type: ignore[arg-type]
        model="gpt-4o-mini",
        tools=tools,  # type: ignore[arg-type]
        max_tool_rounds=max_tool_rounds,
    )
    return agent, client, tools


def test_returns_plain_model_response_and_continues_session() -> None:
    agent, client, _ = make_agent(
        [
            text_response("response-1", "Welcome to the trail!"),
            text_response("response-2", "Onward!"),
        ]
    )

    assert agent.reply("Hello") == "Welcome to the trail!"
    assert agent.reply("Thanks") == "Onward!"

    assert "previous_response_id" not in client.responses.requests[0]
    assert client.responses.requests[1]["previous_response_id"] == "response-1"


def test_executes_tool_and_returns_output_to_model() -> None:
    call = tool_call(
        "lookup_reservation",
        "call-1",
        '{"email":"hiker@example.com","reservation_number":"#H001"}',
    )
    agent, client, tools = make_agent(
        [
            tool_response("response-1", call),
            text_response("response-2", "Your reservation is confirmed."),
        ]
    )

    result = agent.reply("Where is my reservation?")

    assert result == "Your reservation is confirmed."
    assert tools.calls == [("lookup_reservation", call.arguments)]
    output = client.responses.requests[1]["input"][0]
    assert output["type"] == "function_call_output"
    assert output["call_id"] == "call-1"
    assert '"ok":true' in output["output"]


def test_handles_multiple_tool_calls_from_one_response() -> None:
    first = tool_call("lookup_reservation", "call-1")
    second = tool_call("get_available_rooms", "call-2")
    agent, client, tools = make_agent(
        [
            tool_response("response-1", first, second),
            text_response("response-2", "Here are both answers."),
        ]
    )

    agent.reply("Check my reservation and recommend a room.")

    assert [name for name, _ in tools.calls] == [
        "lookup_reservation",
        "get_available_rooms",
    ]
    assert len(client.responses.requests[1]["input"]) == 2


def test_failed_tool_loop_keeps_last_successful_session_checkpoint() -> None:
    agent, client, _ = make_agent(
        [
            text_response("response-1", "Welcome to the trail!"),
            tool_response("response-2", tool_call("lookup_reservation", "call-1")),
            tool_response("response-3", tool_call("lookup_reservation", "call-2")),
            text_response("response-4", "Let's try that again."),
        ],
        max_tool_rounds=1,
    )

    agent.reply("Hello")

    with pytest.raises(AgentLoopError, match="tool-call limit"):
        agent.reply("Keep calling tools.")

    assert agent.reply("Retry my request") == "Let's try that again."
    assert client.responses.requests[-1]["previous_response_id"] == "response-1"
