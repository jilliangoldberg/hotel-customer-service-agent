"""Composition-root tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sierra_agent import factory
from sierra_agent.config import PROJECT_ROOT, Settings


class FakeResponses:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        return SimpleNamespace(
            id="response-1",
            output=[],
            output_text="Hello from Sierra.",
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_build_agent_composes_configured_model_and_registered_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr(factory, "OpenAI", lambda **_kwargs: client)
    settings = Settings(
        openai_api_key="test-key",
        openai_model="agent-model",
        openai_eval_model="judge-model",
        promotion_secret="a-valid-promotion-secret",
        data_dir=Path(PROJECT_ROOT / "data"),
    )

    agent = factory.build_agent(settings)

    assert agent.reply("Hello") == "Hello from Sierra."
    request = client.responses.requests[0]
    assert request["model"] == "agent-model"
    assert {tool["name"] for tool in request["tools"]} == {
        "lookup_reservation",
        "get_available_rooms",
    }
