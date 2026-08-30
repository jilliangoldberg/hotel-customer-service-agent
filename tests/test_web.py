"""Web-session contract tests."""

from collections.abc import Callable

import pytest

from sierra_agent import web


class FakeAgent:
    def __init__(self, label: int) -> None:
        self._label = label

    def reply(self, message: str) -> str:
        return f"agent {self._label}: {message}"

    @property
    def last_trace(self) -> dict[str, object]:
        return {
            "outcome": "completed",
            "tool_rounds": 0,
            "tools": [],
            "duration_ms": 1,
        }


@pytest.fixture
def client_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], object]:
    next_label = 0

    def build_fake_agent() -> FakeAgent:
        nonlocal next_label
        next_label += 1
        return FakeAgent(next_label)

    monkeypatch.setattr(web, "build_agent", build_fake_agent)
    web.app.config.update(TESTING=True, SECRET_KEY="test-session-secret")
    with web._sessions_lock:
        web._sessions.clear()
    yield web.app.test_client
    with web._sessions_lock:
        web._sessions.clear()


def test_each_browser_keeps_its_own_conversation(
    client_factory: Callable[[], object],
) -> None:
    first_browser = client_factory()
    second_browser = client_factory()

    first_reply = first_browser.post("/chat", json={"message": "first"}).get_json()
    second_reply = second_browser.post("/chat", json={"message": "second"}).get_json()

    assert first_reply["reply"] == "agent 1: first"
    assert first_reply["trace"]["outcome"] == "completed"
    assert second_reply["reply"] == "agent 2: second"

    assert first_browser.post("/reset").get_json() == {"ok": True}
    assert first_browser.post("/chat", json={"message": "again"}).get_json()[
        "reply"
    ] == "agent 3: again"
    assert second_browser.post("/chat", json={"message": "still here"}).get_json()[
        "reply"
    ] == "agent 2: still here"
