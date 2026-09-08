"""Web-session contract tests."""

from collections.abc import Callable
from pathlib import Path

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


def test_eval_scenario_list_and_missing_result(
    client_factory: Callable[[], object],
) -> None:
    browser = client_factory()
    listing = browser.get("/eval/scenarios").get_json()
    ids = [item["id"] for item in listing["scenarios"]]
    assert "jailbreak" in ids
    assert "happy-reservation" in ids
    missing = browser.get("/eval/results/nope")
    assert missing.status_code == 404


def test_eval_result_from_saved_file(
    client_factory: Callable[[], object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web, "EVAL_RESULTS_DIR", tmp_path)
    tmp_path.joinpath("happy-reservation.json").write_text(
        '{"scenario_id":"happy-reservation","title":"Straightforward reservation lookup",'
        '"interview_note":"Keep it simple.","stop_reason":"done","pass":true,'
        '"scores":{"task_success":5,"grounding":5,"guardrails":5,"recovery":4},'
        '"notes":"ok","turns":[{"customer":"Hi","agent":"Hello"}],"tools":[]}',
        encoding="utf-8",
    )
    browser = client_factory()
    data = browser.get("/eval/results/happy-reservation").get_json()
    assert data["pass"] is True
    assert data["turns"][0]["customer"] == "Hi"
