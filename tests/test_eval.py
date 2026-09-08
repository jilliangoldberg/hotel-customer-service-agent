"""Tests for the adversarial eval harness using a fake OpenAI client."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sierra_agent.agent import SierraAgent
from sierra_agent.eval.harness import (
    EvalRun,
    RecordingTools,
    Turn,
    enforce_capability_hard_gates,
    parse_judge_json,
    run_scenario,
)
from sierra_agent.eval.__main__ import main
from sierra_agent.eval.reports import format_report, parse_report, write_reports
from sierra_agent.eval.scenarios import Scenario, select_scenarios


JUDGMENT = {
    "pass": True,
    "scores": {
        "task_success": 4,
        "grounding": 5,
        "guardrails": 5,
        "recovery": 4,
    },
    "notes": "The agent stayed on task.",
}


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


class FakeInnerTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def execute(self, name: str, arguments: str) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"ok": True, "found": True, "tool": name}


def text_response(text: str, response_id: str = "response") -> Any:
    return SimpleNamespace(id=response_id, output=[], output_text=text)


def tool_call(name: str, call_id: str, arguments: str = "{}") -> Any:
    return SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


def tool_response(response_id: str, *calls: Any) -> Any:
    return SimpleNamespace(id=response_id, output=list(calls), output_text="")


def tiny_scenario(**overrides: Any) -> Scenario:
    values: dict[str, Any] = {
        "id": "tiny",
        "title": "Tiny",
        "interview_note": "Keep it simple.",
        "simulator_goal": "Ask one question.",
        "simulator_behavior": "Be brief.",
        "judge_criteria": "Answer the question.",
        "max_turns": 8,
    }
    values.update(overrides)
    return Scenario(**values)


def make_agent(responses: list[Any]) -> tuple[SierraAgent, RecordingTools]:
    recording = RecordingTools(FakeInnerTools())
    agent = SierraAgent(
        client=FakeClient(responses),  # type: ignore[arg-type]
        model="gpt-4o-mini",
        tools=recording,  # type: ignore[arg-type]
    )
    return agent, recording


def test_done_stops_the_loop_without_calling_the_agent() -> None:
    eval_client = FakeClient(
        [text_response("DONE"), text_response(json.dumps(JUDGMENT))]
    )
    agent, recording = make_agent([])

    run = run_scenario(
        eval_client=eval_client,
        model="gpt-4o-mini",
        scenario=tiny_scenario(),
        agent=agent,
        recording=recording,
    )

    assert run.stop_reason == "done"
    assert run.turns == []
    assert run.judgment["pass"] is True
    assert len(eval_client.responses.requests) == 2


def test_run_scenario_emits_progress_events() -> None:
    events: list[str] = []
    eval_client = FakeClient(
        [
            text_response("Where is my pack?"),
            text_response("DONE"),
            text_response(json.dumps(JUDGMENT)),
        ]
    )
    agent, recording = make_agent([text_response("Hello", "agent-1")])

    run_scenario(
        eval_client=eval_client,
        model="gpt-4o-mini",
        scenario=tiny_scenario(),
        agent=agent,
        recording=recording,
        on_progress=lambda event, _info: events.append(event),
    )

    assert events == ["customer", "agent", "customer", "judge"]


def test_progress_bar_fills_with_completed_count() -> None:
    from sierra_agent.eval.__main__ import _progress_bar

    empty = _progress_bar(0, 6)
    full = _progress_bar(6, 6)
    assert empty.startswith("[")
    assert empty.endswith("]")
    assert "#" not in empty
    assert "-" not in full[1:-1]
    assert _progress_bar(3, 6).count("#") == 9


def test_max_turns_stops_the_loop() -> None:
    eval_client = FakeClient(
        [
            text_response("Where is my pack?"),
            text_response("Still waiting."),
            text_response(json.dumps(JUDGMENT)),
        ]
    )
    agent, recording = make_agent(
        [
            text_response("Could I have your email?", "agent-1"),
            text_response("I am still here.", "agent-2"),
        ]
    )

    run = run_scenario(
        eval_client=eval_client,
        model="gpt-4o-mini",
        scenario=tiny_scenario(max_turns=2),
        agent=agent,
        recording=recording,
    )

    assert run.stop_reason == "max_turns"
    assert [turn.customer for turn in run.turns] == [
        "Where is my pack?",
        "Still waiting.",
    ]
    assert len(eval_client.responses.requests) == 3


def test_recording_captures_execute_during_a_run() -> None:
    arguments = '{"email":"hiker@example.com","reservation_number":"#H001"}'
    eval_client = FakeClient(
        [
            text_response("Where is reservation #H001?"),
            text_response("DONE"),
            text_response(json.dumps(JUDGMENT)),
        ]
    )
    agent, recording = make_agent(
        [
            tool_response(
                "tool-1",
                tool_call("lookup_reservation", "call-1", arguments),
            ),
            text_response("Your reservation is confirmed.", "agent-2"),
        ]
    )

    run = run_scenario(
        eval_client=eval_client,
        model="gpt-4o-mini",
        scenario=tiny_scenario(),
        agent=agent,
        recording=recording,
    )

    assert run.tool_calls == [
        {
            "name": "lookup_reservation",
            "arguments": {
                "email": "hiker@example.com",
                "reservation_number": "#H001",
            },
            "result": {"ok": True, "found": True, "tool": "lookup_reservation"},
        }
    ]
    assert recording.calls == run.tool_calls


def test_parse_judge_json_accepts_fenced_block() -> None:
    fenced = f"```json\n{json.dumps(JUDGMENT, indent=2)}\n```"

    assert parse_judge_json(fenced) == JUDGMENT


def test_parse_judge_json_rejects_invalid_scores() -> None:
    payload = {
        **JUDGMENT,
        "scores": {**JUDGMENT["scores"], "grounding": 9},
    }

    with pytest.raises(ValueError, match="grounding"):
        parse_judge_json(json.dumps(payload))


def test_run_uses_fenced_judge_json() -> None:
    fenced = f"```json\n{json.dumps(JUDGMENT)}\n```"
    eval_client = FakeClient([text_response("DONE"), text_response(fenced)])
    agent, recording = make_agent([])

    run = run_scenario(
        eval_client=eval_client,
        model="gpt-4o-mini",
        scenario=tiny_scenario(),
        agent=agent,
        recording=recording,
    )

    assert run.judgment == JUDGMENT


def test_unparsed_judge_output_fails_the_run() -> None:
    eval_client = FakeClient(
        [text_response("DONE"), text_response("I liked this chat.")]
    )
    agent, recording = make_agent([])

    run = run_scenario(
        eval_client=eval_client,
        model="gpt-4o-mini",
        scenario=tiny_scenario(),
        agent=agent,
        recording=recording,
    )

    assert run.judgment["pass"] is False
    assert "could not be parsed" in run.judgment["notes"]


def test_capability_hard_gate_overrides_a_passing_judge() -> None:
    result = enforce_capability_hard_gates(
        JUDGMENT,
        [Turn(customer="Refund me.", agent="I'll escalate this to a manager.")],
        [],
    )

    assert result["pass"] is False
    assert result["scores"]["guardrails"] == 1
    assert "unsupported_future_action" in result["notes"]


def test_capability_hard_gate_rejects_off_catalog_recommendation() -> None:
    result = enforce_capability_hard_gates(
        JUDGMENT,
        [Turn(customer="What should I buy?", agent="I recommend Marriott.")],
        [
            {
                "name": "get_available_rooms",
                "arguments": {},
                "result": {
                    "ok": True,
                    "rooms": [{"name": "Garden King Room"}],
                },
            }
        ],
    )

    assert result["pass"] is False
    assert "off_catalog_recommendation" in result["notes"]


def test_agent_error_still_reaches_the_judge() -> None:
    eval_client = FakeClient(
        [text_response("Hello"), text_response(json.dumps(JUDGMENT))]
    )
    agent, recording = make_agent([text_response("")])

    run = run_scenario(
        eval_client=eval_client,
        model="gpt-4o-mini",
        scenario=tiny_scenario(),
        agent=agent,
        recording=recording,
    )

    assert run.stop_reason == "agent_error"
    assert run.turns[0].error is not None
    assert "AgentLoopError" in run.turns[0].error
    assert run.judgment["pass"] is True


def test_select_scenarios_filters_by_id() -> None:
    selected = select_scenarios("derail-reservation")

    assert [scenario.id for scenario in selected] == ["derail-reservation"]
    assert [scenario.id for scenario in select_scenarios(None)] == [
        "happy-reservation",
        "derail-reservation",
        "confused-ids",
        "jailbreak",
        "out-of-scope",
        "promo-then-rec",
    ]


def test_select_scenarios_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        select_scenarios("nope")


def test_cli_list_prints_scenario_ids(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "happy-reservation" in output
    assert "derail-reservation" in output
    assert "promo-then-rec" in output


def test_cli_unknown_scenario_exits_without_setup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--scenario", "nope"]) == 1
    error = capsys.readouterr().err
    assert "Unknown scenario" in error


def test_markdown_report_is_readable() -> None:
    run = EvalRun(
        scenario=tiny_scenario(id="tiny", title="Tiny"),
        turns=[
            Turn(customer="Where is my pack?", agent="Could I have your email?"),
        ],
        tool_calls=[
            {
                "name": "lookup_reservation",
                "arguments": {},
                "result": {"ok": True, "found": True},
            }
        ],
        stop_reason="done",
        judgment=JUDGMENT,
    )

    report = format_report(run)

    assert report.startswith("# tiny: Tiny")
    assert "**You:** Where is my pack?" in report
    assert "**Agent:** Could I have your email?" in report
    assert "- `lookup_reservation` → found" in report
    assert "**Result:** PASS" in report
    assert "## Judge" in report


def test_write_reports_creates_markdown_files(tmp_path: Path) -> None:
    run = EvalRun(
        scenario=tiny_scenario(id="tiny", title="Tiny"),
        turns=[Turn(customer="Hello", agent="Hi there.")],
        tool_calls=[],
        stop_reason="done",
        judgment=JUDGMENT,
    )

    written = write_reports([run], tmp_path)

    transcript = tmp_path / "tiny.md"
    saved = tmp_path / "tiny.json"
    summary = tmp_path / "summary.md"
    assert written == [transcript, saved, summary]
    assert "Hello" in transcript.read_text(encoding="utf-8")
    assert "[tiny](tiny.md)" in summary.read_text(encoding="utf-8")
    assert "PASS" in summary.read_text(encoding="utf-8")


def test_parse_report_roundtrip() -> None:
    run = EvalRun(
        scenario=tiny_scenario(id="tiny", title="Tiny"),
        turns=[Turn(customer="Hello", agent="Hi there.")],
        tool_calls=[
            {
                "name": "lookup_reservation",
                "arguments": {},
                "result": {"ok": True, "found": True},
            }
        ],
        stop_reason="done",
        judgment=JUDGMENT,
    )

    parsed = parse_report(format_report(run))

    assert parsed["scenario_id"] == "tiny"
    assert parsed["pass"] is True
    assert parsed["turns"][0]["customer"] == "Hello"
    assert parsed["turns"][0]["agent"] == "Hi there."
    assert parsed["tools"] == [{"name": "lookup_reservation", "outcome": "found"}]
    assert parsed["scores"]["grounding"] == 5
