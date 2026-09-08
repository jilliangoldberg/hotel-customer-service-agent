"""Run one simulated-customer conversation and judge the agent."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Protocol

from sierra_agent.agent import SierraAgent, find_capability_violations
from sierra_agent.eval.scenarios import (
    JUDGE_INSTRUCTIONS,
    SIMULATOR_INSTRUCTIONS,
    Scenario,
)
from sierra_agent.tools import HotelTools


DONE_TOKEN = "DONE"
SCORE_KEYS = ("task_success", "grounding", "guardrails", "recovery")


class ToolHost(Protocol):
    """Anything the agent can call through execute(), including test doubles."""

    def execute(self, name: str, arguments: str) -> dict[str, Any]: ...


@dataclass
class Turn:
    """One customer message and the agent reply or error that followed."""

    customer: str
    agent: str = ""
    error: str | None = None


@dataclass
class EvalRun:
    """Transcript, tool log, and judgment for one scenario."""

    scenario: Scenario
    turns: list[Turn]
    tool_calls: list[dict[str, Any]]
    stop_reason: str
    judgment: dict[str, Any]


class RecordingTools:
    """Wrap HotelTools.execute and keep a full log for the judge.

    Production traces stay redacted. Eval records names, arguments, and
    results locally so grounding can be scored.
    """

    def __init__(self, inner: ToolHost) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def execute(self, name: str, arguments: str) -> dict[str, Any]:
        """Dispatch one tool call and append it to the log."""

        result = self._inner.execute(name, arguments)
        self.calls.append(
            {
                "name": name,
                "arguments": _parse_arguments(arguments),
                "result": result,
            }
        )
        return result


def capability_hard_gate_violations(
    turns: list[Turn],
    tool_calls: list[dict[str, Any]],
) -> tuple[str, ...]:
    """Find customer-visible claims that the judge must not excuse."""

    room_names: set[str] = set()
    for call in tool_calls:
        if call.get("name") != "get_available_rooms":
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        rooms = result.get("rooms")
        if not isinstance(rooms, list):
            continue
        room_names.update(
            room["name"]
            for room in rooms
            if isinstance(room, dict) and isinstance(room.get("name"), str)
        )

    violations: set[str] = set()
    for turn in turns:
        if turn.error is None:
            violations.update(find_capability_violations(turn.agent, room_names))
    return tuple(sorted(violations))


def enforce_capability_hard_gates(
    judgment: dict[str, Any],
    turns: list[Turn],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Override an LLM judge that overlooks an unsupported commitment."""

    violations = capability_hard_gate_violations(turns, tool_calls)
    if not violations:
        return judgment

    scores = dict(judgment["scores"])
    scores["guardrails"] = 1
    notes = judgment["notes"].rstrip()
    labels = ", ".join(violations)
    return {
        **judgment,
        "pass": False,
        "scores": scores,
        "notes": (
            f"{notes}\n\nDeterministic capability hard gate failed: {labels}."
        ),
    }


def is_done(message: str) -> bool:
    """Return True when the simulated customer has ended the chat."""

    return message.strip() == DONE_TOKEN


def parse_judge_json(text: str) -> dict[str, Any]:
    """Parse a judge reply, including optional markdown fences."""

    payload = _extract_json_object(text)
    if not isinstance(payload.get("pass"), bool):
        raise ValueError("Judge JSON must include boolean pass.")
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Judge JSON must include scores.")
    parsed_scores: dict[str, int] = {}
    for key in SCORE_KEYS:
        value = scores.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            raise ValueError(f"scores.{key} must be an integer from 1 to 5.")
        parsed_scores[key] = value
    notes = payload.get("notes")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("Judge JSON must include notes.")
    return {
        "pass": payload["pass"],
        "scores": parsed_scores,
        "notes": notes.strip(),
    }


def run_scenario(
    *,
    eval_client: Any,
    model: str,
    scenario: Scenario,
    agent: SierraAgent,
    recording: RecordingTools,
    on_progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> EvalRun:
    """Play the simulated customer against the agent, then score the transcript."""

    def notify(event: str, **info: Any) -> None:
        if on_progress is not None:
            on_progress(event, info)

    turns: list[Turn] = []
    stop_reason = "max_turns"

    for _ in range(scenario.max_turns):
        turn_number = len(turns) + 1
        notify("customer", turn=turn_number, max_turns=scenario.max_turns)
        message = _complete(
            eval_client,
            model,
            SIMULATOR_INSTRUCTIONS,
            _simulator_input(scenario, turns),
        )
        if is_done(message):
            stop_reason = "done"
            break

        notify("agent", turn=turn_number, max_turns=scenario.max_turns)
        try:
            reply = agent.reply(message)
        except Exception as error:
            turns.append(
                Turn(
                    customer=message,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            stop_reason = "agent_error"
            break

        turns.append(Turn(customer=message, agent=reply))
    else:
        stop_reason = "max_turns"

    notify("judge")
    try:
        raw_judgment = _complete(
            eval_client,
            model,
            JUDGE_INSTRUCTIONS,
            _judge_input(scenario, turns, recording.calls, stop_reason),
        )
        judgment = parse_judge_json(raw_judgment)
        judgment = enforce_capability_hard_gates(
            judgment,
            turns,
            recording.calls,
        )
    except Exception as error:
        judgment = {
            "pass": False,
            "scores": {key: 1 for key in SCORE_KEYS},
            "notes": f"Judge output could not be parsed: {error}",
        }

    return EvalRun(
        scenario=scenario,
        turns=turns,
        tool_calls=list(recording.calls),
        stop_reason=stop_reason,
        judgment=judgment,
    )


def execute_scenario(
    client: Any,
    model: str,
    scenario: Scenario,
    data_dir: Path,
    promotion_secret: str,
    on_progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> EvalRun:
    """Build tools and an agent, then run one simulated-customer scenario."""

    clock: Callable[[], datetime] | None = None
    if scenario.clock is not None:
        current = scenario.clock
        clock = lambda: current
    recording = RecordingTools(
        HotelTools(data_dir, promotion_secret, now=clock)
    )
    agent = SierraAgent(
        client=client,
        model=model,
        tools=recording,  # type: ignore[arg-type]
    )
    return run_scenario(
        eval_client=client,
        model=model,
        scenario=scenario,
        agent=agent,
        recording=recording,
        on_progress=on_progress,
    )


def _complete(client: Any, model: str, instructions: str, input_text: str) -> str:
    """Send one Responses API request with no tools."""

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
    )
    text = (response.output_text or "").strip()
    if not text:
        raise RuntimeError("Eval model returned no text.")
    return text


def _parse_arguments(arguments: str) -> Any:
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def _facts_block(facts: dict[str, str]) -> str:
    if not facts:
        return "(none)"
    return "\n".join(f"- {key}: {value}" for key, value in facts.items())


def _transcript_block(turns: list[Turn]) -> str:
    if not turns:
        return "(no messages yet; you speak first)"
    lines: list[str] = []
    for turn in turns:
        lines.append(f"Customer: {turn.customer}")
        if turn.error is not None:
            lines.append(f"Agent error: {turn.error}")
        else:
            lines.append(f"Agent: {turn.agent}")
    return "\n".join(lines)


def _simulator_input(scenario: Scenario, turns: list[Turn]) -> str:
    return (
        f"Your goal:\n{scenario.simulator_goal}\n\n"
        f"How you talk:\n{scenario.simulator_behavior}\n\n"
        f"Facts you know:\n{_facts_block(scenario.facts)}\n\n"
        f"Conversation so far:\n{_transcript_block(turns)}"
    )


def _judge_input(
    scenario: Scenario,
    turns: list[Turn],
    tool_calls: list[dict[str, Any]],
    stop_reason: str,
) -> str:
    return (
        f"Scenario: {scenario.id} — {scenario.title}\n"
        f"Interview note: {scenario.interview_note}\n\n"
        f"Success criteria (for the AGENT):\n{scenario.judge_criteria}\n\n"
        f"Stop reason: {stop_reason}\n\n"
        f"Transcript:\n{_transcript_block(turns)}\n\n"
        f"Tool log:\n{json.dumps(tool_calls, indent=2)}"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        fence_at = stripped.rfind("```")
        if fence_at != -1:
            stripped = stripped[:fence_at]
        stripped = stripped.strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Judge response did not contain JSON.")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Judge JSON must be an object.")
    return payload
