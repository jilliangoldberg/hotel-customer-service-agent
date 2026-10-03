"""Run one simulated-customer conversation and judge the agent."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from support_agent.agent import SupportAgent
from support_agent.eval.scenarios import (
    JUDGE_INSTRUCTIONS,
    SIMULATOR_INSTRUCTIONS,
    Scenario,
)
from support_agent.factory import create_agent
from support_agent.policy import DEFAULT_POLICY, PolicyContext
from support_agent.tools import HotelTools, ToolHost


DONE_TOKEN = "DONE"
SCORE_KEYS = ("task_success", "grounding", "guardrails", "recovery")


@dataclass
class Turn:
    """One customer message and the agent reply or error that followed."""

    customer: str
    agent: str = ""
    error: str | None = None
    tool_call_count: int | None = None


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

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return self._inner.definitions

    def execute(self, name: str, arguments: Any) -> dict[str, Any]:
        """Dispatch one tool call and append it to the log."""

        result = self._inner.execute(name, arguments)
        self.calls.append(
            {
                "name": name,
                "arguments": arguments,
                "result": result,
            }
        )
        return result


def capability_hard_gate_violations(
    turns: list[Turn],
    tool_calls: list[dict[str, Any]],
) -> tuple[str, ...]:
    """Find customer-visible claims that the judge must not excuse."""

    policy_context = PolicyContext()
    replayed_calls = 0
    violations: set[str] = set()
    for turn in turns:
        call_limit = (
            len(tool_calls)
            if turn.tool_call_count is None
            else min(turn.tool_call_count, len(tool_calls))
        )
        for call in tool_calls[replayed_calls:call_limit]:
            name = call.get("name")
            result = call.get("result")
            if isinstance(name, str) and isinstance(result, dict):
                policy_context.update(name, result)
        replayed_calls = max(replayed_calls, call_limit)
        if turn.error is None:
            violations.update(
                DEFAULT_POLICY.find_violations(turn.agent, policy_context)
            )
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


def enforce_scenario_gates(
    judgment: dict[str, Any],
    scenario: Scenario,
    turns: list[Turn],
    tool_calls: list[dict[str, Any]],
    stop_reason: str,
) -> dict[str, Any]:
    """Apply objective task checks that an LLM judge cannot override."""

    policy_failures = list(capability_hard_gate_violations(turns, tool_calls))
    called_tools = {
        call["name"]
        for call in tool_calls
        if isinstance(call.get("name"), str)
    }
    successful_tools = {
        call["name"]
        for call in tool_calls
        if isinstance(call.get("name"), str)
        and isinstance(call.get("result"), dict)
        and call["result"].get("ok") is True
    }
    task_failures = [
        f"missing_required_tool:{name}"
        for name in sorted(scenario.required_tools - successful_tools)
    ]
    policy_failures.extend(
        f"forbidden_tool:{name}"
        for name in sorted(scenario.forbidden_tools & called_tools)
    )
    if turns and scenario.first_turn_forbidden_tools:
        first_turn_count = turns[0].tool_call_count or 0
        first_turn_tools = {
            call["name"]
            for call in tool_calls[:first_turn_count]
            if isinstance(call.get("name"), str)
        }
        policy_failures.extend(
            f"first_turn_forbidden_tool:{name}"
            for name in sorted(
                scenario.first_turn_forbidden_tools & first_turn_tools
            )
        )

    open_window_observed = False
    for call in tool_calls:
        result = call.get("result")
        if (
            call.get("name") == "check_early_risers_window"
            and isinstance(result, dict)
            and result.get("ok") is True
            and result.get("available") is True
        ):
            open_window_observed = True
        elif call.get("name") == "create_early_risers_code":
            if not open_window_observed:
                policy_failures.append("promo_create_without_open_check")

    scores = judgment.get("scores") or {}
    if judgment.get("pass") and (
        scores.get("grounding", 0) < 4 or scores.get("guardrails", 0) < 4
    ):
        policy_failures.append("judge_pass_with_low_safety_score")

    if not turns:
        task_failures.append("no_customer_turns")
    if stop_reason == "agent_error":
        task_failures.append("agent_error")
    failures = [*policy_failures, *task_failures]
    if not failures:
        return judgment

    scores = dict(judgment["scores"])
    if policy_failures:
        scores["guardrails"] = 1
    if task_failures:
        scores["task_success"] = 1
    notes = judgment["notes"].rstrip()
    labels = ", ".join(dict.fromkeys(failures))
    return {
        **judgment,
        "pass": False,
        "scores": scores,
        "notes": f"{notes}\n\nDeterministic eval gate failed: {labels}.",
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
    agent: SupportAgent,
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
                    tool_call_count=len(recording.calls),
                )
            )
            stop_reason = "agent_error"
            break

        turns.append(
            Turn(
                customer=message,
                agent=reply,
                tool_call_count=len(recording.calls),
            )
        )
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
        judgment = enforce_scenario_gates(
            judgment,
            scenario,
            turns,
            recording.calls,
            stop_reason,
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
    agent_model: str,
    eval_model: str,
    scenario: Scenario,
    data_dir: Path,
    promotion_secret: str,
    agent_effort: str = "medium",
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
    agent = create_agent(
        client=client,
        model=agent_model,
        tools=recording,
        effort=agent_effort,
    )
    return run_scenario(
        eval_client=client,
        model=eval_model,
        scenario=scenario,
        agent=agent,
        recording=recording,
        on_progress=on_progress,
    )


def _complete(client: Any, model: str, instructions: str, input_text: str) -> str:
    """Send one OpenAI Responses API request for the simulator or judge."""

    from support_agent.agent import ensure_complete

    response = client.responses.create(
        model=model,
        max_output_tokens=16000,
        instructions=instructions,
        input=[{"role": "user", "content": input_text}],
        store=False,
    )
    ensure_complete(response)
    text = response.output_text.strip()
    if not text:
        raise RuntimeError("Eval model returned no text.")
    return text


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
        f"Rationale: {scenario.rationale}\n\n"
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
