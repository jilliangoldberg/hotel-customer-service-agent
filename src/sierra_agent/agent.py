"""The minimal OpenAI Responses API and tool-calling loop."""

import json
import re
from time import perf_counter
from typing import Any

from openai import OpenAI

from sierra_agent.prompt import CAPABILITY_REWRITE_INSTRUCTIONS, SYSTEM_PROMPT
from sierra_agent.tools import HotelTools, TOOL_DEFINITIONS


class AgentLoopError(RuntimeError):
    """Raised when the model cannot produce a safe final response."""


SAFE_CAPABILITY_FALLBACK = (
    "I can’t help with that here. I can check a reservation’s status, suggest "
    "available Sierra rooms, or help with the Early Risers Promotion."
)

_FUTURE_UNSUPPORTED_ACTION = re.compile(
    r"\b(?:i(?:['’]ll|\s+(?:will|can|am going to))|"
    r"we(?:['’]ll|\s+(?:will|can|are going to)))\s+"
    r"(?:escalate|contact|reach out to|follow up(?:\s+with)?|"
    r"keep (?:you )?updated|notify|process|cancel|refund|send)\b",
    re.IGNORECASE,
)
_PAST_UNSUPPORTED_ACTION = re.compile(
    r"\b(?:i(?:['’]ve|\s+(?:have|am))|"
    r"we(?:['’]ve|\s+(?:have|are)))\s+"
    r"(?:escalated|contacted|reached out|followed up|notified|processed|"
    r"cancelled|refunded|sent|noted)\b",
    re.IGNORECASE,
)
_RECOMMENDATION = re.compile(
    r"\b(?:recommend|suggest|consider|look at|check out)\b",
    re.IGNORECASE,
)


def find_capability_violations(
    text: str,
    room_names: set[str] | None = None,
) -> tuple[str, ...]:
    """Return policy labels for unsupported customer-facing claims.

    Room recommendations must name an item returned by the catalog tool.
    This intentionally applies only to explicit recommendation language, so
    ordinary reservation-status and promotion responses are not constrained by it.
    """

    violations: list[str] = []
    if _FUTURE_UNSUPPORTED_ACTION.search(text):
        violations.append("unsupported_future_action")
    if _PAST_UNSUPPORTED_ACTION.search(text):
        violations.append("unsupported_claimed_action")

    allowed_names = {
        name.casefold() for name in (room_names or set()) if name.strip()
    }
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if not _RECOMMENDATION.search(sentence):
            continue
        if not any(name in sentence.casefold() for name in allowed_names):
            violations.append("off_catalog_recommendation")
            break
    return tuple(violations)


class SierraAgent:
    """Manage one in-memory customer conversation."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools: HotelTools,
        max_tool_rounds: int = 5,
    ) -> None:
        self._client = client
        self._model = model
        self._tools = tools
        self._max_tool_rounds = max_tool_rounds
        self._previous_response_id: str | None = None
        self._known_room_names: set[str] = set()
        self._last_trace: dict[str, Any] = {}

    @property
    def last_trace(self) -> dict[str, Any]:
        """Return the latest redacted execution trace."""

        return {
            **self._last_trace,
            "tools": list(self._last_trace.get("tools", [])),
        }

    def reply(self, user_message: str) -> str:
        """Send one customer message and resolve any requested tool calls."""

        previous_response_id = self._previous_response_id
        previous_room_names = set(self._known_room_names)
        known_room_names = set(previous_room_names)
        started_at = perf_counter()
        trace: dict[str, Any] = {"tools": [], "tool_rounds": 0}
        try:
            response = self._create_response(
                input_items=user_message,
                previous_response_id=previous_response_id,
            )
            tool_rounds = 0

            while True:
                tool_calls = [
                    item
                    for item in response.output
                    if item.type == "function_call"
                ]
                if not tool_calls:
                    text = response.output_text.strip()
                    if not text:
                        raise AgentLoopError("The model returned no text response.")
                    text, response_id, policy_outcome = self._safe_final_response(
                        response,
                        text,
                        known_room_names,
                    )
                    self._previous_response_id = response_id
                    self._known_room_names = known_room_names
                    trace["outcome"] = "completed"
                    trace["response_policy"] = policy_outcome
                    return text

                if tool_rounds >= self._max_tool_rounds:
                    raise AgentLoopError("The model exceeded the tool-call limit.")
                tool_rounds += 1
                trace["tool_rounds"] = tool_rounds

                tool_outputs = []
                for call in tool_calls:
                    result = self._tools.execute(call.name, call.arguments)
                    if call.name == "get_available_rooms":
                        known_room_names.update(
                            room["name"]
                            for room in result.get("rooms", [])
                            if isinstance(room, dict)
                            and isinstance(room.get("name"), str)
                        )
                    trace["tools"].append(
                        {
                            "name": call.name,
                            "outcome": self._tool_outcome(result),
                        }
                    )
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(result, separators=(",", ":")),
                        }
                    )

                response = self._create_response(
                    input_items=tool_outputs,
                    previous_response_id=response.id,
                )
        except Exception as error:
            self._previous_response_id = previous_response_id
            self._known_room_names = previous_room_names
            trace["outcome"] = "failed"
            trace["error"] = type(error).__name__
            raise
        finally:
            trace["duration_ms"] = round((perf_counter() - started_at) * 1000)
            self._last_trace = trace

    def _safe_final_response(
        self,
        response: Any,
        text: str,
        room_names: set[str],
    ) -> tuple[str, str, str]:
        """Return a compliant final response, rewriting once when needed."""

        if not find_capability_violations(text, room_names):
            return text, response.id, "passed"

        corrected = self._create_response(
            input_items=(
                "Replace your immediately preceding customer-facing reply using "
                "the correction instructions."
            ),
            previous_response_id=response.id,
            instructions=f"{SYSTEM_PROMPT}\n\n{CAPABILITY_REWRITE_INSTRUCTIONS}",
            tool_choice="none",
        )
        corrected_text = corrected.output_text.strip()
        corrected_tool_calls = [
            item for item in corrected.output if item.type == "function_call"
        ]
        if (
            corrected_text
            and not corrected_tool_calls
            and not find_capability_violations(corrected_text, room_names)
        ):
            return corrected_text, corrected.id, "rewritten"
        return SAFE_CAPABILITY_FALLBACK, corrected.id, "fallback"

    @staticmethod
    def _tool_outcome(result: dict[str, Any]) -> str:
        """Summarize a tool result without retaining its arguments or data."""

        if not result.get("ok"):
            return "rejected"
        if "found" in result:
            return "found" if result["found"] else "not_found"
        if "eligible" in result:
            return "eligible" if result["eligible"] else "not_eligible"
        return "completed"

    def _create_response(
        self,
        input_items: str | list[dict[str, Any]],
        previous_response_id: str | None,
        instructions: str = SYSTEM_PROMPT,
        tool_choice: str | None = None,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self._model,
            "instructions": instructions,
            "input": input_items,
            "tools": TOOL_DEFINITIONS,
            "parallel_tool_calls": True,
        }
        if previous_response_id is not None:
            request["previous_response_id"] = previous_response_id
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        return self._client.responses.create(**request)
