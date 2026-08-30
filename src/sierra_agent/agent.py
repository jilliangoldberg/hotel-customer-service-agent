"""The minimal OpenAI Responses API and tool-calling loop."""

import json
from time import perf_counter
from typing import Any

from openai import OpenAI

from sierra_agent.prompt import SYSTEM_PROMPT
from sierra_agent.tools import HotelTools, TOOL_DEFINITIONS


class AgentLoopError(RuntimeError):
    """Raised when the model cannot produce a safe final response."""


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
                    self._previous_response_id = response.id
                    trace["outcome"] = "completed"
                    return text

                if tool_rounds >= self._max_tool_rounds:
                    raise AgentLoopError("The model exceeded the tool-call limit.")
                tool_rounds += 1
                trace["tool_rounds"] = tool_rounds

                tool_outputs = []
                for call in tool_calls:
                    result = self._tools.execute(call.name, call.arguments)
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
            trace["outcome"] = "failed"
            trace["error"] = type(error).__name__
            raise
        finally:
            trace["duration_ms"] = round((perf_counter() - started_at) * 1000)
            self._last_trace = trace

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
    ) -> Any:
        request: dict[str, Any] = {
            "model": self._model,
            "instructions": SYSTEM_PROMPT,
            "input": input_items,
            "tools": TOOL_DEFINITIONS,
            "parallel_tool_calls": True,
        }
        if previous_response_id is not None:
            request["previous_response_id"] = previous_response_id
        return self._client.responses.create(**request)
