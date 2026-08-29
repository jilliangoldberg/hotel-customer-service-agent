"""The minimal OpenAI Responses API and tool-calling loop."""

import json
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

    def reply(self, user_message: str) -> str:
        """Send one customer message and resolve any requested tool calls."""

        previous_response_id = self._previous_response_id
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
                    return text

                if tool_rounds >= self._max_tool_rounds:
                    raise AgentLoopError("The model exceeded the tool-call limit.")
                tool_rounds += 1

                tool_outputs = []
                for call in tool_calls:
                    result = self._tools.execute(call.name, call.arguments)
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
        except Exception:
            self._previous_response_id = previous_response_id
            raise

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
