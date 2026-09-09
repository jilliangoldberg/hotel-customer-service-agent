"""The minimal OpenAI Responses API and tool-calling loop."""

import json
import re
from time import perf_counter
from typing import Any

from openai import OpenAI

from sierra_agent.policy import CapabilityPolicy, DEFAULT_POLICY, PolicyContext
from sierra_agent.prompt import system_prompt
from sierra_agent.tools import ToolHost, exposed_tool_definitions


_EARLY_RISERS_PATTERN = re.compile(r"\bearly[\s-]+risers?\b", re.IGNORECASE)


class AgentLoopError(RuntimeError):
    """Raised when the model cannot produce a safe final response."""


class SierraAgent:
    """Manage one in-memory customer conversation."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools: ToolHost,
        policy: CapabilityPolicy = DEFAULT_POLICY,
        # Healthy turns need 1–2 rounds; 5 is for retries
        max_tool_rounds: int = 5,
    ) -> None:
        self._client = client
        self._model = model
        self._tools = tools
        self._policy = policy
        self._max_tool_rounds = max_tool_rounds
        self._previous_response_id: str | None = None
        self._policy_context = PolicyContext()
        self._early_risers_mentioned = False
        self._last_trace: dict[str, Any] = {}

    @property
    def last_trace(self) -> dict[str, Any]:
        """Return the latest redacted execution trace."""

        return {
            **self._last_trace,
            "tools": list(self._last_trace.get("tools", [])),
            "exposed_tools": list(self._last_trace.get("exposed_tools", [])),
        }

    def reply(self, user_message: str) -> str:
        """Send one customer message and resolve any requested tool calls."""

        if _EARLY_RISERS_PATTERN.search(user_message):
            self._early_risers_mentioned = True

        previous_response_id = self._previous_response_id
        previous_policy_context = self._policy_context.copy()
        policy_context = previous_policy_context.copy()
        started_at = perf_counter()
        exposed_tools = exposed_tool_definitions(
            self._tools,
            include_early_risers=self._early_risers_mentioned,
        )
        trace: dict[str, Any] = {
            "tools": [],
            "tool_rounds": 0,
            "early_risers_guard": (
                "enabled" if self._early_risers_mentioned else "hidden"
            ),
            "promotion_prompt_enabled": self._early_risers_mentioned,
            "exposed_tools": [
                str(definition["name"]) for definition in exposed_tools
            ],
        }
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
                        policy_context,
                    )
                    self._previous_response_id = response_id
                    self._policy_context = (
                        policy_context if response_id is not None else PolicyContext()
                    )
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
                    policy_context.update(call.name, result)
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
            self._policy_context = previous_policy_context
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
        policy_context: PolicyContext,
    ) -> tuple[str, str | None, str]:
        """Return a compliant final response, rewriting once when needed."""

        if not self._policy.find_violations(text, policy_context):
            return text, response.id, "passed"

        corrected = self._create_response(
            input_items=(
                "Replace your immediately preceding customer-facing reply using "
                "the correction instructions."
            ),
            previous_response_id=response.id,
            instructions=(
                f"{system_prompt(include_early_risers=self._early_risers_mentioned)}"
                f"\n\n{self._policy.rewrite_instructions}"
            ),
            tool_choice="none",
        )
        corrected_text = corrected.output_text.strip()
        corrected_tool_calls = [
            item for item in corrected.output if item.type == "function_call"
        ]
        if (
            corrected_text
            and not corrected_tool_calls
            and not self._policy.find_violations(
                corrected_text,
                policy_context,
            )
        ):
            return corrected_text, corrected.id, "rewritten"
        # The fallback is generated locally, so no API response represents what
        # the customer saw. Reset the remote chain rather than continuing from
        # an unsafe hidden draft on the next turn.
        return self._policy.fallback, None, "fallback"

    @staticmethod
    def _tool_outcome(result: dict[str, Any]) -> str:
        """Summarize a tool result without retaining its arguments or data."""

        if not result.get("ok"):
            return "rejected"
        if "found" in result:
            return "found" if result["found"] else "not_found"
        if "available" in result:
            return "open" if result["available"] else "closed"
        if "eligible" in result:
            return "eligible" if result["eligible"] else "not_eligible"
        return "completed"

    def _create_response(
        self,
        input_items: str | list[dict[str, Any]],
        previous_response_id: str | None,
        instructions: str | None = None,
        tool_choice: str | None = None,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self._model,
            "instructions": (
                instructions
                if instructions is not None
                else system_prompt(
                    include_early_risers=self._early_risers_mentioned,
                )
            ),
            "input": input_items,
            "tools": exposed_tool_definitions(
                self._tools,
                include_early_risers=self._early_risers_mentioned,
            ),
            "parallel_tool_calls": True,
        }
        if previous_response_id is not None:
            request["previous_response_id"] = previous_response_id
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        return self._client.responses.create(**request)
