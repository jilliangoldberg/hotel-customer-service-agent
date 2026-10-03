"""Hotel guest conversations orchestrated by the OpenAI Agents SDK."""

import asyncio
import json
import re
from time import perf_counter
from typing import Any

from agents import Agent, FunctionTool, MaxTurnsExceeded, Model, ModelBehaviorError, ModelSettings
from agents import OpenAIResponsesModel, RunConfig, RunHooks, Runner
from openai import AsyncOpenAI, OpenAI
from openai.types.shared import Reasoning

from support_agent.policy import CapabilityPolicy, DEFAULT_POLICY, PolicyContext
from support_agent.prompt import EARLY_RISERS_PROMPT, SYSTEM_PROMPT
from support_agent.tools import ToolHost, exposed_tool_definitions

_EARLY_RISERS_PATTERN = re.compile(r"\bearly[\s-]+risers?\b", re.IGNORECASE)


class AgentLoopError(RuntimeError):
    """Raised when a complete, safe customer response cannot be produced."""


def ensure_complete(response: Any) -> None:
    """Reject incomplete Responses API results and explicit refusals."""

    if response.status != "completed":
        reason = getattr(response.incomplete_details, "reason", response.status)
        raise AgentLoopError(f"The model stopped unexpectedly ({reason}).")
    ensure_output_complete(response.output)


def ensure_output_complete(output: list[Any]) -> None:
    """Reject partial messages and refusal blocks before the runner uses them."""

    for item in output:
        if item.type == "message":
            if item.status != "completed":
                raise AgentLoopError("The model returned an incomplete message.")
            if any(block.type == "refusal" for block in item.content):
                raise AgentLoopError("The model declined to respond.")


class CheckedResponsesModel(OpenAIResponsesModel):
    """Validate output messages after the SDK checks the raw response status."""

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        response = await super().get_response(*args, **kwargs)
        ensure_output_complete(response.output)
        return response


class SupportAgent:
    """Own one conversation, committing history only after a safe response."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools: ToolHost,
        policy: CapabilityPolicy = DEFAULT_POLICY,
        effort: str = "medium",
        max_tool_rounds: int = 5,
        max_tokens: int = 16000,
        sdk_model: Model | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._tools = tools
        self._policy = policy
        self._effort = effort
        self._max_tool_rounds = max_tool_rounds
        self._max_tokens = max_tokens
        self._sdk_model = sdk_model
        self._messages: list[dict[str, Any]] = []
        self._policy_context = PolicyContext()
        self._early_risers_mentioned = False
        self._last_trace: dict[str, Any] = {}

    @property
    def last_trace(self) -> dict[str, Any]:
        return {
            **self._last_trace,
            "tools": [dict(item) for item in self._last_trace.get("tools", [])],
            "exposed_tools": list(self._last_trace.get("exposed_tools", [])),
        }

    def reply(self, user_message: str) -> str:
        """Run a synchronous terminal/Flask turn with a scoped async client."""

        # A Flask session can move between request threads. Creating and closing
        # the async client inside each run avoids reusing it on a closed loop.
        return asyncio.run(self._reply(user_message))

    async def _reply(self, user_message: str) -> str:
        if self._sdk_model is not None:
            return await self._run_turn(user_message, self._sdk_model)
        async with AsyncOpenAI(
            api_key=self._client.api_key,
            base_url=self._client.base_url,
            organization=self._client.organization,
            project=self._client.project,
        ) as client:
            return await self._run_turn(
                user_message, CheckedResponsesModel(self._model, client)
            )

    async def _run_turn(self, user_message: str, model: Model) -> str:
        started_at = perf_counter()
        mentioned = self._early_risers_mentioned or bool(
            _EARLY_RISERS_PATTERN.search(user_message)
        )
        policy_context = self._policy_context.copy()
        inputs = [*self._messages, {"role": "user", "content": user_message}]
        definitions = exposed_tool_definitions(
            self._tools, include_early_risers=mentioned
        )
        trace: dict[str, Any] = {
            "tools": [], "tool_rounds": 0,
            "early_risers_guard": "enabled" if mentioned else "hidden",
            "promotion_prompt_enabled": mentioned,
            "exposed_tools": [definition["name"] for definition in definitions],
        }
        config = RunConfig(tracing_disabled=True)
        limit = self._max_tool_rounds

        class ToolRoundLimit(RunHooks):
            async def on_llm_end(self, _context: Any, _agent: Any, response: Any) -> None:
                if any(item.type == "function_call" for item in response.output):
                    if trace["tool_rounds"] >= limit:
                        raise AgentLoopError("The model exceeded the tool-call limit.")
                    trace["tool_rounds"] += 1

        instructions = SYSTEM_PROMPT
        if mentioned:
            instructions += "\n\n" + EARLY_RISERS_PROMPT

        def sdk_tool(definition: dict[str, Any]) -> FunctionTool:
            name = definition["name"]

            async def invoke(_context: Any, arguments: str) -> str:
                result = self._tools.execute(name, json.loads(arguments))
                policy_context.update(name, result)
                trace["tools"].append({
                    "name": name, "outcome": self._tool_outcome(result)
                })
                return json.dumps(result, separators=(",", ":"))

            return FunctionTool(
                name=name, description=definition["description"],
                params_json_schema=definition["parameters"],
                on_invoke_tool=invoke,
                strict_json_schema=definition.get("strict", True),
            )

        sdk_agent = Agent(
            name="Trailhead Hotel Guest Support",
            instructions=instructions,
            model=model,
            tools=[sdk_tool(definition) for definition in definitions],
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=self._effort),
                max_tokens=self._max_tokens, parallel_tool_calls=True,
                store=False,
            ),
        )
        try:
            # A turn is one model call; five tool rounds need six model calls.
            result = await Runner.run(
                sdk_agent, inputs, max_turns=self._max_tool_rounds + 1,
                run_config=config, hooks=ToolRoundLimit(),
            )
            trace["tool_rounds"] = sum(
                any(item.type == "function_call" for item in response.output)
                for response in result.raw_responses
            )
            text = str(result.final_output).strip()
            if not text:
                raise AgentLoopError("The model returned no text response.")
            history = result.to_input_list()
            policy_outcome = "passed"
            if self._policy.find_violations(text, policy_context):
                rewrite_agent = sdk_agent.clone(
                    instructions=instructions + "\n\n" + self._policy.rewrite_instructions,
                    tools=[], model_settings=sdk_agent.model_settings.resolve(
                        ModelSettings(tool_choice="none")
                    ),
                )
                rewrite = await Runner.run(
                    rewrite_agent,
                    [*history, {"role": "user", "content": "Replace the preceding reply using the correction instructions."}],
                    max_turns=1, run_config=config,
                )
                corrected = str(rewrite.final_output).strip()
                if corrected and not self._policy.find_violations(corrected, policy_context):
                    text, policy_outcome = corrected, "rewritten"
                else:
                    text, policy_outcome = self._policy.fallback, "fallback"
                # Preserve trusted tool evidence, replacing unsafe drafts and
                # removing correction instructions from future customer turns.
                history = [
                    *inputs,
                    *(item.to_input_item() for item in result.new_items
                      if item.type in {"tool_call_item", "tool_call_output_item", "reasoning_item"}),
                    {"role": "assistant", "content": text},
                ]
            self._messages = history
            self._policy_context = policy_context
            self._early_risers_mentioned = mentioned
            trace.update(outcome="completed", response_policy=policy_outcome)
            return text
        except ModelBehaviorError as error:
            trace.update(outcome="failed", error="AgentLoopError")
            raise AgentLoopError(f"The model returned an invalid response: {error}") from error
        except MaxTurnsExceeded as error:
            trace.update(outcome="failed", error="AgentLoopError")
            raise AgentLoopError("The model exceeded the tool-call limit.") from error
        except Exception as error:
            trace.update(outcome="failed", error=type(error).__name__)
            raise
        finally:
            trace["duration_ms"] = round((perf_counter() - started_at) * 1000)
            self._last_trace = trace

    @staticmethod
    def _tool_outcome(result: dict[str, Any]) -> str:
        if not result.get("ok"):
            return "rejected"
        for field, yes, no in (("found", "found", "not_found"),
                               ("available", "open", "closed"),
                               ("eligible", "eligible", "not_eligible")):
            if field in result:
                return yes if result[field] else no
        return "completed"
