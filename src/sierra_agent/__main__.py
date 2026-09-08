"""Terminal entry point for the Trailhead Hotel agent."""

import sys

from openai import OpenAIError

from sierra_agent.agent import AgentLoopError
from sierra_agent.config import ConfigurationError
from sierra_agent.factory import build_agent
from sierra_agent.tools import DataError


def main() -> int:
    """Run an interactive chat session until the customer exits."""

    try:
        agent = build_agent()
    except (ConfigurationError, DataError) as error:
        print(f"Setup error: {error}", file=sys.stderr)
        return 1

    print("Trailhead Hotel Agent 🏨")
    print("Ask a question, or type 'exit' to end the chat.")

    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAgent: Safe travels!")
            return 0

        if user_message.casefold() in {"exit", "quit"}:
            print("Agent: Safe travels!")
            return 0
        if not user_message:
            continue

        try:
            response = agent.reply(user_message)
        except OpenAIError:
            print(
                "Agent: I couldn't reach the support service. "
                "Please try again in a moment.",
                file=sys.stderr,
            )
            continue
        except (AgentLoopError, DataError, ValueError):
            print(
                "Agent: I hit an unexpected issue. "
                "Please try your request again.",
                file=sys.stderr,
            )
            continue

        print(f"Agent: {response}")


if __name__ == "__main__":
    raise SystemExit(main())
