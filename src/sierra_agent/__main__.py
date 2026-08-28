"""Terminal entry point for the Trailhead Hotel agent."""

import sys

from openai import OpenAI, OpenAIError

from sierra_agent.agent import AgentLoopError, SierraAgent
from sierra_agent.config import ConfigurationError, Settings
from sierra_agent.tools import DataError, HotelTools


def build_agent() -> SierraAgent:
    """Construct the agent and its local dependencies."""

    settings = Settings.from_env()
    client = OpenAI(api_key=settings.openai_api_key)
    tools = HotelTools(
        data_dir=settings.data_dir,
        promotion_secret=settings.promotion_secret,
    )
    return SierraAgent(client=client, model=settings.openai_model, tools=tools)


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
