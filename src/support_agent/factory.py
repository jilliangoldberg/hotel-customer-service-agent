"""Composition helpers shared by terminal, web, and evaluation entry points."""

from openai import OpenAI

from support_agent.agent import SupportAgent
from support_agent.config import Settings
from support_agent.policy import CapabilityPolicy, DEFAULT_POLICY
from support_agent.tools import HotelTools, ToolHost


def create_client(settings: Settings) -> OpenAI:
    """Build the OpenAI client shared by the agent and evaluation harness."""

    return OpenAI(api_key=settings.api_key)


def create_agent(
    *,
    client: OpenAI,
    model: str,
    tools: ToolHost,
    effort: str = "medium",
    policy: CapabilityPolicy = DEFAULT_POLICY,
) -> SupportAgent:
    """Compose an agent from explicit dependencies."""

    return SupportAgent(
        client=client,
        model=model,
        tools=tools,
        policy=policy,
        effort=effort,
    )


def build_agent(settings: Settings | None = None) -> SupportAgent:
    """Load local dependencies and construct one customer conversation."""

    resolved = settings or Settings.from_env()
    tools = HotelTools(
        data_dir=resolved.data_dir,
        promotion_secret=resolved.promotion_secret,
    )
    return create_agent(
        client=create_client(resolved),
        model=resolved.agent_model,
        tools=tools,
        effort=resolved.agent_effort,
    )
