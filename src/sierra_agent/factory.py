"""Composition helpers shared by terminal, web, and evaluation entry points."""

from openai import OpenAI

from sierra_agent.agent import SierraAgent
from sierra_agent.config import Settings
from sierra_agent.policy import CapabilityPolicy, DEFAULT_POLICY
from sierra_agent.tools import HotelTools, ToolHost


def create_agent(
    *,
    client: OpenAI,
    model: str,
    tools: ToolHost,
    policy: CapabilityPolicy = DEFAULT_POLICY,
) -> SierraAgent:
    """Compose an agent from explicit dependencies."""

    return SierraAgent(
        client=client,
        model=model,
        tools=tools,
        policy=policy,
    )


def build_agent(settings: Settings | None = None) -> SierraAgent:
    """Load local dependencies and construct one customer conversation."""

    resolved = settings or Settings.from_env()
    client = OpenAI(api_key=resolved.openai_api_key)
    tools = HotelTools(
        data_dir=resolved.data_dir,
        promotion_secret=resolved.promotion_secret,
    )
    return create_agent(
        client=client,
        model=resolved.openai_model,
        tools=tools,
    )
