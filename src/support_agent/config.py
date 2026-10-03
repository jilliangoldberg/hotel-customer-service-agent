"""Environment-backed configuration for the agent."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENT_MODEL = "gpt-5.4"
DEFAULT_EVAL_MODEL = "gpt-5.4-mini"
DEFAULT_EFFORT = "medium"
EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh")


class ConfigurationError(ValueError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Values needed to start one agent session."""

    api_key: str | None
    agent_model: str
    eval_model: str
    agent_effort: str
    promotion_secret: str
    data_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from a local .env file and the process environment."""

        load_dotenv(PROJECT_ROOT / ".env")

        api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
        model = os.getenv("AGENT_MODEL", DEFAULT_AGENT_MODEL).strip()
        eval_model = os.getenv("EVAL_MODEL", DEFAULT_EVAL_MODEL).strip()
        effort = os.getenv("AGENT_EFFORT", DEFAULT_EFFORT).strip().lower()
        promotion_secret = os.getenv("PROMOTION_SECRET", "").strip()

        if not api_key or api_key == "your-api-key":
            raise ConfigurationError("Set OPENAI_API_KEY in .env.")
        if not model:
            raise ConfigurationError("Set AGENT_MODEL in .env.")
        if not eval_model:
            raise ConfigurationError("Set EVAL_MODEL in .env.")
        if effort not in EFFORT_LEVELS:
            raise ConfigurationError(
                f"Set AGENT_EFFORT to one of: {', '.join(EFFORT_LEVELS)}."
            )
        if len(promotion_secret) < 16:
            raise ConfigurationError(
                "Set PROMOTION_SECRET in .env to a random value of at least 16 characters."
            )

        return cls(
            api_key=api_key,
            agent_model=model,
            eval_model=eval_model,
            agent_effort=effort,
            promotion_secret=promotion_secret,
            data_dir=PROJECT_ROOT / "data",
        )
