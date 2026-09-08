"""Environment-backed configuration for the agent."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(ValueError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Values needed to start one agent session."""

    openai_api_key: str
    openai_model: str
    openai_eval_model: str
    promotion_secret: str
    data_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from a local .env file and the process environment."""

        load_dotenv(PROJECT_ROOT / ".env")

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        eval_model = os.getenv("OPENAI_EVAL_MODEL", "").strip() or model
        promotion_secret = os.getenv("PROMOTION_SECRET", "").strip()

        if not api_key or api_key == "your-api-key":
            raise ConfigurationError("Set OPENAI_API_KEY in .env.")
        if not model:
            raise ConfigurationError("Set OPENAI_MODEL in .env.")
        if len(promotion_secret) < 16:
            raise ConfigurationError(
                "Set PROMOTION_SECRET in .env to a random value of at least 16 characters."
            )

        return cls(
            openai_api_key=api_key,
            openai_model=model,
            openai_eval_model=eval_model,
            promotion_secret=promotion_secret,
            data_dir=PROJECT_ROOT / "data",
        )
