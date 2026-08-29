"""Configuration contract tests."""

import pytest

from sierra_agent.config import ConfigurationError, Settings


def test_rejects_placeholder_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "your-api-key")
    monkeypatch.setenv("PROMOTION_SECRET", "a-valid-promotion-secret")

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        Settings.from_env()
