"""Configuration contract tests."""

import pytest

from sierra_agent.config import ConfigurationError, Settings


def test_rejects_placeholder_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "your-api-key")
    monkeypatch.setenv("PROMOTION_SECRET", "a-valid-promotion-secret")

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        Settings.from_env()


def test_eval_model_defaults_to_agent_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "agent-model")
    monkeypatch.delenv("OPENAI_EVAL_MODEL", raising=False)
    monkeypatch.setenv("PROMOTION_SECRET", "a-valid-promotion-secret")

    settings = Settings.from_env()

    assert settings.openai_model == "agent-model"
    assert settings.openai_eval_model == "agent-model"


def test_eval_model_can_be_configured_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "agent-model")
    monkeypatch.setenv("OPENAI_EVAL_MODEL", "judge-model")
    monkeypatch.setenv("PROMOTION_SECRET", "a-valid-promotion-secret")

    settings = Settings.from_env()

    assert settings.openai_eval_model == "judge-model"
