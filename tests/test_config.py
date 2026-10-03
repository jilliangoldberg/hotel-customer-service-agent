"""Configuration contract tests."""

import pytest

from support_agent.config import ConfigurationError, Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("support_agent.config.load_dotenv", lambda *_a, **_k: None)
    for name in ("OPENAI_API_KEY", "AGENT_MODEL", "EVAL_MODEL", "AGENT_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROMOTION_SECRET", "a-valid-promotion-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_rejects_placeholder_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "your-api-key")

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        Settings.from_env()


def test_missing_api_key_is_rejected(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        Settings.from_env()


def test_defaults_to_current_openai_models() -> None:
    settings = Settings.from_env()

    assert settings.agent_model == "gpt-5.4"
    assert settings.eval_model == "gpt-5.4-mini"
    assert settings.agent_effort == "medium"


def test_models_and_effort_can_be_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "agent-model")
    monkeypatch.setenv("EVAL_MODEL", "judge-model")
    monkeypatch.setenv("AGENT_EFFORT", "High")

    settings = Settings.from_env()

    assert settings.api_key == "test-key"
    assert settings.agent_model == "agent-model"
    assert settings.eval_model == "judge-model"
    assert settings.agent_effort == "high"


def test_rejects_unknown_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EFFORT", "turbo")

    with pytest.raises(ConfigurationError, match="AGENT_EFFORT"):
        Settings.from_env()
