import pytest
from pathlib import Path

from llm_router.providers import AnthropicProvider, OpenAIProvider, GoogleProvider
from llm_router.schemas.env_validator import EnvVarError
from llm_router.exceptions.exceptions import (
    ProviderCompletionError,
    ProviderCostError,
)


def test_providers_load_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=a\nOPENAI_API_KEY=b\nGEMINI_API_KEY=c\n"
    )

    # Should not raise when keys are present
    AnthropicProvider(env_path=env_file)
    OpenAIProvider(env_path=env_file)
    GoogleProvider(env_path=env_file)


def test_provider_missing_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("")

    with pytest.raises(EnvVarError):
        OpenAIProvider(env_path=env_file)


def test_provider_completion_error(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=a\n")
    provider = OpenAIProvider(env_path=env_file)

    def boom(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr("llm_router.providers.openai.completion", boom)
    with pytest.raises(ProviderCompletionError):
        provider.complete(model="gpt", prompt="hi")


def test_provider_cost_error(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=a\n")
    provider = OpenAIProvider(env_path=env_file)

    def boom(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr("llm_router.providers.openai.cost_per_token", boom)
    with pytest.raises(ProviderCostError):
        provider.get_cost(model="gpt", prompt_tokens=1, completion_tokens=1)
