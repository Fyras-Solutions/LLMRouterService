import pytest
from pathlib import Path

from llm_router.providers import AnthropicProvider, OpenAIProvider, GoogleProvider
from llm_router.schemas.env_validator import EnvVarError


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
