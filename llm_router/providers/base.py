from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from llm_router.schemas.env_validator import EnvVarError


logger = logging.getLogger(__name__)


class ProviderResponse(BaseModel):
    """Standard response returned from a provider."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class Provider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must define the ``api_key_env`` class attribute with the name of
    the environment variable LiteLLM expects for authentication. During
    initialization we optionally load variables from a ``.env`` file and ensure
    the required key is present, raising a detailed :class:`EnvVarError` if not.
    """

    #: Name of the environment variable used for the provider API key
    api_key_env: str

    def __init__(self, env_path: Path | None = None) -> None:
        self.env_path = env_path
        if env_path:
            if not env_path.exists():
                raise EnvVarError(f"Environment file not found at {env_path}")
            if not load_dotenv(env_path):
                raise EnvVarError(
                    f"Unable to load environment variables from {env_path}"
                )
        key = os.getenv(self.api_key_env)
        if not key:
            raise EnvVarError(
                f"Environment variable {self.api_key_env} is required for the"
                f" {self.name} provider. Set it in your environment or .env"
                f" file at {env_path}."
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Human readable provider name."""
        raise NotImplementedError

    @abstractmethod
    def complete(self, model: str, prompt: str) -> ProviderResponse:
        """Execute a completion request against the provider."""
        raise NotImplementedError

    @abstractmethod
    def get_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Return the cost for the request based on token usage."""
        raise NotImplementedError
