from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class ProviderResponse:
    """Standard response returned from a provider."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class Provider(ABC):
    """Abstract base class for LLM providers."""

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
