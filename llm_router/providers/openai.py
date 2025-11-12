"""Provider implementation for OpenAI's GPT models."""

from __future__ import annotations

import logging
from typing import Any

from litellm import completion, cost_per_token

from llm_router.exceptions.exceptions import (
    ProviderCompletionError,
    ProviderCostError,
)
from .base import Provider, ProviderResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(Provider):
    """Provider implementation for OpenAI's GPT family."""

    api_key_env = "OPENAI_API_KEY"

    @property
    def name(self) -> str:  # pragma: no cover - simple property
        return "openai"

    def complete(self, model: str, prompt: str) -> ProviderResponse:
        try:
            resp: Any = completion(model=model, messages=[{"role": "user", "content": prompt}])
            text = resp["choices"][0]["message"]["content"]
            usage = getattr(resp, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
        except Exception as exc:
            raise ProviderCompletionError(str(exc), provider=self.name, model=model) from exc
        return ProviderResponse(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    def get_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        try:
            cost = cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            if isinstance(cost, tuple) and cost:
                cost = cost[0]
            return float(cost)
        except Exception as exc:
            raise ProviderCostError(str(exc), provider=self.name, model=model) from exc

