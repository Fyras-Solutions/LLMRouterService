"""Provider implementation for Google's Gemini models."""

from __future__ import annotations

import logging
from typing import Any

from litellm import completion, cost_per_token

from .base import Provider, ProviderResponse

logger = logging.getLogger(__name__)


class GoogleProvider(Provider):
    """Provider implementation for Google Gemini models."""

    api_key_env = "GEMINI_API_KEY"

    @property
    def name(self) -> str:  # pragma: no cover - simple property
        return "google"

    def complete(self, model: str, prompt: str) -> ProviderResponse:
        resp: Any = completion(model=model, messages=[{"role": "user", "content": prompt}])
        text = resp["choices"][0]["message"]["content"]
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
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
        except Exception as exc:  # pragma: no cover - protective
            logger.warning("Cost calculation failed: %s", exc)
            return 0.0

