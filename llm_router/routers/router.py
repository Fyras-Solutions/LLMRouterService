from __future__ import annotations
import json
import logging
import time
import promptlayer
import threading
from pathlib import Path
from llm_router.schemas.abstractions import Selector
from fyras_models import (
    SelectorVote,
    LLMRouterResponse
)

from llm_router.exceptions.exceptions import (
    ModelExecutionError,
    RouterError,
    ProviderError,
)
from llm_router.schemas.env_validator import validate_env_vars, get_env_var
from llm_router.providers import Provider, AnthropicProvider
from typing import Optional

logger = logging.getLogger(__name__)


class LLMRouterService:
    def __init__(
        self,
        Selector: Selector,
        api_key: str | None = None,
        env_path: Optional[Path] = None,
        provider: Provider | None = None,
    ):
        """Initialize the LLM Router Service.

        Args:
            council: The council implementation to use for decision making.
            api_key: Optional PromptLayer API key. If not provided, will look for
                ``PROMPTLAYER_API_KEY`` in the environment.
            env_path: Optional path to a ``.env`` file to load required variables.
            provider: Optional provider implementation. Defaults to
                :class:`AnthropicProvider`.

        Raises:
            EnvVarError: If required environment variables are missing.
        """
        self.Selector = Selector
        self.provider = provider or AnthropicProvider(env_path=env_path)

        # Validate all required environment variables
        validate_env_vars(env_path)

        # If no API key provided, get from validated environment
        if not api_key:
            api_key = get_env_var("PROMPTLAYER_API_KEY", env_path)

        self.pl_client = promptlayer.PromptLayer(api_key=api_key)

    def _execute(self, Selector: SelectorVote, prompt: str) -> LLMRouterResponse:
        """Execute call through provider and log with PromptLayer."""
        model = Selector.model

        try:
            start = time.time()
            resp = self.provider.complete(model=model, prompt=prompt)
            end = time.time()
        except ProviderError as exc:  # pragma: no cover - network issues
            logger.exception("Model execution failed")
            raise ModelExecutionError(str(exc)) from exc

        # Cost tracking handled by provider
        try:
            cost = self.provider.get_cost(
                model=model,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
            )
        except ProviderError as exc:  # pragma: no cover - cost issues shouldn't block
            logger.warning("Cost calculation failed: %s", exc)
            cost = 0.0

        response_text = resp.text

        return LLMRouterResponse(
            model=model,
            prompt=prompt,
            response=response_text,
            cost=cost,
            latency=end - start,
        )

    def invoke(self, prompt: str) -> LLMRouterResponse:
        """Main entry point: ask council to decide, then execute."""
        try:
            decision = self.Selector.select_model(prompt)
        except Exception as exc:  # pragma: no cover - protective
            logger.exception("Council decision failed")
            raise RouterError(str(exc)) from exc

        return self._execute(decision, prompt)

