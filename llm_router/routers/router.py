from __future__ import annotations
import json
import logging
import time
import promptlayer
import threading
from pathlib import Path
from llm_router.schemas.abstractions import Council
from llm_router.schemas.council_schemas import (
    CouncilDecision,
    LLMRouterResponse,
    RouterMetadata,
)
from llm_router.exceptions.exceptions import ModelExecutionError, RouterError
from llm_router.schemas.env_validator import validate_env_vars, get_env_var
from llm_router.providers import Provider, AnthropicProvider
from typing import Optional

logger = logging.getLogger(__name__)


class LLMRouterService:
    def __init__(
        self,
        council: Council,
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
        self.council = council
        self.provider = provider or AnthropicProvider(env_path=env_path)

        # Validate all required environment variables
        validate_env_vars(env_path)

        # If no API key provided, get from validated environment
        if not api_key:
            api_key = get_env_var("PROMPTLAYER_API_KEY", env_path)

        self.pl_client = promptlayer.PromptLayer(api_key=api_key)

    def _execute(self, decision: CouncilDecision, prompt: str) -> LLMRouterResponse:
        """Execute call through provider and log with PromptLayer."""
        model = decision.final_model

        try:
            start = time.time()
            resp = self.provider.complete(model=model, prompt=prompt)
            end = time.time()
        except Exception as exc:  # pragma: no cover - network issues
            logger.exception("Model execution failed")
            raise ModelExecutionError(str(exc)) from exc

        # Cost tracking handled by provider
        cost = self.provider.get_cost(
            model=model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )

        response_text = resp.text

        # Build structured I/O for PromptLayer
        prompt_struct = {
            "content": [{"type": "text", "text": prompt}],
            "input_variables": [],
            "template_format": "f-string",
            "type": "completion",
        }
        response_struct = {
            "content": [{"type": "text", "text": response_text}],
            "input_variables": [],
            "template_format": "f-string",
            "type": "completion",
        }

        def log_promptlayer() -> None:
            try:
                self.pl_client.log_request(
                    provider=self.provider.name,
                    model=model,
                    input=prompt_struct,
                    output=response_struct,
                    request_start_time=start,
                    request_end_time=end,
                    parameters={},
                    tags=["council-router", self.provider.name],
                    metadata={
                        "votes": json.dumps([v.model_dump() for v in decision.votes]),
                        "weighted_results": json.dumps(decision.weighted_results),
                    },
                    function_name="LLMRouterService._execute",
                )
            except Exception as exc:  # pragma: no cover - logging shouldn't block flow
                logger.warning("PromptLayer logging failed: %s", exc)

        threading.Thread(target=log_promptlayer, daemon=True).start()

        router_metadata = RouterMetadata(
            votes=[v.model_dump() for v in decision.votes],
            weighted_results=decision.weighted_results,
            tags=["council-router", self.provider.name],
        )

        return LLMRouterResponse(
            model=model,
            prompt=prompt,
            response=response_text,
            cost=cost,
            latency=end - start,
            metadata=router_metadata,
        )

    def invoke(self, prompt: str) -> LLMRouterResponse:
        """Main entry point: ask council to decide, then execute."""
        try:
            decision = self.council.decide(prompt)
        except Exception as exc:  # pragma: no cover - protective
            logger.exception("Council decision failed")
            raise RouterError(str(exc)) from exc

        return self._execute(decision, prompt)

