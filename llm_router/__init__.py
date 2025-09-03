"""LLM Router Service for intelligent model selection and routing."""

from llm_router.routers.router import LLMRouterService  # was LLMRouter
from llm_router.schemas.council_schemas import (
    LLMResponse,
    RouterMetadata,
    LLMRouterResponse,
)
from llm_router.exceptions.exceptions import (
    LLMRouterError,
    UsableModelForPromptError,
    SelectorError,
    CouncilError,
    ModelExecutionError,
    RouterError,
    ProviderError,
    ProviderCompletionError,
    ProviderCostError,
)

__version__ = "0.1.0"

__all__ = [
    "LLMRouterService",
    "LLMResponse",
    "RouterMetadata",
    "LLMRouterResponse",
    "LLMRouterError",
    "UsableModelForPromptError",
    "SelectorError",
    "CouncilError",
    "ModelExecutionError",
    "RouterError",
    "ProviderError",
    "ProviderCompletionError",
    "ProviderCostError",
]
