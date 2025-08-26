"""Custom exception hierarchy for the LLM Router service."""


class LLMRouterError(Exception):
    """Base class for all custom exceptions."""


class UsableModelForPromptError(LLMRouterError):
    """Raised when no model is usable for the given prompt."""


class SelectorError(LLMRouterError):
    """Raised when a selector fails unexpectedly."""


class CouncilError(LLMRouterError):
    """Raised when the council cannot reach a decision."""


class ModelExecutionError(LLMRouterError):
    """Raised when executing the final model fails."""


class RouterError(LLMRouterError):
    """Raised when the router fails to aggregate votes or return a decision."""
