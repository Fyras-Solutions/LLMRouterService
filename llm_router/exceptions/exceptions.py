"""Custom exception hierarchy for the LLM Router service."""


class LLMRouterError(Exception):
    """Base class for all custom exceptions."""
    def __init__(self, message: str, **kwargs):
        self.message = message
        self.details = kwargs
        super().__init__(self.message)


class UsableModelForPromptError(LLMRouterError):
    """Raised when no model is usable for the given prompt."""
    def __init__(self, message: str, prompt: str = None, **kwargs):
        super().__init__(message, prompt=prompt, **kwargs)


class SelectorError(LLMRouterError):
    """Raised when a selector fails unexpectedly."""
    def __init__(self, message: str, selector: str = None, **kwargs):
        super().__init__(message, selector=selector, **kwargs)


class CouncilError(LLMRouterError):
    """Raised when the council cannot reach a decision."""
    def __init__(self, message: str, votes: list = None, **kwargs):
        super().__init__(message, votes=votes, **kwargs)


class ModelExecutionError(LLMRouterError):
    """Raised when executing the final model fails."""
    def __init__(self, message: str, model: str = None, error_code: int = None, **kwargs):
        super().__init__(message, model=model, error_code=error_code, **kwargs)


class RouterError(LLMRouterError):
    """Raised when the router fails to aggregate votes or return a decision."""
    def __init__(self, message: str, selectors: list = None, **kwargs):
        super().__init__(message, selectors=selectors, **kwargs)
