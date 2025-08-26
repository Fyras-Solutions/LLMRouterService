# ---------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------
class UsableModelForPromptError(Exception):
    """Raised when no model is usable for the given prompt."""


class SelectorError(Exception):
    """Raised when a selector fails unexpectedly."""


class RouterError(Exception):
    """Raised when the router fails to aggregate votes or return a decision."""