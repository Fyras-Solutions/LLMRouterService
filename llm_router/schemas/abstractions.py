from typing import Protocol
from llm_router.schemas.council_schemas import SelectorVote, CouncilDecision


class Selector(Protocol):
    """
    Base abstraction for all Selectors.
    A Selector is responsible for analyzing a prompt and returning a model choice.
    """
    def select_model(self, prompt: str) -> SelectorVote:
        ...


class Council(Protocol):
    """
    Base abstraction for all Councils.
    A Router orchestrates multiple selectors and returns a final decision.
    """
    def decide(self, prompt: str) -> CouncilDecision:
        ...
