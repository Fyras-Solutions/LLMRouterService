from typing import Protocol, runtime_checkable

from fyras_models import CouncilDecision,SelectorVote


@runtime_checkable
class Selector(Protocol):
    """Base abstraction for all Selectors."""

    def select_model(self, prompt: str) -> SelectorVote:
        ...


@runtime_checkable
class Council(Protocol):
    """Base abstraction for all Councils."""

    def decide(self, prompt: str) -> CouncilDecision:
        ...

