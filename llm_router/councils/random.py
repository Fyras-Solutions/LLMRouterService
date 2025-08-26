from typing import List
from llm_router.schemas.abstractions import Council, Selector
from llm_router.schemas.council_schemas import CouncilDecision, SelectorVote

class RandomCouncil(Council):
    """
    A router that randomly selects a model from the votes of multiple selectors.
    """
    import random

    def __init__(self, selectors: List[Selector]):
        self.selectors = selectors

    def decide(self, prompt: str) -> CouncilDecision:
        votes: List[SelectorVote] = []

        for selector in self.selectors:
            vote = selector.select_model(prompt)
            votes.append(vote)

        final_model = self.random.choice([vote.model for vote in votes])

        weighted_results = {vote.model: 1 for vote in votes}

        return CouncilDecision(
            final_model=final_model,
            votes=votes,
            weighted_results=weighted_results,
            metadata={"prompt_length": str(len(prompt.split()))}
        )