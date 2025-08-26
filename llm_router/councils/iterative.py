from typing import List, Dict, Any
from llm_router.schemas.abstractions import Council, Selector
from llm_router.schemas.council_schemas import CouncilDecision, SelectorVote


class CascadeCouncil(Council):
    """
    A router that uses a cascade approach: selectors are queried in sequence
    until one provides a confident model choice. If none do, it defaults
    to a predefined model.
    """

    def __init__(self, selectors: List[Selector], default_model: str):
        self.selectors = selectors
        self.default_model = default_model

    def decide(self, prompt: str) -> CouncilDecision:
        votes: List[SelectorVote] = []
        final_model = self.default_model

        for selector in self.selectors:
            vote = selector.select_model(prompt)
            votes.append(vote)
            if vote.confidence > 0.7:  # Assuming confidence is a float between 0 and 1
                final_model = vote.model
                break

        weighted_results = {vote.model: vote.confidence for vote in votes}

        return CouncilDecision(
            final_model=final_model,
            votes=votes,
            weighted_results=weighted_results,
            metadata={"prompt_length": str(len(prompt.split()))}
        )
