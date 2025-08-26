from typing import List, Dict, Optional
from llm_router.schemas.abstractions import Council, Selector
from llm_router.schemas.council_schemas import CouncilDecision, SelectorVote


class WeightedMajorityVoteCouncil(Council):
    """
    A hybrid router that aggregates multiple selectors' votes
    and decides the final model via weighted majority voting.
    """

    def __init__(self, selectors: List[Selector], weights: Optional[Dict[str, float]] = None):
        self.selectors = selectors
        self.weights = weights if weights else {s.__class__.__name__: 1.0 for s in selectors}

    def decide(self, prompt: str) -> CouncilDecision:
        votes: List[SelectorVote] = []
        weighted_results: Dict[str, float] = {}

        for selector in self.selectors:
            vote = selector.select_model(prompt)
            weight = self.weights.get(vote.selector_name, 1.0)
            vote.weight = weight
            votes.append(vote)
            weighted_results[vote.model] = weighted_results.get(vote.model, 0.0) + weight

        final_model = max(weighted_results, key=weighted_results.get)

        return CouncilDecision(
            final_model=final_model,
            votes=votes,
            weighted_results=weighted_results,
            metadata={"prompt_length": str(len(prompt.split()))}
        )


class UnanimousCouncil(Council):
    """
    A router that requires unanimous agreement among selectors
    to choose a model. If no unanimous decision is reached,
    it defaults to a predefined model.
    """

    def __init__(self, selectors: List[Selector], default_model: str):
        self.selectors = selectors
        self.default_model = default_model

    def decide(self, prompt: str) -> CouncilDecision:
        votes: List[SelectorVote] = []
        model_votes: Dict[str, int] = {}

        for selector in self.selectors:
            vote = selector.select_model(prompt)
            votes.append(vote)
            model_votes[vote.model] = model_votes.get(vote.model, 0) + 1

        unanimous_model = None
        for model, count in model_votes.items():
            if count == len(self.selectors):
                unanimous_model = model
                break

        final_model = unanimous_model if unanimous_model else self.default_model

        return CouncilDecision(
            final_model=final_model,
            votes=votes,
            weighted_results=model_votes,
            metadata={"prompt_length": str(len(prompt.split()))}
        )