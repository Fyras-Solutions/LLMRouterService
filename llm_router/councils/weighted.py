import asyncio
import logging
from typing import List, Dict, Optional

from tqdm.asyncio import tqdm_asyncio

from llm_router.schemas.abstractions import Council, Selector
from llm_router.schemas.council_schemas import CouncilDecision, SelectorVote
from llm_router.exceptions.exceptions import CouncilError

logger = logging.getLogger(__name__)


class WeightedMajorityVoteCouncil(Council):
    """Aggregate votes via weighted majority and return the winning model."""

    def __init__(self, selectors: List[Selector], weights: Optional[Dict[str, float]] = None):
        self.selectors = selectors
        self.weights = weights if weights else {s.__class__.__name__: 1.0 for s in selectors}

    async def decide(self, prompt: str) -> CouncilDecision:
        tasks = [selector.select_model(prompt) for selector in self.selectors]
        raw_votes = await tqdm_asyncio.gather(*tasks, desc="Selector votes", return_exceptions=True)

        votes: List[SelectorVote] = []
        weighted_results: Dict[str, float] = {}

        for result in raw_votes:
            if isinstance(result, Exception):
                logger.exception("Selector failed during voting", exc_info=result)
                continue
            weight = self.weights.get(result.selector_name, 1.0)
            result.weight = weight
            votes.append(result)
            weighted_results[result.model] = weighted_results.get(result.model, 0.0) + weight

        if not weighted_results:
            raise CouncilError("No valid selector votes collected")

        final_model = max(weighted_results, key=weighted_results.get)

        return CouncilDecision(
            final_model=final_model,
            votes=votes,
            weighted_results=weighted_results,
            metadata={"prompt_length": str(len(prompt.split()))},
        )


class UnanimousCouncil(Council):
    """Require unanimous agreement among selectors, otherwise use a default model."""

    def __init__(self, selectors: List[Selector], default_model: str):
        self.selectors = selectors
        self.default_model = default_model

    async def decide(self, prompt: str) -> CouncilDecision:
        tasks = [selector.select_model(prompt) for selector in self.selectors]
        raw_votes = await tqdm_asyncio.gather(*tasks, desc="Selector votes", return_exceptions=True)

        votes: List[SelectorVote] = []
        model_votes: Dict[str, int] = {}

        for result in raw_votes:
            if isinstance(result, Exception):
                logger.exception("Selector failed during voting", exc_info=result)
                continue
            votes.append(result)
            model_votes[result.model] = model_votes.get(result.model, 0) + 1

        if not votes:
            raise CouncilError("No valid selector votes collected")

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
            metadata={"prompt_length": str(len(prompt.split()))},
        )
