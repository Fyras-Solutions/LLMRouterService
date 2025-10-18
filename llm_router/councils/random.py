import logging
import random
from typing import List

from tqdm import tqdm

from llm_router.schemas.abstractions import Council, Selector
from fyras_models import CouncilDecision,SelectorVote
from llm_router.exceptions.exceptions import CouncilError

logger = logging.getLogger(__name__)


class RandomCouncil(Council):
    """Randomly select a model from the selectors' votes."""

    def __init__(self, selectors: List[Selector]):
        self.selectors = selectors
        self.random = random.Random()

    def decide(self, prompt: str) -> CouncilDecision:
        votes: List[SelectorVote] = []
        for selector in tqdm(self.selectors, desc="Selector votes"):
            try:
                vote = selector.select_model(prompt)
            except Exception as exc:
                logger.exception("Selector failed during voting", exc_info=exc)
                continue
            votes.append(vote)

        if not votes:
            raise CouncilError("No valid selector votes collected")

        final_model = self.random.choice([vote.model for vote in votes])
        weighted_results = {vote.model: 1 for vote in votes}

        return CouncilDecision(
            final_model=final_model,
            votes=votes,
            weighted_results=weighted_results,
            metadata={"prompt_length": str(len(prompt.split()))},
        )
