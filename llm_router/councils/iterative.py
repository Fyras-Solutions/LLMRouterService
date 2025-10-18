import logging
from typing import List, Dict, Any

from tqdm import tqdm

from llm_router.schemas.abstractions import Council, Selector
from fyras_models import CouncilDecision,SelectorVote
from llm_router.exceptions.exceptions import CouncilError


logger = logging.getLogger(__name__)


class CascadeCouncil(Council):
    """Query selectors sequentially until one is confident."""

    def __init__(self, selectors: List[Selector], default_model: str):
        self.selectors = selectors
        self.default_model = default_model

    def decide(self, prompt: str) -> CouncilDecision:
        votes: List[SelectorVote] = []
        final_model = self.default_model

        for selector in tqdm(self.selectors, desc="Cascade selectors"):
            try:
                vote = selector.select_model(prompt)
            except Exception as exc:
                logger.exception("Selector failed in cascade", exc_info=exc)
                continue
            votes.append(vote)
            if vote.confidence > 0.7:
                final_model = vote.model
                break

        if not votes:
            raise CouncilError("No valid selector votes collected")

        weighted_results = {vote.model: vote.confidence for vote in votes}

        return CouncilDecision(
            final_model=final_model,
            votes=votes,
            weighted_results=weighted_results,
            metadata={"prompt_length": str(len(prompt.split()))},
        )
