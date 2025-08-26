import logging
from typing import Dict, List

from tqdm import tqdm

from llm_router.schemas.abstractions import Council, Selector
from llm_router.schemas.council_schemas import CouncilDecision, SelectorVote
from llm_router.exceptions.exceptions import CouncilError

logger = logging.getLogger(__name__)


class ParallelCouncil(Council):
    """Run selectors and choose the majority model."""

    def __init__(self, selectors: List[Selector], default_model: str = "ollama/phi3:latest"):
        self.selectors = selectors
        self.default_model = default_model

    def decide(self, prompt: str) -> CouncilDecision:
        votes: List[SelectorVote] = []
        tally: Dict[str, int] = {}
        for selector in tqdm(self.selectors, desc="Selector votes"):
            try:
                result = selector.select_model(prompt)
            except Exception as exc:
                logger.exception("Selector failed during voting", exc_info=exc)
                continue
            votes.append(result)
            tally[result.model] = tally.get(result.model, 0) + 1

        if not tally:
            raise CouncilError("No valid selector votes collected")

        final_model = max(tally, key=tally.get) if tally else self.default_model

        return CouncilDecision(
            final_model=final_model,
            votes=votes,
            weighted_results=tally,
            metadata={"prompt_length": str(len(prompt.split()))},
        )

