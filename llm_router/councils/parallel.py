import logging
from typing import Dict, List, Optional

from tqdm import tqdm

from llm_router.schemas.abstractions import Council, Selector
from fyras_models import CouncilDecision,SelectorVote
from llm_router.exceptions.exceptions import CouncilError
from llm_router.schemas.config import TOPIC_TO_MODEL

logger = logging.getLogger(__name__)


class ParallelCouncil(Council):
    """Run selectors and choose the majority model."""

    def __init__(
        self,
        selectors: List[Selector],
        provider_name: str = "anthropic",
        default_model: Optional[str] = None,
    ) -> None:
        self.selectors = selectors
        self.provider_name = provider_name
        self.default_model = default_model or TOPIC_TO_MODEL["SIMPLE"][provider_name]

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

