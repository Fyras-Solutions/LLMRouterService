import asyncio
import logging
from llm_router.schemas.council_schemas import SelectorVote
from llm_router.exceptions.exceptions import SelectorError

logger = logging.getLogger(__name__)


class PromptLengthSelector:
    """Selects a model purely based on the prompt length."""

    def __init__(self, thresholds=None):
        # thresholds: list of tuples (max_length, model)
        self.thresholds = thresholds or [
            (15, "ollama/gemma2:2b"),
            (80, "ollama/phi3:latest"),
        ]
        self.default_model = "ollama/mistral:7b"

    def _select_sync(self, prompt: str) -> SelectorVote:
        length = len(prompt.split())
        for limit, model in self.thresholds:
            if length <= limit:
                return SelectorVote(
                    selector_name=self.__class__.__name__,
                    model=model,
                    rationale=f"Prompt length {length} <= {limit}",
                )
        return SelectorVote(
            selector_name=self.__class__.__name__,
            model=self.default_model,
            rationale=f"Prompt length {length} exceeds thresholds",
        )

    async def select_model(self, prompt: str) -> SelectorVote:
        try:
            return await asyncio.to_thread(self._select_sync, prompt)
        except Exception as exc:  # pragma: no cover - protective
            logger.exception("PromptLengthSelector failed")
            raise SelectorError(str(exc)) from exc
