import json
import logging
from transformers import pipeline
from transformers.pipelines.base import PipelineException

from fyras_models import SelectorVote
from llm_router.schemas.config import CANDIDATE_LABELS, TOPIC_TO_MODEL
from llm_router.exceptions.exceptions import SelectorError

logger = logging.getLogger(__name__)


class HFZeroShotSelector:
    """Selector that uses HuggingFace zero-shot classification to choose a model."""

    def __init__(self, provider_name: str = "anthropic") -> None:
        self.provider_name = provider_name
        self.model_name = "facebook/bart-large-mnli"

    def select_model(self, prompt: str) -> SelectorVote:
        try:
            classifier = pipeline("zero-shot-classification", model=self.model_name)
        except Exception as exc:
            logger.exception("Failed to load HF zero-shot model")
            raise SelectorError("Could not initialize zero-shot classifier") from exc

        try:
            result = classifier(prompt, CANDIDATE_LABELS)
        except (PipelineException, ValueError, json.JSONDecodeError) as exc:
            logger.exception("Error during zero-shot classification")
            return self._fallback_vote("Classification failed or returned invalid JSON")

        labels = result.get("labels")
        if not labels:
            logger.warning("Classifier returned no labels")
            return self._fallback_vote("No labels returned from classifier")

        top_label = labels[0]
        selected_model = TOPIC_TO_MODEL.get(top_label, {}).get(
            self.provider_name,
            TOPIC_TO_MODEL["SIMPLE"][self.provider_name]
        )

        return SelectorVote(
            selector_name=self.__class__.__name__,
            model=selected_model,
            rationale=f"Classified as '{top_label}' by zero-shot model"
        )

    def _fallback_vote(self, reason: str) -> SelectorVote:
        fallback_model = TOPIC_TO_MODEL["SIMPLE"][self.provider_name]
        return SelectorVote(
            selector_name=self.__class__.__name__,
            model=fallback_model,
            rationale=reason
        )
