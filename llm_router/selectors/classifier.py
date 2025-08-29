import json
import logging
from typing import Any

import requests

from llm_router.schemas.council_schemas import SelectorVote
from llm_router.schemas.config import (
    CANDIDATE_LABELS,
    HF_API_URL,
    TOPIC_TO_MODEL,
    get_hf_headers,
)
from llm_router.exceptions.exceptions import SelectorError

logger = logging.getLogger(__name__)


class HFZeroShotSelector:
    """Selector that queries a HuggingFace zero-shot classifier."""

    def __init__(self, provider_name: str = "anthropic") -> None:
        self.provider_name = provider_name

    def select_model(self, prompt: str) -> SelectorVote:
        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {"candidate_labels": CANDIDATE_LABELS},
        }
        try:
            resp = requests.post(
                HF_API_URL,
                headers=get_hf_headers(),
                json=payload,
                timeout=30,
            )
        except Exception as exc:  # pragma: no cover - network issues
            logger.exception("HFZeroShotSelector request failed")
            raise SelectorError(str(exc)) from exc

        if resp.status_code != 200:
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model=TOPIC_TO_MODEL["general"][self.provider_name],
                rationale=f"HF API error status {resp.status_code}",
            )

        try:
            result = resp.json()
        except (ValueError, json.JSONDecodeError) as exc:
            logger.exception("HFZeroShotSelector JSON parse error")
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model=TOPIC_TO_MODEL["general"][self.provider_name],
                rationale="Invalid JSON from HF API",
            )

        if "labels" in result:
            top_label = result["labels"][0]
            provider_models = TOPIC_TO_MODEL.get(top_label, {})
            mapped_model = provider_models.get(
                self.provider_name, TOPIC_TO_MODEL["general"][self.provider_name]
            )
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model=mapped_model,
                rationale=f"Zero-shot classified as {top_label}",
            )

        return SelectorVote(
            selector_name=self.__class__.__name__,
            model=TOPIC_TO_MODEL["general"][self.provider_name],
            rationale="Fallback default model",
        )
