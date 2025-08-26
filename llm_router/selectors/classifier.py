import asyncio
import json
import logging
from typing import Any

import requests

from llm_router.schemas.council_schemas import SelectorVote
from llm_router.schemas.config import CANDIDATE_LABELS, HF_HEADERS, HF_API_URL, TOPIC_TO_MODEL
from llm_router.exceptions.exceptions import SelectorError

logger = logging.getLogger(__name__)


class HFZeroShotSelector:
    """Selector that queries a HuggingFace zero-shot classifier."""

    async def select_model(self, prompt: str) -> SelectorVote:
        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {"candidate_labels": CANDIDATE_LABELS},
        }
        try:
            resp = await asyncio.to_thread(
                requests.post,
                HF_API_URL,
                headers=HF_HEADERS,
                json=payload,
                timeout=30,
            )
        except Exception as exc:  # pragma: no cover - network issues
            logger.exception("HFZeroShotSelector request failed")
            raise SelectorError(str(exc)) from exc

        if resp.status_code != 200:
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model="ollama/phi3:latest",
                rationale=f"HF API error status {resp.status_code}",
            )

        try:
            result = resp.json()
        except (ValueError, json.JSONDecodeError) as exc:
            logger.exception("HFZeroShotSelector JSON parse error")
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model="ollama/phi3:latest",
                rationale="Invalid JSON from HF API",
            )

        if "labels" in result:
            top_label = result["labels"][0]
            mapped_model = TOPIC_TO_MODEL.get(top_label, "ollama/phi3:latest")
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model=mapped_model,
                rationale=f"Zero-shot classified as {top_label}",
            )

        return SelectorVote(
            selector_name=self.__class__.__name__,
            model="ollama/phi3:latest",
            rationale="Fallback default model",
        )
