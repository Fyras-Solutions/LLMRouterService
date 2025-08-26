import asyncio
import logging

import tiktoken
from textstat import textstat

from llm_router.schemas.council_schemas import SelectorVote
from llm_router.exceptions.exceptions import SelectorError

logger = logging.getLogger(__name__)


class HeuristicsSelector:
    """Simple heuristic-based selector using token counts and keywords."""

    def _select_sync(self, prompt: str) -> SelectorVote:
        enc = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(enc.encode(prompt))
        readability = textstat.flesch_kincaid_grade(prompt)

        if any(x in prompt.lower() for x in ["code", "python", "function", "class"]):
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model="ollama/qwen2.5-coder:latest",
                rationale="Keyword match: code-related",
            )
        if any(x in prompt.lower() for x in ["solve", "integral", "equation", "math"]):
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model="ollama/qwen2-math:latest",
                rationale="Keyword match: math-related",
            )
        if num_tokens < 15 and readability < 6:
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model="ollama/gemma2:2b",
                rationale="Short/simple prompt",
            )
        if num_tokens < 80:
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model="ollama/phi3:latest",
                rationale="Medium complexity",
            )
        return SelectorVote(
            selector_name=self.__class__.__name__,
            model="ollama/mistral:7b",
            rationale="Long/complex prompt",
        )

    async def select_model(self, prompt: str) -> SelectorVote:
        try:
            return await asyncio.to_thread(self._select_sync, prompt)
        except Exception as exc:  # pragma: no cover - protective
            logger.exception("HeuristicsSelector failed")
            raise SelectorError(str(exc)) from exc
