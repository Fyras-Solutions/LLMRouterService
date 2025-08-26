import logging
from typing import Optional

from langchain_ollama import ChatOllama

from llm_router.schemas.council_schemas import SelectorVote
from llm_router.schemas.config import TOPIC_TO_MODEL
from llm_router.exceptions.exceptions import SelectorError

logger = logging.getLogger(__name__)


class SLMSelector:
    def __init__(self, model: str = "gemma2:2b", prompt_template: Optional[str] = None):
        self.model = model
        self.selector_prompt = prompt_template or (
            """
            You are a STRICT model selector. Choose exactly one from:
            - ollama/gemma2:2b → very short/simple factual questions
            - ollama/phi3:latest → general-purpose, moderate complexity
            - ollama/mistral:7b → long/complex reasoning tasks
            - ollama/qwen2.5-coder:latest → coding, programming, debugging
            - ollama/qwen2-math:latest → math/calculus/algebra

            Return only the model code.
            """
        )

    def select_model(self, prompt: str) -> SelectorVote:
        chat = ChatOllama(model=self.model, temperature=0, num_predict=50)
        full_prompt = f"{self.selector_prompt}\n\nUser Prompt:\n{prompt}\n\nSelected Model:"
        try:
            resp = chat.invoke(full_prompt)
        except Exception as exc:  # pragma: no cover - protective
            logger.exception("SLMSelector failed")
            raise SelectorError(str(exc)) from exc

        selection = resp.content.strip()
        if selection not in TOPIC_TO_MODEL.values():
            selection = "ollama/phi3:latest"

        return SelectorVote(
            selector_name=self.__class__.__name__,
            model=selection,
            rationale="Decision made by small local model",
        )
