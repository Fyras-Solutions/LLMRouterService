import logging
from typing import Optional

from langchain_ollama import ChatOllama

from llm_router.schemas.council_schemas import SelectorVote
from llm_router.schemas.config import TOPIC_TO_MODEL
from llm_router.exceptions.exceptions import SelectorError

logger = logging.getLogger(__name__)


class SLMSelector:
    def __init__(
        self,
        model: str = "gemma2:2b",
        prompt_template: Optional[str] = None,
        provider_name: str = "anthropic",
    ):
        self.model = model
        self.provider_name = provider_name
        if prompt_template:
            self.selector_prompt = prompt_template
        else:
            lines = ["You are a STRICT model selector. Choose exactly one from:"]
            for topic, models in TOPIC_TO_MODEL.items():
                lines.append(
                    f"- {models[provider_name]} → {topic} tasks"
                )
            lines.append("\nReturn only the model code.")
            self.selector_prompt = "\n".join(lines)

    def select_model(self, prompt: str) -> SelectorVote:
        chat = ChatOllama(model=self.model, temperature=0, num_predict=50)
        full_prompt = f"{self.selector_prompt}\n\nUser Prompt:\n{prompt}\n\nSelected Model:"
        try:
            resp = chat.invoke(full_prompt)
        except Exception as exc:  # pragma: no cover - protective
            logger.exception("SLMSelector failed")
            raise SelectorError(str(exc)) from exc

        selection = resp.content.strip()
        allowed_models = [m[self.provider_name] for m in TOPIC_TO_MODEL.values()]
        if selection not in allowed_models:
            selection = TOPIC_TO_MODEL["simple"][self.provider_name]

        return SelectorVote(
            selector_name=self.__class__.__name__,
            model=selection,
            rationale="Decision made by small local model",
        )
