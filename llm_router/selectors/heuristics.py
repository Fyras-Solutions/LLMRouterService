import logging

import tiktoken
from textstat import textstat

from llm_router.schemas.council_schemas import SelectorVote
from llm_router.schemas.config import TOPIC_TO_MODEL
from llm_router.exceptions.exceptions import SelectorError

logger = logging.getLogger(__name__)


class HeuristicsSelector:
    """Simple heuristic-based selector using token counts and keywords."""

    def __init__(self, provider_name: str = "anthropic") -> None:
        self.provider_name = provider_name

    def _select_sync(self, prompt: str) -> SelectorVote:
        enc = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(enc.encode(prompt))
        readability = textstat.flesch_kincaid_grade(prompt)
        
        if any(x in prompt.lower() for x in ["programming","code","coding","debugging","data-structures","scripting","git","api","database","python","java","javascript","c++","c#","typescript","query","database"]):
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model=TOPIC_TO_MODEL["PROGRAMMING"][self.provider_name],
                rationale="Keyword match: PROGRAMMING-related",
            )
        if any(x in prompt.lower() for x in ["finance","financial-analysis","investment","stocks","trading","banking","fintech","personal-finance","cryptocurrency","blockchain","accounting","economics","risk-management","portfolio-management","financial-modeling","budgeting","loans","insurance","taxation","wealth-management"]):
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model=TOPIC_TO_MODEL["FINANCE"][self.provider_name],
                rationale="Keyword match: FINANCE-related",
            )
        if any(x in prompt.lower() for x in ["technology","tech","innovation","gadgets","blockchain","iot","cybersecurity","cloud-computing","hardware","robotics","virtual-reality","augmented-reality","5g","automation","digital-transformation","mobile","phone","laptop"]):
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model=TOPIC_TO_MODEL["TECHNOLOGY"][self.provider_name],
                rationale="Keyword match:TECHNOLOGY-related",
            )
        if any(x in prompt.lower() for x in ["health","wellness","fitness","nutrition","mental-health","healthcare","medicine","public-health","exercise","disease-prevention","medical-research","health-tech","patient-care","health-education","chronic-disease","health-policy","telemedicine","nutritionist","mental-wellbeing","health-awareness"]):
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model=TOPIC_TO_MODEL["HEALTH"][self.provider_name],
                rationale="Keyword match:HEALTH-related",
            )
        if any(x in prompt.lower() for x in ["entertainment","movie","music","tv-shows","gaming","anime","manga","awards","comedy","pop-culture","serie","film"]):
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model=TOPIC_TO_MODEL["ENTERTAINMENT"][self.provider_name],
                rationale="Keyword match:ENTERTAINMENT-related",
            )
        if num_tokens < 20 and readability < 6:
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model=TOPIC_TO_MODEL["SIMPLE"][self.provider_name],
                rationale="Short/simple prompt",
            )
        if num_tokens > 20 and readability > 6:
            return SelectorVote(
                selector_name=self.__class__.__name__,
                model=TOPIC_TO_MODEL["COMPLEX"][self.provider_name],
                rationale="COMPLEX complexity",
            )
        return SelectorVote(
            selector_name=self.__class__.__name__,
            model=TOPIC_TO_MODEL["GENERAL"][self.provider_name],
            rationale="GENERAL prompt",
        )

    def select_model(self, prompt: str) -> SelectorVote:
        try:
            return self._select_sync(prompt)
        except Exception as exc:  # pragma: no cover - protective
            logger.exception("HeuristicsSelector failed")
            raise SelectorError(str(exc)) from exc
