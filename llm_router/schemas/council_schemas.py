from typing import Optional, List, Dict, Any
from pydantic import BaseModel


# ---------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------
class SelectorVote(BaseModel):
    """
    A single vote cast by a Selector.
    """
    selector_name: str               # e.g. "HeuristicsSelector"
    model: str                       # the model chosen by this selector
    weight: float = 1.0              # weight applied during aggregation
    rationale: Optional[str] = None  # optional explanation for traceability


class CouncilDecision(BaseModel):
    """
    The final outcome of a Router (e.g. Hybrid Router).
    """
    final_model: str                       # the selected model
    votes: List[SelectorVote]              # raw votes from each selector
    weighted_results: Dict[str, float]     # aggregated tallies
    metadata: Dict[str, str] = {}          # traceability info (e.g. run_id, prompt length)


class LLMResponse(BaseModel):
    """
    Standardized LLM execution response.
    """
    model: str
    prompt: str
    response: str
    cost: float
    latency: float


class RouterMetadata(BaseModel):
    votes: Optional[List[Dict[str, Any]]] = None
    weighted_results: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class LLMRouterResponse(LLMResponse):
    metadata: Optional[RouterMetadata] = None
