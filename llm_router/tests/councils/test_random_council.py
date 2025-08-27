import pytest
from llm_router.councils.random import RandomCouncil
from llm_router.selectors.heuristics import HeuristicsSelector
from llm_router.schemas.council_schemas import CouncilDecision

@pytest.fixture
def random_council():
    selector = HeuristicsSelector()
    return RandomCouncil(selectors=[selector])

def test_random_council_creation(random_council):
    """Test RandomCouncil initialization"""
    assert len(random_council.selectors) == 1
    assert isinstance(random_council.selectors[0], HeuristicsSelector)

def test_random_council_decision(random_council):
    """Test RandomCouncil decision making"""
    prompt = "What is the capital of France?"
    decision = random_council.decide(prompt)

    assert isinstance(decision, CouncilDecision)
    assert decision.final_model is not None
    assert len(decision.votes) == 1
    assert len(decision.weighted_results) >= 1
    assert "prompt_length" in decision.metadata
