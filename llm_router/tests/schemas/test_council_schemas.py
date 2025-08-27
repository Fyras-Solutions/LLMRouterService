import pytest
from pydantic import ValidationError
from llm_router.schemas.council_schemas import (
    LLMResponse,
    RouterMetadata,
    LLMRouterResponse,
    SelectorVote,
    CouncilDecision
)

def test_llm_response_creation():
    """Test LLMResponse creation with valid data"""
    response = LLMResponse(
        model="gpt-3.5-turbo",
        prompt="test prompt",
        response="test response",
        cost=0.001,
        latency=0.5
    )
    assert response.model == "gpt-3.5-turbo"
    assert response.prompt == "test prompt"
    assert response.response == "test response"
    assert response.cost == 0.001
    assert response.latency == 0.5

def test_router_metadata_creation():
    """Test RouterMetadata creation with valid data"""
    metadata = RouterMetadata(
        votes=[{"selector": "test", "vote": 1}],
        weighted_results={"model1": 0.7, "model2": 0.3},
        tags=["test", "router"]
    )
    assert len(metadata.votes) == 1
    assert len(metadata.weighted_results) == 2
    assert len(metadata.tags) == 2

def test_llm_router_response_creation():
    """Test LLMRouterResponse creation with metadata"""
    metadata = RouterMetadata(
        votes=[{"selector": "test", "vote": 1}],
        weighted_results={"model1": 0.7, "model2": 0.3},
        tags=["test"]
    )
    response = LLMRouterResponse(
        model="gpt-3.5-turbo",
        prompt="test prompt",
        response="test response",
        cost=0.001,
        latency=0.5,
        metadata=metadata
    )
    assert response.metadata.tags == ["test"]
    assert response.model == "gpt-3.5-turbo"

def test_selector_vote_creation():
    """Test SelectorVote creation and validation"""
    vote = SelectorVote(
        selector_name="TestSelector",
        model="gpt-3.5-turbo",
        weight=1.0,
        rationale="Test rationale"
    )
    assert vote.selector_name == "TestSelector"
    assert vote.weight == 1.0

def test_council_decision_creation():
    """Test CouncilDecision creation and validation"""
    vote = SelectorVote(
        selector_name="TestSelector",
        model="gpt-3.5-turbo",
        weight=1.0
    )
    decision = CouncilDecision(
        final_model="gpt-3.5-turbo",
        votes=[vote],
        weighted_results={"gpt-3.5-turbo": 1.0},
        metadata={"test": "metadata"}
    )
    assert decision.final_model == "gpt-3.5-turbo"
    assert len(decision.votes) == 1
    assert decision.metadata["test"] == "metadata"

def test_invalid_llm_response():
    # response should be a string, cost should be a float
    with pytest.raises(ValidationError):
        LLMResponse(
            model="ollama/phi3:latest",
            prompt="Test prompt",
            response=123,  # Invalid: should be str
            cost="free",   # Invalid: should be float
            latency=0.1
        )
