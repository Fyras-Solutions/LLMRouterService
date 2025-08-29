import pytest
import responses

from llm_router.selectors.heuristics import HeuristicsSelector
from llm_router.exceptions.exceptions import SelectorError
from llm_router.schemas.config import TOPIC_TO_MODEL

# Test HFZeroShotSelector
@pytest.fixture
def hf_selector(monkeypatch):
    monkeypatch.setenv("HF_API_KEY", "test")
    monkeypatch.setenv("PROMPTLAYER_API_KEY", "test")
    from llm_router.selectors.classifier import HFZeroShotSelector

    return HFZeroShotSelector()

@responses.activate
def test_hf_selector_successful_classification(hf_selector):
    # Mock successful API response
    responses.add(
        responses.POST,
        "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
        json={"labels": ["code", "general", "math"], "scores": [0.8, 0.1, 0.1]},
        status=200,
    )

    result = hf_selector.select_model(
        "Write a Python function with a Flask API to sort a list in reverse order."
    )
    assert result.model == TOPIC_TO_MODEL["code"]["anthropic"]
    assert "Zero-shot classified" in result.rationale

@responses.activate
def test_hf_selector_api_error(hf_selector):
    # Mock API error
    responses.add(
        responses.POST,
        "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
        status=500
    )

    result = hf_selector.select_model("test prompt")
    assert result.model == TOPIC_TO_MODEL["simple"]["anthropic"]
    assert "HF API error status 500" in result.rationale

@responses.activate
def test_hf_selector_invalid_json(hf_selector):
    # Mock invalid JSON response
    responses.add(
        responses.POST,
        "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
        body="invalid json",
        status=200
    )

    result = hf_selector.select_model("test prompt")
    assert result.model == TOPIC_TO_MODEL["simple"]["anthropic"]
    assert "Invalid JSON from HF API" in result.rationale

# Test HeuristicsSelector
@pytest.fixture
def heuristics_selector():
    return HeuristicsSelector()

def test_heuristics_code_related(heuristics_selector):
    result = heuristics_selector.select_model("Write a Python code to sort a list")
    assert result.model == TOPIC_TO_MODEL["code"]["anthropic"]
    assert "code-related" in result.rationale

def test_heuristics_math_related(heuristics_selector):
    result = heuristics_selector.select_model("Solve this integral equation")
    assert result.model == TOPIC_TO_MODEL["math"]["anthropic"]
    assert "math-related" in result.rationale

def test_heuristics_short_simple(heuristics_selector):
    result = heuristics_selector.select_model("Hi there")
    assert result.model == TOPIC_TO_MODEL["simple"]["anthropic"]
    assert "Short/simple prompt" in result.rationale

def test_heuristics_medium_complexity(heuristics_selector):
    medium_prompt = "Write a summary of this paragraph that talks about various topics"
    result = heuristics_selector.select_model(medium_prompt)
    assert result.model == TOPIC_TO_MODEL["general"]["anthropic"]
    assert "Medium complexity" in result.rationale

def test_heuristics_long_complex(heuristics_selector):
    long_prompt = " ".join(["complex"] * 100)
    result = heuristics_selector.select_model(long_prompt)
    assert result.model == TOPIC_TO_MODEL["complex"]["anthropic"]
    assert "Long/complex prompt" in result.rationale

# Error handling test for HeuristicsSelector
def test_heuristics_selector_error_handling(heuristics_selector, monkeypatch):
    def mock_error(*args, **kwargs):
        raise Exception("Test error")

    monkeypatch.setattr(heuristics_selector, "_select_sync", mock_error)
    with pytest.raises(SelectorError):
        heuristics_selector.select_model("test")
