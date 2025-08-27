import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from llm_router.routers.router import LLMRouterService
from llm_router.councils.random import RandomCouncil
from llm_router.selectors.heuristics import HeuristicsSelector
from llm_router.schemas.council_schemas import LLMRouterResponse, RouterMetadata

@pytest.fixture
def mock_completion():
    mock_resp = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": Mock(prompt_tokens=10, completion_tokens=20)
    }
    return mock_resp

@pytest.fixture
def env_file(tmp_path) -> Path:
    file = tmp_path / ".env"
    file.write_text("HF_API_KEY=test\nPROMPTLAYER_API_KEY=test\n")
    return file


@pytest.fixture
def router_service(env_file):
    selector = HeuristicsSelector()
    council = RandomCouncil(selectors=[selector])
    return LLMRouterService(council=council, env_path=env_file)

@patch('llm_router.routers.router.completion')
@patch('llm_router.routers.router.cost_per_token')
def test_router_service_invoke(mock_cost, mock_completion_func, router_service, mock_completion):
    """Test LLMRouterService invoke method with mocked LLM calls"""
    # Setup mocks
    mock_completion_func.return_value = mock_completion
    mock_cost.return_value = 0.0

    # Test invocation
    prompt = "What is the capital of France?"
    response = router_service.invoke(prompt)

    # Verify response
    assert isinstance(response, LLMRouterResponse)
    assert response.prompt == prompt
    assert response.response == "Test response"
    assert response.cost == 0.0
    assert response.latency >= 0
    assert isinstance(response.metadata, RouterMetadata)

@patch('llm_router.routers.router.completion')
def test_router_service_error_handling(mock_completion_func, router_service):
    """Test LLMRouterService error handling"""
    mock_completion_func.side_effect = Exception("API Error")

    with pytest.raises(Exception):
        router_service.invoke("Test prompt")

def test_router_service_initialization(env_file):
    """Test LLMRouterService initialization with different configurations"""
    # Test with custom API key
    selector = HeuristicsSelector()
    council = RandomCouncil(selectors=[selector])
    service = LLMRouterService(council=council, api_key="test_key", env_path=env_file)
    assert service.council == council

    # Test without API key (should use environment variable)
    service_no_key = LLMRouterService(council=council, env_path=env_file)
    assert service_no_key.council == council

@patch('llm_router.routers.router.completion')
@patch('llm_router.routers.router.cost_per_token')
def test_router_metadata_handling(mock_cost, mock_completion_func, router_service, mock_completion):
    """Test metadata handling in router responses"""
    # Setup mocks
    mock_completion_func.return_value = mock_completion
    mock_cost.return_value = 0.0

    response = router_service.invoke("Test prompt")

    # Verify metadata structure
    assert response.metadata is not None
    assert response.metadata.votes is not None
    assert response.metadata.weighted_results is not None
    assert response.metadata.tags is not None
    assert "council-router" in response.metadata.tags
