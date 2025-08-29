import pytest
from pathlib import Path
from unittest.mock import patch

from llm_router.routers.router import LLMRouterService
from llm_router.councils.random import RandomCouncil
from llm_router.selectors.heuristics import HeuristicsSelector
from llm_router.schemas.council_schemas import LLMRouterResponse, RouterMetadata
from llm_router.providers import ProviderResponse

@pytest.fixture
def mock_completion():
    return ProviderResponse(text="Test response", prompt_tokens=10, completion_tokens=20)

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

def test_router_service_invoke(router_service, mock_completion):
    """Test LLMRouterService invoke method with mocked LLM calls"""
    with patch.object(router_service.provider, "complete", return_value=mock_completion), \
        patch.object(router_service.provider, "get_cost", return_value=0.0):
        prompt = "What is the capital of France?"
        response = router_service.invoke(prompt)

    # Verify response
    assert isinstance(response, LLMRouterResponse)
    assert response.prompt == prompt
    assert response.response == "Test response"
    assert response.cost == 0.0
    assert response.latency >= 0
    assert isinstance(response.metadata, RouterMetadata)

def test_router_service_error_handling(router_service):
    """Test LLMRouterService error handling"""
    with patch.object(router_service.provider, "complete", side_effect=Exception("API Error")):
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

def test_router_metadata_handling(router_service, mock_completion):
    """Test metadata handling in router responses"""
    with patch.object(router_service.provider, "complete", return_value=mock_completion), \
        patch.object(router_service.provider, "get_cost", return_value=0.0):
        response = router_service.invoke("Test prompt")

    # Verify metadata structure
    assert response.metadata is not None
    assert response.metadata.votes is not None
    assert response.metadata.weighted_results is not None
    assert response.metadata.tags is not None
    assert "council-router" in response.metadata.tags
