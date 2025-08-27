import pytest
from llm_router.exceptions.exceptions import (
    LLMRouterError,
    UsableModelForPromptError,
    SelectorError,
    CouncilError,
    ModelExecutionError,
    RouterError
)

def test_llm_router_error_base():
    """Test base LLMRouterError"""
    with pytest.raises(LLMRouterError) as exc_info:
        raise LLMRouterError("Base error")
    assert str(exc_info.value) == "Base error"
    assert isinstance(exc_info.value, Exception)

def test_usable_model_for_prompt_error():
    """Test UsableModelForPromptError"""
    prompt = "Test prompt"
    with pytest.raises(UsableModelForPromptError) as exc_info:
        raise UsableModelForPromptError(f"No model available for prompt: {prompt}")
    assert "No model available for prompt" in str(exc_info.value)
    assert isinstance(exc_info.value, LLMRouterError)

def test_selector_error():
    """Test SelectorError"""
    selector_name = "TestSelector"
    with pytest.raises(SelectorError) as exc_info:
        raise SelectorError(f"Selector {selector_name} failed")
    assert "Selector TestSelector failed" in str(exc_info.value)
    assert isinstance(exc_info.value, LLMRouterError)

def test_council_error():
    """Test CouncilError"""
    with pytest.raises(CouncilError) as exc_info:
        raise CouncilError("Council failed to reach consensus")
    assert "Council failed to reach consensus" in str(exc_info.value)
    assert isinstance(exc_info.value, LLMRouterError)

def test_model_execution_error():
    """Test ModelExecutionError"""
    model_name = "gpt-4"
    with pytest.raises(ModelExecutionError) as exc_info:
        raise ModelExecutionError(f"Failed to execute model: {model_name}")
    assert "Failed to execute model: gpt-4" in str(exc_info.value)
    assert isinstance(exc_info.value, LLMRouterError)

def test_router_error():
    """Test RouterError"""
    with pytest.raises(RouterError) as exc_info:
        raise RouterError("Failed to aggregate votes")
    assert "Failed to aggregate votes" in str(exc_info.value)
    assert isinstance(exc_info.value, LLMRouterError)

def test_exception_inheritance():
    """Test exception inheritance hierarchy"""
    # All custom exceptions should inherit from LLMRouterError
    assert issubclass(UsableModelForPromptError, LLMRouterError)
    assert issubclass(SelectorError, LLMRouterError)
    assert issubclass(CouncilError, LLMRouterError)
    assert issubclass(ModelExecutionError, LLMRouterError)
    assert issubclass(RouterError, LLMRouterError)

    # LLMRouterError should inherit from Exception
    assert issubclass(LLMRouterError, Exception)

def test_exception_messages():
    """Test exception message formatting"""
    # Test with different message formats
    try:
        raise ModelExecutionError("Test error", model="gpt-4", error_code=500)
    except ModelExecutionError as e:
        assert "Test error" in str(e)

    try:
        raise SelectorError("Failed to compute score", selector="TestSelector")
    except SelectorError as e:
        assert "Failed to compute score" in str(e)
