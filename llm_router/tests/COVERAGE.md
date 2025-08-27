# Test Coverage Report

**Last Revised:** August 27th, 2025

## Overall Status
All test cases are passing except for `test_hf_selector_successful_classification` in `test_selectors.py` due to a deterministic error with the HuggingFace Zero-Shot classification model.

## Test Coverage by Component

### Selectors (`test_selectors.py`)
- **HFZeroShotSelector**
  - ✅ API Error handling
  - ✅ Invalid JSON response handling
  - ❌ Successful classification (Known issue with HF model determinism)
  - Coverage: ~90%

- **PromptLengthSelector**
  - ✅ Short prompt handling (≤ 15 words)
  - ✅ Medium prompt handling (≤ 80 words)
  - ✅ Long prompt handling (> 80 words)
  - ✅ Custom thresholds configuration
  - ✅ Error handling
  - Coverage: 100%

- **HeuristicsSelector**
  - ✅ Code-related prompts detection
  - ✅ Math-related prompts detection
  - ✅ Short/simple prompts handling
  - ✅ Medium complexity prompts
  - ✅ Long/complex prompts
  - ✅ Error handling
  - Coverage: 100%

### Councils
#### Random Council (`test_random_council.py`)
- ✅ Random selection from available models
- ✅ Error handling for empty model list
- ✅ Metadata recording
- Coverage: 100%

### Routers (`test_router.py`)
- ✅ Router initialization
- ✅ Vote aggregation
- ✅ Model selection logic
- ✅ Error handling
- Coverage: 100%

### Schemas (`test_council_schemas.py`)
- ✅ LLMResponse validation
- ✅ RouterMetadata validation
- ✅ LLMRouterResponse validation
- Coverage: 100%

### Exceptions (`test_exceptions.py`)
- ✅ LLMRouterError base class
- ✅ UsableModelForPromptError
- ✅ SelectorError
- ✅ CouncilError
- ✅ ModelExecutionError
- ✅ RouterError
- Coverage: 100%

## Known Issues and Limitations

1. **HFZeroShotSelector Classification Test**
   - Issue: The `test_hf_selector_successful_classification` test is currently failing
   - Reason: Deterministic behavior issues with the HuggingFace Zero-Shot classification model
   - Workaround: The system falls back to default models when classification fails, ensuring system stability

## Future Test Improvements

1. Consider implementing more robust mocking for the HF Zero-Shot classifier to avoid dependency on external service
2. Add more edge cases for prompt variations
3. Implement integration tests for full workflow scenarios
4. Add performance benchmarking tests for selectors and councils

## Test Dependencies
- pytest ^8.4.1
- pytest-cov ^4.1.0
- pytest-mock ^3.11.1
- responses ^0.24.1

## Running Tests
```bash
poetry run pytest -v --cov=llm_router --cov-report=term-missing
```
