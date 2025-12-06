from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass

import builtins

from llm.config import LLMConfig
from llm.service import LLMService
from llm.types import LLMRequest, LLMResponse



class DummyTokenizer:
    """Minimal tokenizer stub for testing LLMService in isolation."""

    def __init__(self) -> None:
        self.eos_token_id = 0
        self.last_input: str | None = None

    def __call__(self, text: str, return_tensors: str | None = None) -> dict:
        self.last_input = text
        # The concrete values do not matter for our tests, as long as they
        # can be passed through to the model stub.
        return {"input_ids": [1, 2, 3]}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        # For testing we return a fixed string independent of token_ids.
        return "dummy_response"


class DummyModel:
    """Minimal model stub capturing generation parameters."""

    def __init__(self) -> None:
        self.generate_called_with: dict | None = None

    def generate(self, **kwargs):
        self.generate_called_with = kwargs
        # Return a dummy sequence of token ids. Only structure matters.
        return [[1, 2, 3, 4]]


def test_generate_returns_llmresponse_with_text() -> None:
    config = LLMConfig()
    tokenizer = DummyTokenizer()
    model = DummyModel()
    service = LLMService(config=config, model=model, tokenizer=tokenizer)

    request = LLMRequest(text="Hello world")
    response = service.generate(request)

    assert isinstance(response, LLMResponse)
    assert response.text == "dummy_response"
    assert tokenizer.last_input == "Hello world"


def test_generate_uses_overrides_for_parameters() -> None:
    config = LLMConfig(max_new_tokens=16, temperature=0.1)
    tokenizer = DummyTokenizer()
    model = DummyModel()
    service = LLMService(config=config, model=model, tokenizer=tokenizer)

    request = LLMRequest(text="Hello", max_new_tokens=42, temperature=0.5)
    _ = service.generate(request)

    assert model.generate_called_with is not None
    assert model.generate_called_with["max_new_tokens"] == 42
    assert model.generate_called_with["temperature"] == 0.5


def test_llm_runtime_with_real_model() -> None:
    """Runtime test that loads the actual model and generates a response.
    
    This test uses the real Qwen model to verify end-to-end functionality.
    It outputs the results for manual inspection.
    """
    from pathlib import Path
    
    # Use the actual model path from the workspace
    model_path = PROJECT_ROOT / "models" / "models--Qwen--Qwen2-0.5B-Instruct" / "snapshots" / "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
    
    config = LLMConfig(
        model_dir=model_path,
        device="cpu",
        max_new_tokens=50,
        temperature=0.7
    )
    
    # Initialize service with real model (no mocks)
    print("\n" + "=" * 60)
    print("RUNTIME LLM TEST - Loading model...")
    service = LLMService(config=config)
    print("Model loaded successfully!")
    
    # Test with a basic prompt
    test_prompt = "What is 2 + 2?"
    print(f"\nPrompt: {test_prompt}")
    
    from llm.types import LLMRequest
    request = LLMRequest(text=test_prompt, max_new_tokens=50, temperature=0.7)
    response = service.generate(request)
    
    print(f"Response: {response.text}")
    print("=" * 60 + "\n")
    
    # Basic assertions
    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0, "Response should not be empty"
    
    print("Runtime LLM test passed!")


if __name__ == "__main__":
    # Run the runtime test directly
    print("Running runtime LLM test...")
    test_llm_runtime_with_real_model()

