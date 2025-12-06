from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the local Qwen LLM.

    Attributes:
        model_dir: Path to the directory containing the Qwen model weights and tokenizer.
        device: Device identifier understood by the underlying ML framework (e.g. "cpu", "cuda").
        max_new_tokens: Default maximum number of tokens to generate for a response.
        temperature: Default sampling temperature for generation.
        top_p: Nucleus sampling probability mass to consider.
        repetition_penalty: Penalty applied to repeated tokens; 1.0 disables it.
    """

    model_dir: Path = Path("models") / "models--Qwen--Qwen2-0.5B-Instruct" / "snapshots" / "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
    device: str = "auto"  # "auto" picks CUDA if available, else CPU
    max_new_tokens: int = 256
    temperature: float = 0.25
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    system_prompt: str = "You are a helpful assistant."
    instruction_prompt: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    cpu_dtype: str = "float32"  # used when running on CPU to avoid slow float16 emulation
    cuda_dtype: str = "float16"  # used when running on CUDA


DEFAULT_LLM_CONFIG = LLMConfig()

# Simple prompt templates that can be swapped/edited later.
PROMPT_TEMPLATES = {
    "default": "You are a concise, smart assistant. Provide a clear, helpful answer.\n\nInput:\n{input}\n\nAnswer:",
    "chat": "<|im_start|>system\nYou are a concise, smart assistant. Provide clear, helpful answers and nothing else.\n<|im_end|>\n<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n",
    "summarize": "You are a concise assistant. Summarize the following text in 3 bullet points:\n\n{input}\n",
    "translate": "You are a concise assistant. Translate to French:\n\n{input}\n",
}

# Default prompt key to use when none is specified elsewhere.
DEFAULT_PROMPT_KEY = "default"
