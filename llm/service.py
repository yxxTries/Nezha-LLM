from __future__ import annotations

import logging
from typing import Any

from .config import LLMConfig, DEFAULT_LLM_CONFIG, PROMPT_TEMPLATES, DEFAULT_PROMPT_KEY
from .model_loader import QwenModelLoader
from .types import LLMRequest, LLMResponse


logger = logging.getLogger(__name__)


class LLMService:
    """High-level service to generate responses from a local Qwen model.

    This service is intentionally minimal: it accepts plain text input (e.g. ASR output)
    and returns a plain text response suitable for feeding back into a UI layer.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self._config = config or DEFAULT_LLM_CONFIG

        if model is None or tokenizer is None:
            loader = QwenModelLoader(self._config.model_dir, device=self._config.device)
            tokenizer, model = loader.load()

        self._model = model
        self._tokenizer = tokenizer

        logger.info(
            "LLMService initialized with model_dir=%s, device=%s",
            self._config.model_dir,
            self._config.device,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for the given request.

        Args:
            request: Request object containing the input text and optional parameters.

        Returns:
            LLMResponse with the generated text.
        """
        base_text = request.text
        if not base_text:
            raise ValueError("LLMRequest.text must not be empty")

        prompt_key = request.prompt_key or DEFAULT_PROMPT_KEY
        template = PROMPT_TEMPLATES.get(prompt_key, PROMPT_TEMPLATES[DEFAULT_PROMPT_KEY])
        prompt = template.format(
            instruction=self._config.instruction_prompt,
            input=base_text,
            system=self._config.system_prompt,
            user=base_text,
        )

        max_new_tokens = request.max_new_tokens or self._config.max_new_tokens
        temperature = request.temperature if request.temperature is not None else self._config.temperature
        top_p = request.top_p if request.top_p is not None else self._config.top_p
        repetition_penalty = (
            request.repetition_penalty
            if request.repetition_penalty is not None
            else self._config.repetition_penalty
        )

        logger.debug(
            "Generating response (prompt_key=%s, max_new_tokens=%s, temperature=%s, top_p=%s, repetition_penalty=%s)",
            prompt_key,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
        )

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")

        # Generate output token ids
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
        )

        # Decode the full sequence
        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip the original prompt if the model echoes it back
        if full_text.startswith(prompt):
            generated_text = full_text[len(prompt):].strip()
        else:
            generated_text = full_text

        logger.debug("Generated response length=%d characters", len(generated_text))

        return LLMResponse(text=generated_text)

