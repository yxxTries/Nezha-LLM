from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple


logger = logging.getLogger(__name__)


class QwenModelLoader:
    """Loads a local Qwen model and tokenizer from disk.

    This loader expects the model to be available in a directory that is compatible
    with `transformers.AutoTokenizer.from_pretrained` and
    `transformers.AutoModelForCausalLM.from_pretrained`.
    """

    def __init__(
        self,
        model_dir: Path,
        device: str = "cpu",
        *,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        cpu_dtype: str = "float32",
        cuda_dtype: str = "float16",
    ) -> None:
        self._model_dir = Path(model_dir)
        self._device = device
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._cpu_dtype = cpu_dtype
        self._cuda_dtype = cuda_dtype

    def load(self) -> Tuple[Any, Any]:
        """Load the tokenizer and model.

        Returns:
            A `(tokenizer, model)` tuple.

        Raises:
            RuntimeError: If the `transformers` library is not installed.
            FileNotFoundError: If the model directory does not exist.
        """
        if not self._model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {self._model_dir}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover - exercised only in real runtime
            raise RuntimeError(
                "The 'transformers' package is required to load the Qwen model. "
                "Install it via 'pip install transformers'."
            ) from exc

        # Resolve device selection
        resolved_device = self._device
        if self._device == "auto":
            try:
                import torch
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                resolved_device = "cpu"
            print(f"[LLM] Resolved device: {resolved_device}")

        torch_dtype = None
        try:
            import torch
            if resolved_device == "cuda":
                torch_dtype = getattr(torch, self._cuda_dtype, torch.float16)
            else:
                torch_dtype = getattr(torch, self._cpu_dtype, torch.float32)
        except ImportError:
            torch = None  # type: ignore

        # Prepare quantization flags
        use_4bit = self._load_in_4bit and resolved_device == "cuda"
        use_8bit = self._load_in_8bit and not use_4bit and resolved_device == "cuda"

        if (self._load_in_4bit or self._load_in_8bit) and resolved_device != "cuda":
            logger.warning("Quantization requested but CUDA not available; falling back to CPU fp32/fp16.")

        model_kwargs = {"trust_remote_code": True}

        if use_4bit or use_8bit:
            try:
                import bitsandbytes  # type: ignore[import-not-found]
                model_kwargs.update({
                    "device_map": "auto",
                    "load_in_4bit": use_4bit,
                    "load_in_8bit": use_8bit,
                })
                if use_4bit and torch is not None:
                    model_kwargs["bnb_4bit_compute_dtype"] = torch_dtype or torch.float16
            except ImportError:
                logger.warning("bitsandbytes not installed; loading without quantization.")
                use_4bit = use_8bit = False

        if torch_dtype and not (use_4bit or use_8bit):
            model_kwargs["torch_dtype"] = torch_dtype

        logger.info("Loading Qwen model from %s", self._model_dir)
        tokenizer = AutoTokenizer.from_pretrained(self._model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self._model_dir, **model_kwargs)

        if not (use_4bit or use_8bit) and hasattr(model, "to") and resolved_device:
            logger.info("Moving model to device: %s", resolved_device)
            model = model.to(resolved_device)

        logger.info(
            "Qwen model loaded successfully on %s%s",
            resolved_device,
            " with quantization" if (use_4bit or use_8bit) else "",
        )
        return tokenizer, model
