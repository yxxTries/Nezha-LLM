
from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass
class ASRConfig:
    model_name: str = os.getenv("ASR_MODEL_NAME", "tiny")
    device: str = os.getenv("ASR_DEVICE", "cpu")
    compute_type: str = os.getenv("ASR_COMPUTE_TYPE", "int8")
    sample_rate: int = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
    beam_size: int = int(os.getenv("ASR_BEAM_SIZE", "5"))
    vad_filter: bool = os.getenv("ASR_VAD_FILTER", "1") == "1"
    language: str | None = os.getenv("ASR_LANGUAGE") or None

config = ASRConfig()
