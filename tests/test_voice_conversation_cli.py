#!/usr/bin/env python
from __future__ import annotations

import sys
import io
import logging
import threading
from pathlib import Path
from typing import Optional, List, Tuple

import sounddevice as sd
import numpy as np

# Add project root so imports work when running test manually
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from asr.service import ASRService
from asr.types import ASRResult
from llm.config import LLMConfig
from llm.service import LLMService
from llm.types import LLMRequest


logger = logging.getLogger(__name__)


def setup_logging(verbosity: int) -> None:
    """Configure root logging based on verbosity level."""
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class AudioRecorder:
    """Records audio with manual start/stop control."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_frames: List[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream."""
        if status:
            logger.warning("Audio status: %s", status)
        if self.is_recording:
            self.audio_frames.append(indata.copy())
    
    def start_recording(self) -> None:
        """Start recording audio."""
        self.audio_frames = []
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()
        logger.info("Recording started.")
    
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return audio data as numpy array."""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        logger.info("Recording stopped.")
        
        # Combine all audio frames
        if self.audio_frames:
            audio = np.concatenate(self.audio_frames, axis=0).flatten()
        else:
            audio = np.array([], dtype="float32")
        
        return audio


def run_pipeline_from_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    asr_service: ASRService,
    llm_service: LLMService,
) -> Tuple[ASRResult, str]:
    """
    Run the end-to-end pipeline with in-memory audio data.
    """
    logger.info("Starting ASR transcription from audio data")
    
    # faster_whisper can accept numpy array directly
    asr_result = asr_service.backend.transcribe_audio(audio_data, sample_rate)
    
    logger.info("Transcription completed. Detected duration: %.2fs", asr_result.duration or 0)
    logger.debug("Transcribed text: %s", asr_result.text)

    logger.info("Sending transcription to LLM.")
    llm_output = llm_service.generate(LLMRequest(text=asr_result.text))
    logger.info("LLM generation completed.")

    return asr_result, llm_output.text


def start_conversation() -> None:
    """
    Loop:
      1. User presses ENTER → start recording.
      2. User presses ENTER again → stop recording.
      3. Send audio to ASR.
      4. Send transcription to LLM.
      5. Print LLM response.
    """
    setup_logging(1)

    logger.info("Initializing ASR service.")
    asr_service = ASRService()

    logger.info("Initializing LLM service.")
    llm_service = LLMService(LLMConfig())

    recorder = AudioRecorder()

    print("Voice Conversation Test")
    print("Press ENTER to start recording.")
    print("Press ENTER again to stop recording.")
    print("Type 'exit' to quit.\n")

    while True:
        inp = input(">>> ").strip().lower()
        if inp == "exit":
            print("Exiting conversation.")
            break

        print("Recording... press ENTER to stop.")
        recorder.start_recording()
        
        # Wait for user to press ENTER to stop
        input()
        
        audio_data = recorder.stop_recording()

        asr_result, llm_output = run_pipeline_from_audio(
            audio_data=audio_data,
            sample_rate=recorder.sample_rate,
            asr_service=asr_service,
            llm_service=llm_service,
        )

        print("\n--- Transcription ---")
        print(asr_result.text)

        print("\n--- LLM Response ---")
        print(llm_output)
        print("\n")


def test_voice_conversation_manual():
    """
    Manual test entry point.
    This is not an automated pytest test.
    Run manually:
        python tests/test_voice_conversation_cli.py
    """
    start_conversation()


if __name__ == "__main__":
    start_conversation()
