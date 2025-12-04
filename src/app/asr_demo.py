# src/app/main_asr_demo.py

import os
import re
import time
import threading
import keyboard
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

from asr.model import ASRModel
from local_llm_test import load_model as load_llm, run_inference

# Recording settings
SAMPLE_RATE = 16000  # Whisper expects 16kHz
MAX_DURATION = 15    # Max recording length in seconds
CHANNELS = 1
AUDIO_DIR = Path(__file__).parent / "audio"

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.start_time = None
        
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.start_time = time.time()
        print("\n[REC] Recording started... (Press SPACE to stop, max 15s)")
        
        def callback(indata, frames, time_info, status):
            if self.recording:
                self.audio_data.append(indata.copy())
                
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=callback,
            blocksize=1024
        )
        self.stream.start()
        
        # Auto-stop thread after MAX_DURATION
        def auto_stop():
            while self.recording:
                elapsed = time.time() - self.start_time
                if elapsed >= MAX_DURATION:
                    print(f"\n[REC] Max duration ({MAX_DURATION}s) reached.")
                    self.stop_recording()
                    break
                time.sleep(0.1)
        
        self.timer_thread = threading.Thread(target=auto_stop, daemon=True)
        self.timer_thread.start()
        
    def stop_recording(self):
        if not self.recording:
            return None
            
        self.recording = False
        self.stream.stop()
        self.stream.close()
        
        if not self.audio_data:
            print("[REC] No audio recorded.")
            return None
            
        # Combine all audio chunks
        audio = np.concatenate(self.audio_data, axis=0)
        duration = len(audio) / SAMPLE_RATE
        print(f"[REC] Recording stopped. Duration: {duration:.1f}s")
        
        # Save to file
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = AUDIO_DIR / "recording.wav"
        
        # Convert to int16 for WAV file
        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(str(audio_path), SAMPLE_RATE, audio_int16)
        
        return str(audio_path)


def main():
    print("=" * 50)
    print("       ASR Demo - Voice to Text")
    print("=" * 50)
    print("\nCommands:")
    print("  SPACE  - Start/Stop recording")
    print("  Q      - Quit")
    print("-" * 50)
    
    # Pre-load ASR model for faster inference
    print("\n[INFO] Loading ASR model...")
    model_size = "small.en"  # Use smaller model for speed
    asr = ASRModel(model_size=model_size)
    
    # Pre-load LLM for faster inference
    print("[INFO] Loading LLM...")
    llm_tokenizer, llm_model = load_llm()
    print("[INFO] All models loaded. Ready to record!\n")
    
    recorder = AudioRecorder()
    
    while True:
        if not recorder.recording:
            print("Press SPACE to start recording, Q to quit...")
            
        event = keyboard.read_event(suppress=True)
        
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'space':
                if not recorder.recording:
                    recorder.start_recording()
                else:
                    audio_path = recorder.stop_recording()
                    
                    if audio_path:
                        # Transcribe
                        print("\n[ASR] Transcribing...")
                        result = asr.transcribe_file(audio_path)
                        
                        # Clean text
                        text = result["text"].strip()
                        text = " ".join(text.split())
                        text = re.sub(r'([.!?])\1+', r'\1', text)
                        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
                        
                        # Print result
                        print("\n" + "=" * 40)
                        print("TRANSCRIPTION:")
                        print(text)
                        print("=" * 40 + "\n")
                        
                        # Save last output to output.txt
                        output_path = Path(__file__).parent / "output.txt"
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        
                        # Run LLM inference
                        print("[LLM] Generating response...")
                        llm_response = run_inference(text, llm_tokenizer, llm_model)
                        print("\n" + "=" * 40)
                        print("LLM RESPONSE:")
                        print(llm_response)
                        print("=" * 40 + "\n")
                        
            elif event.name == 'q' and not recorder.recording:
                print("\n[INFO] Goodbye!")
                break


if __name__ == "__main__":
    main()
