# Nezha-LLM – Copilot Instructions

- **What it is**: Local voice→text→LLM stack. FastAPI (`api/`) wraps ASR (`asr/` faster-whisper + ffmpeg) and LLM (`llm/` Qwen) with a static JS UI (`ui/`). No persistence; temp audio files are deleted after each request.

- **How to run locally**
  - `pip install -r requirements.txt` (use `.venv` per `start.bat`).
  - API: `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload` (or run `start.bat` to launch API + `python -m http.server 8080` for UI and open `http://localhost:8080`).
  - Swagger/Redoc at `/docs` and `/redoc` once the API is up.

- **Configuration knobs**
  - API settings via env prefix `NEZHA_LLM_API_` (`host`, `port`, CORS, `log_level`), defined in `api/config.py`.
  - ASR env vars in `asr/config.py`: `ASR_MODEL_NAME` (default `tiny`), `ASR_DEVICE` (`cpu`/`cuda`), `ASR_COMPUTE_TYPE` (`int8`), `ASR_SAMPLE_RATE` (16000), `ASR_BEAM_SIZE`, `ASR_VAD_FILTER`, `ASR_LANGUAGE`.
  - LLM defaults live in `llm/config.py` (`LLMConfig`): model dir points to `models/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540...`, `device="auto"` (prefers CUDA), `load_in_4bit=True`, temperature/top_p/max_new_tokens defaults, prompt templates (`PROMPT_TEMPLATES`, default key `default`). Adjust by passing a custom `LLMConfig` when constructing `LLMService`.

- **Service lifecycle / DI**
  - `api.deps` caches singleton `ASRService` and `LLMService`; startup event in `api/main.py` preloads them. In tests, override with `app.dependency_overrides` (see `tests/test_api_routes.py`).

- **API contract** (`api/routes.py`)
  - `POST /api/llm` expects JSON `{ message, max_new_tokens?, temperature? }`; wraps message into `LLMRequest` and returns `LLMReply` `{ text, timestamp }`. Exceptions become `HTTP 500` with logged stack.
  - `POST /api/asr-llm` accepts multipart `audio` file, writes to temp file with original suffix, normalizes to WAV, runs ASR then LLM. Guard: if transcript empty or duration <0.3s, returns friendly message and skips LLM. Temp file always removed in `finally`.

- **ASR pipeline**
  - `ASRService.transcribe_file` uses `ffmpeg` CLI via `asr/ffmpeg_io.py` to convert any input to mono WAV at `config.sample_rate` (default 16 kHz), then calls `FasterWhisperBackend.transcribe`.
  - Backend defined in `asr/faster_whisper_backend.py` wraps `faster_whisper.WhisperModel`, applying `beam_size`, `vad_filter`, optional `language/prompt`. Segments filtered for non-empty text; whitespace collapsed.
  - `transcribe_audio` supports NumPy arrays (used by interactive CLI) and handles resampling if input sample rate != 16k.

- **LLM pipeline**
  - `LLMService.generate` builds a prompt from `PROMPT_TEMPLATES` and `instruction/system` strings, tokenizes, moves tensors to the model device, and calls `model.generate` with `max_new_tokens`, `temperature`, `top_p`, `repetition_penalty` (defaults can be overridden per `LLMRequest`).
  - Loader (`llm/model_loader.py`) resolves device (`auto` → CUDA if available), optional 4/8-bit quantization via bitsandbytes, picks dtype per CPU/CUDA, and moves model to device when not using quantization. Requires `transformers` (and optionally `bitsandbytes`, `torch` with CUDA for quantized loads).
  - Post-processing strips the prompt prefix if the model echoes it, returning clean text.

- **Frontend expectations** (`ui/app.js`)
  - Endpoints are hardcoded: `TEXT_ENDPOINT=http://localhost:8000/api/llm`, `AUDIO_ENDPOINT=http://localhost:8000/api/asr-llm`.
  - Voice flow uses `MediaRecorder`, uploads `audio` FormData field named `"audio"`, and updates the user bubble with `transcript` if returned. Timestamps formatted client-side.

- **Developer utilities & tests**
  - Fast API smoke test with dependency overrides: `tests/test_api_routes.py` (no models needed).
  - ASR runtime check against a real sample in `audiosample/`: `tests/test_asr_service.py` (needs ffmpeg + model download).
  - LLM unit tests with dummy tokenizer/model in `tests/test_llm_service.py`; also includes an optional runtime test loading the real Qwen snapshot (CPU by default).
  - CLI demos: `tests/voice_llm_cli.py` processes `audiosample/recording.wav`; `tests/test_voice_conversation_cli.py` starts an interactive record→ASR→LLM loop using in-memory audio.

- **Conventions / guardrails**
  - Keep ffmpeg normalization before ASR; failing to normalize raises `AudioPreprocessingError`.
  - Preserve temp-file cleanup in `/api/asr-llm`; ensure new code keeps `finally` unlink behavior.
  - Maintain the short-duration guard to avoid empty-audio LLM calls.
  - When adding endpoints, mirror the dependency-injection pattern (`Depends(get_*_service)`) so services stay singleton-cached.
  - When changing the model path or device logic, update `LLMConfig` defaults and `QwenModelLoader` consistently so CLI/UI/users stay aligned.
