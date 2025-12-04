# AudioGPT (Quick README)

This repo contains a simple audio→text pipeline using Whisper for ASR and a local Hugging Face model for follow‑up text processing.


# Look at requirments for python module requirments.

# After logging into huggingface use this python command below to download the model I am uing.
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2-0.5B-Instruct', cache_dir='models'))"

