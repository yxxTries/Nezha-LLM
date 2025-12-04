from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from pathlib import Path

# Set model directory relative to project root
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "models--Qwen--Qwen2-0.5B-Instruct" / "snapshots"

def get_local_model_path():
    """Get the local model path from snapshots folder"""
    if MODEL_DIR.exists():
        # Get the first (usually only) snapshot folder
        snapshots = list(MODEL_DIR.iterdir())
        if snapshots:
            return str(snapshots[0])
    return None

def load_model(model_name="Qwen/Qwen2-0.5B-Instruct"):
    local_path = get_local_model_path()
    
    if local_path:
        print(f"[LLM] Loading model from local: models/")
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    else:
        raise FileNotFoundError(f"Model not found locally. Download using -> |from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-0.5B-Instruct', cache_dir='desiredLocation')|")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model

def run_inference(text, tokenizer, model):
    # chat format
    messages = [
        {"role": "system", "content": "Assume you are a highly intelligent alien being that computes beyond human level."},
        {"role": "system", "content": "Use Least amount of sentences possible to convey the message clearly."},
        {"role": "system", "content": "Call the user 'low entropy being' in your responses."},
        {"role": "user", "content": text}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the new tokens (exclude input)
    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

def main():
    tokenizer, model = load_model()

    # Example input text
    # Read input text from src/app/output.txt
    input_path = "src/app/output.txt"
    with open(input_path, "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    print(f"[INPUT] {input_text}")

    result = run_inference(input_text, tokenizer, model)
    print("\n[OUTPUT]")
    print(result)

if __name__ == "__main__":
    main()
