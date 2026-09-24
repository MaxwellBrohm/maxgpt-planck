# Sequentially download the probe models (weights + tokenizer + config only).
from huggingface_hub import snapshot_download
MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "unsloth/gemma-3-270m-it",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "LiquidAI/LFM2.5-350M",
    "LiquidAI/LFM2-350M",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
]
pats = ["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt", "*.py", "README.md"]
for m in MODELS:
    p = snapshot_download(m, allow_patterns=pats)
    print("OK", m, p, flush=True)
