"""Known local-LLM model sizes and quantization-aware size estimation.

Model IDs from OpenAI-compatible endpoints (Ollama, LM Studio, llama.cpp
server, vLLM...) are messy strings like "qwen2.5:7b-instruct-q4_K_M". We
extract a parameter count and quantization tag from the ID, estimate the
download size on disk, and derive the RAM needed to run it.

Copyright (c) 2026 Asdale Dynamics, LLC
"""

from __future__ import annotations

import re
from dataclasses import dataclass

GiB = 1024**3

# Bytes per parameter for each quantization level. These are approximations
# chosen to match commonly published GGUF file sizes (e.g. Llama 3 8B at
# Q4_K_M lands around 4.9 GiB).
QUANT_BYTES_PER_PARAM = {
    "q2": 0.35,
    "q3": 0.45,
    "q4": 0.60,
    "q5": 0.70,
    "q6": 0.82,
    "q8": 1.13,
    "fp16": 2.05,
    "f16": 2.05,
    "bf16": 2.05,
    "fp32": 4.10,
    "f32": 4.10,
}

# Ollama's bare ":latest" tags usually ship a Q4_K_M-style file.
DEFAULT_QUANT = "q4"

# Some popular model IDs don't state their size ("mistral-nemo", "phi4",
# "deepseek-r1:latest"). Fallback table checked before regex guessing.
KNOWN_MODEL_PARAMS: dict[str, float] = {
    "llama3.2:1b": 1.2,
    "llama3.2:3b": 3.2,
    "llama3.2-vision": 11.2,
    "llama3.1:8b": 8.0,
    "llama3:8b": 8.0,
    "llama3.1:70b": 70.0,
    "mistral-nemo": 12.2,
    "mistral:7b": 7.2,
    "phi4": 14.7,
    "phi3": 3.8,
    "gemma3:1b": 1.0,
    "gemma3:4b": 4.3,
    "gemma3:12b": 12.2,
    "gemma3:27b": 27.4,
    "gemma2:9b": 9.2,
    "qwen3:8b": 8.2,
    "qwen3:4b": 4.0,
    "qwen2.5:7b": 7.6,
    "qwen2.5-coder:7b": 7.6,
    "qwen2.5-coder:1.5b": 1.5,
    "deepseek-r1:1.5b": 1.8,
    "deepseek-r1:7b": 7.6,
    "deepseek-r1:8b": 7.6,
    "deepseek-r1:32b": 33.6,
    "codellama:7b": 7.0,
    "tinyllama": 1.1,
    "smollm2:360m": 0.36,
    "smollm2:1.7b": 1.7,
    "mixtral:8x7b": 46.7,
    "moondream": 1.9,
    "llava:7b": 7.0,
    "orca-mini:3b": 3.0,
}

# Param-count token, e.g. "8b", "70B", "1.5b", "0.36b", "123b".
PARAM_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", re.IGNORECASE)
# "13b-moE" style false positives to avoid matching "nemo"? handled by guards.
QUANT_RE = re.compile(r"(?i)(fp16|f16|bf16|fp32|f32|q[2-8](?:_[0-9a-z]+)*)")
MOE_RE = re.compile(r"(?i)mixtral|nemo|moex|extra-emb|8x7b|8x22b")


@dataclass
class ModelEstimate:
    model_id: str
    params_billions: float
    quant: str
    download_bytes: int
    family: str

    def download_gib(self) -> float:
        return self.download_bytes / GiB


def _norm(text: str) -> str:
    return text.lower()


def _lookup_known_params(model_id: str) -> float | None:
    name = model_id.lower()
    # Exact-ish key match ignoring tag: split on ":" and try variants.
    base = name.split(":")[0]
    for candidate in (name, base, base.split("/")[-1]):
        if candidate in KNOWN_MODEL_PARAMS:
            return KNOWN_MODEL_PARAMS[candidate]
    # Substring fallback for families like "mistral-nemo:latest".
    for key, params in KNOWN_MODEL_PARAMS.items():
        if key in name:
            return params
    return None


def _guess_quant(model_id: str) -> str:
    m = QUANT_RE.search(model_id)
    if not m:
        return DEFAULT_QUANT
    q = m.group(1).lower()
    if q.startswith("q"):
        return q.split("_")[0]
    return q


def estimate_model(model_id: str) -> ModelEstimate | None:
    """Best-effort size estimate from an endpoint's model ID string."""
    if not model_id or not model_id.strip():
        return None
    known = _lookup_known_params(model_id)
    params: float | None = None
    if known is not None and ":" not in model_id:
        # Bare family name (no size tag): trust the table, don't re-guess.
        params = known
    elif PARAM_RE.search(model_id) and not MOE_RE.search(model_id):
        m = PARAM_RE.search(model_id)
        params = float(m.group(1))  # type: ignore[union-attr]
    elif known is not None:
        params = known

    if params is None or params <= 0:
        return None

    quant = _guess_quant(model_id)
    bytes_per = QUANT_BYTES_PER_PARAM[quant]
    download_bytes = int(params * 1e9 * bytes_per * (1024**3) / 1e9)
    family = model_id.split(":")[0].split("/")[-1].split("-")[0] or model_id
    return ModelEstimate(
        model_id=model_id,
        params_billions=params,
        quant=quant,
        download_bytes=download_bytes,
        family=family,
    )


def ram_needed_bytes(download_bytes: int) -> int:
    """RAM required to run comfortably = model + KV cache + runtime overhead."""
    overhead = int(1.5 * GiB)  # context cache + app overhead, conservative
    return int(download_bytes + overhead + max(download_bytes * 0.15, 0.5 * GiB))
