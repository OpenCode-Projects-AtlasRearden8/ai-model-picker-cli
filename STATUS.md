# STATUS

**Project:** ai-model-picker - a beginner-friendly CLI that recommends which local LLM to run on your machine.
**Owner:** Asdale Dynamics, LLC (copyright (c) 2026 Asdale Dynamics, LLC)
**Date:** 2026-10-23
**Status:** Working, tested, committed. Nothing was published or pushed - local repo only.

## What was built

A single, dependency-free Python package (`ai_model_picker`, requires Python 3.11+) that:

1. **Reads your hardware** (`hardware.py`): total/available RAM from `/proc/meminfo` (with
   `os.sysconf` / Windows `GlobalMemoryStatusEx` / macOS `sysctl` fallbacks), CPU cores and
   model, and optional NVIDIA GPU VRAM by parsing `nvidia-smi --query-gpu=...` output
   (gracefully absent-GPU aware).
2. **Scans an OpenAI-compatible endpoint** (`endpoint.py`): default `http://localhost:11434/v1`
   (Ollama); any base URL via `--base-url` or a TOML config file (`~/.ai-model-picker.toml`
   or `--config`). Normalizes loose base URLs to `/v1/models`, tolerates OpenAI-style,
   plain-list, and Ollama-native JSON, and gives friendly troubleshooting messages for
   connection errors, HTTP 401/403/404, and non-JSON replies. Optional bearer token is read
   at runtime from the `AI_MODEL_PICKER_API_KEY` environment variable - never stored in files.
3. **Estimates model fit** (`models.py`): parses parameter counts (e.g. `7b`, `1.5b`) and
   quantization tags (`q4_K_M`, `q8_0`, `f16`...) from model IDs, uses a curated table for
   models whose IDs don't state the size (`mistral-nemo`, `phi4`, ...), and estimates
   download size plus RAM needed (KV cache + runtime overhead).
4. **Ranks and explains** (`recommend.py`, `cli.py`): five plain-English fit tiers
   (EXCELLENT FIT / GOOD FIT / RUNS BUT SNUG / RISKY FIT / WON'T FIT), VRAM-first preference,
   ranked suggestions with sentence-style reasons, plus `--json`, `--top`, `--show-unfit`,
   `--hardware-only`, `--timeout`, `--config`, `--version` flags. Human-friendly error
   paths exit with code 1; happy paths exit 0.

## How to run it

```bash
# Zero-install, from the project folder:
python -m ai_model_picker

# Or install with the console entry point:
pip install -e .
ai-model-picker

# Useful flags:
python -m ai_model_picker --hardware-only
python -m ai_model_picker --base-url http://localhost:1234/v1   # LM Studio example
python -m ai_model_picker --show-unfit --top 10 --json
```

## Verification results

- **Tests:** 61/61 passing (`python3 -m pytest`, and again from a clean venv install).
- **Lint/format:** `ruff check` clean, `ruff format` applied.
- **Packaging:** `pip install -e '.[dev]'` in a fresh venv works; `ai-model-picker \
  --help/--version` print correctly; `ai-model-picker` entry point runs.
- **Live smoke tests:**
  - Real hardware report on this machine (Linux, 16 GiB RAM, no NVIDIA GPU): expected output.
  - Fake local OpenAI-compatible endpoint (stdlib HTTP server): ranked suggestions with
    reasons, `--json` output parses exactly as intended, exit code 0.
  - Unreachable endpoint: friendly "Could not connect..." guidance, exit code 1.
- Confirmed no secrets or personal data are read or stored anywhere; the only network
  call is the `GET /models` request to the endpoint you point it at.

## Known gaps

- Size estimates are approximations: real GGUF file sizes vary with fine-tunes, MoE
  architecture, and quant mixes; the fit tiers add conservative overhead to compensate.
- Apple Silicon is not GPU-detected (RAM numbers only); AMD/Intel GPU VRAM is not read.
- The endpoint scan only lists models - it does not query disk usage from the server, so a
  model installed only on another disk may size differently.
- Unsized/unknown model IDs that contain no recognizable parameter pattern are skipped
  (they don't show up in suggestions).
- Windows/macOS RAM paths are implemented but only exercised on Linux in this environment
  (unit-tested via mocked parsers; platform code paths are marked no-cover).

## Files

- `ai_model_picker/` - `cli.py`, `hardware.py`, `models.py`, `endpoint.py`, `recommend.py`,
  `config.py`, `__main__.py`
- `tests/` - 61 tests with fully mocked hardware and endpoint data
- `README.md`, `LICENSE` (MIT, Copyright (c) 2026 Asdale Dynamics, LLC), `pyproject.toml`,
  `.gitignore`, `STATUS.md`
