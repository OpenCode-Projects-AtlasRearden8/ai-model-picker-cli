# ai-model-picker

"Which model should I actually run?" - answered like a friend would.

`ai-model-picker` is a beginner-friendly command-line tool that looks at your
computer (RAM, CPU, and optionally an NVIDIA GPU), checks the model list from
your local AI server, and prints a ranked list of the models that will run
well - each with a plain-English reason why.

Built and maintained by **Asdale Dynamics, LLC** as part of a beginner-first
AI toolkit. Local-first, standard library only (no dependencies), and it never
sends your data anywhere: the only network call it makes is to the endpoint
you point it at.

## What it does

- **Reads your hardware.** Total and available RAM, CPU cores, and if
  `nvidia-smi` is present, your NVIDIA GPU's VRAM.
- **Scans one of your AI tools.** Default: Ollama at `http://localhost:11434/v1`.
  Any OpenAI-compatible endpoint works (LM Studio, llama.cpp server, vLLM...).
- **Estimates the fit.** Known model family sizes plus quantization-aware math
  (a `q4_K_M` file is much smaller than an `f16` one).
- **Explains itself in plain English.** "Fits entirely in your 8 GiB of VRAM -
  fast, fully offloaded inference." No jargon walls, no condescension.

## Quick start

```bash
# 1. Start Ollama (or any OpenAI-compatible server), then:
pip install .            # from this folder
ai-model-picker

# or with zero install, from the project folder:
python -m ai_model_picker
```

Example output:

```
1. qwen2.5:7b  [GOOD]
     - ~7.6 B parameters, estimated ~4.4 GiB downloaded (q4-class).
     - Needs roughly 6.9 GiB RAM; you have 16.0 GiB total (8.7 GiB free), so it should run on CPU.
```

## Common options

```bash
ai-model-picker --hardware-only                    # just show detected hardware
ai-model-picker --base-url http://localhost:1234/v1  # LM Studio example
ai-model-picker --top 5                            # fewer suggestions
ai-model-picker --show-unfit                       # include models that won't fit
ai-model-picker --json                             # machine-readable output
ai-model-picker --config ./my-settings.toml        # custom config file
```

Config files are optional. By default the tool reads
`~/.ai-model-picker.toml`:

```toml
base_url = "http://localhost:11434/v1"
timeout = 5.0
top = 8
show_unfit = false
```

## Privacy & secrets

- Everything runs locally. The only HTTP request made is a `GET` on the
  `/models` list of the endpoint you configured.
- If your endpoint needs an API key, set the `AI_MODEL_PICKER_API_KEY`
  environment variable. The tool never writes keys, secrets, or personal data
  to disk.

## Known limits

- Size estimates are approximations (model files vary by fine-tune and quant
  mix). Treat the numbers as guidance, not gospel.
- Apple Silicon memory and AMD/Intel GPU VRAM aren't detected yet; if you run
  on a Mac, RAM numbers are still meaningful.

## Development

```bash
pip install -e '.[dev]'
pytest
```

## License

MIT - see [LICENSE](LICENSE). **Copyright (c) 2026 Asdale Dynamics, LLC.**
