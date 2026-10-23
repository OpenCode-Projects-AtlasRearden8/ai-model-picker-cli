"""Optional TOML config file support (stdlib tomllib).

Default location: ~/.ai-model-picker.toml. Example:

    base_url = "http://localhost:11434/v1"
    timeout = 5.0
    top = 10
    show_unfit = false

No secrets belong in here or anywhere else; API keys, if you need one,
go in the AI_MODEL_PICKER_API_KEY environment variable instead.

Copyright (c) 2026 Asdale Dynamics, LLC
"""

from __future__ import annotations

import os
import tomllib

CONFIG_ENV_VAR = "AI_MODEL_PICKER_CONFIG"
DEFAULT_CONFIG_NAME = ".ai-model-picker.toml"


def default_config_path() -> str | None:
    home = os.path.expanduser("~")
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return override
    return os.path.join(home, DEFAULT_CONFIG_NAME)


def load_config(path: str | None = None) -> dict:
    """Load a TOML config file, returning {} if missing/invalid."""
    if path is None:
        path = default_config_path()
        if not path or not os.path.exists(path):
            return {}
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
