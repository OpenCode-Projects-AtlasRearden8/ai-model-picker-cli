"""Fetch the /models list from an OpenAI-compatible endpoint.

Works with Ollama, LM Studio, llama.cpp server, vLLM, or anything that
speaks the OpenAI "/v1/models" route. Only the stdlib is used, so this
runs anywhere Python does. An optional bearer token can be supplied via
the AI_MODEL_PICKER_API_KEY environment variable - it is read at runtime
and never stored in any file.

Copyright (c) 2026 Asdale Dynamics, LLC
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclass
class EndpointScan:
    base_url: str
    resolved_url: str
    model_ids: list[str]
    error: str | None = None


def normalize_models_url(base_url: str) -> str:
    """Turn a loose base URL into a concrete /models URL.

    http://localhost:11434       -> http://localhost:11434/v1/models
    http://localhost:11434/v1    -> http://localhost:11434/v1/models
    http://localhost:11434/v1/   -> http://localhost:11434/v1/models
    .../v1/models                -> unchanged
    """
    url = base_url.strip().rstrip("/")
    if url.endswith("/models"):
        return url
    if url.endswith(("/v1", "/v1beta")):
        return f"{url}/models"
    return f"{url}/v1/models"


def _auth_header() -> dict[str, str]:
    key = os.environ.get("AI_MODEL_PICKER_API_KEY", "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def fetch_model_ids_raw(base_url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """GET the /models endpoint and return the raw JSON body as text."""
    url = normalize_models_url(base_url)
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", **_auth_header()}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_models_json(body: str) -> list[str]:
    """Extract model IDs from an OpenAI-style response ({"data":[{"id":...}]}).

    Also tolerates plain lists and Ollama-native {"models":[{"name":...}]}.
    """
    payload = json.loads(body)
    entries: list = []
    if isinstance(payload, dict):
        entries = payload.get("data", payload.get("models", []))
    elif isinstance(payload, list):
        entries = payload
    ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for field_name in ("id", "name", "model"):
            value = entry.get(field_name)
            if isinstance(value, str) and value.strip():
                ids.append(value.strip())
                break
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique = [mid for mid in ids if not (mid in seen or seen.add(mid))]  # type: ignore[func-returns-value]
    return unique


def scan_endpoint(
    base_url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    raw_fetcher=fetch_model_ids_raw,
) -> EndpointScan:
    resolved = normalize_models_url(base_url)
    try:
        body = raw_fetcher(base_url, timeout)
        ids = parse_models_json(body)
        if not ids:
            return EndpointScan(
                base_url=base_url,
                resolved_url=resolved,
                model_ids=[],
                error="Endpoint responded but returned no models. Is the service started?",
            )
        return EndpointScan(base_url=base_url, resolved_url=resolved, model_ids=ids)
    except urllib.error.HTTPError as exc:
        hint = ""
        if exc.code == 404:
            hint = " (URL may be wrong - try http://localhost:11434/v1)"
        elif exc.code in (401, 403):
            hint = " (set the AI_MODEL_PICKER_API_KEY environment variable)"
        return EndpointScan(
            base_url=base_url,
            resolved_url=resolved,
            model_ids=[],
            error=f"HTTP {exc.code}{hint}",
        )
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return EndpointScan(
            base_url=base_url,
            resolved_url=resolved,
            model_ids=[],
            error=f"Could not connect: {reason}. Is the local AI server running? Start Ollama (or your server) first.",
        )
    except json.JSONDecodeError:
        return EndpointScan(
            base_url=base_url,
            resolved_url=resolved,
            model_ids=[],
            error="Endpoint returned something that isn't JSON. Is that really an OpenAI-compatible endpoint?",
        )
    except Exception as exc:  # noqa: BLE001 - any failure becomes a friendly message
        return EndpointScan(
            base_url=base_url,
            resolved_url=resolved,
            model_ids=[],
            error=f"Request failed: {exc}",
        )
