"""Command-line interface.

Examples:
    ai-model-picker
    ai-model-picker --base-url http://localhost:1234/v1
    ai-model-picker --top 5 --show-unfit --json

Copyright (c) 2026 Asdale Dynamics, LLC
"""

from __future__ import annotations

import argparse
import json as jsonlib
import sys

from . import __version__, config, endpoint, hardware, recommend
from .hardware import GIB, Hardware


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-model-picker",
        description=(
            "Recommend which local LLM to run on this machine. Scans an "
            "OpenAI-compatible endpoint (Ollama, LM Studio, llama.cpp, ...), "
            "checks the fit against your RAM and GPU, and explains why."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default: Ollama http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a TOML config file (default: ~/.ai-model-picker.toml)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="HTTP timeout in seconds (default 5)",
    )
    parser.add_argument(
        "--top", type=int, default=None, help="How many suggestions to show (default 8)"
    )
    parser.add_argument(
        "--show-unfit", action="store_true", help="Also list models that will not fit"
    )
    parser.add_argument(
        "--json", action="store_true", help="Machine-readable JSON output"
    )
    parser.add_argument(
        "--hardware-only",
        action="store_true",
        help="Just print detected hardware and exit",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def make_recommendations(
    hw: Hardware, model_ids: list[str]
) -> list[recommend.Recommendation]:
    return recommend.recommend(model_ids, hw)


def _fmt_gib(n_bytes: int) -> str:
    return f"{n_bytes / GIB:.1f} GiB"


def print_report(
    hw: Hardware,
    recs: list[recommend.Recommendation],
    scan: endpoint.EndpointScan,
    top: int,
    show_unfit: bool,
) -> None:
    print()
    print("Here's what we found on your machine:")
    print(hw.describe())
    print()
    if scan.error:
        print("Model scan trouble:")
        print(f"  {scan.error}")
        print("  Check the server is running, or point --base-url at it explicitly.")
        return
    if not recs:
        print(
            "The endpoint answered, but none of its models are ones we can size up yet."
        )
        return

    fitting = [
        r
        for r in recs
        if recommend.TIER_PENALTY[r.fit_tier]
        <= recommend.TIER_PENALTY[recommend.TIER_TIGHT]
    ]
    skipped = [r for r in recs if r not in fitting]
    chosen = fitting[:top] if fitting else recs[:0]
    if not chosen and show_unfit:
        chosen = recs[:top]
    print(
        f'Endpoint "{scan.base_url}" answered with {len(recs)} model(s). Ranked suggestions:'
    )
    print()
    for i, rec in enumerate(chosen, start=1):
        badge = recommend.TIER_PENALTY[rec.fit_tier]
        marker = {
            0: "EXCELLENT FIT",
            1: "GOOD FIT",
            2: "RUNS BUT SNUG",
            3: "RISKY FIT",
            4: "WON'T FIT",
        }.get(badge, "?")
        print(f"{i}. {rec.model_id}  [{marker}]")
        for reason in rec.reasons:
            print(f"     - {reason}")
        print()
    if skipped and not show_unfit:
        print(
            f"({len(skipped)} model(s) you already have do not fit this machine - rerun with --show-unfit to see them.)"
        )


def _recommendation_to_dict(rec: recommend.Recommendation) -> dict:
    return {
        "model_id": rec.model_id,
        "fit_tier": rec.fit_tier,
        "params_billions": rec.estimate.params_billions,
        "quant": rec.estimate.quant,
        "download_bytes": rec.estimate.download_bytes,
        "ram_needed_bytes": rec.ram_needed_bytes,
        "gpu_fits": rec.gpu_fits,
        "reasons": rec.reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    conf = config.load_config(args.config)

    base_url = args.base_url or conf.get("base_url") or endpoint.DEFAULT_BASE_URL
    timeout = (
        args.timeout
        if args.timeout is not None
        else float(conf.get("timeout", endpoint.DEFAULT_TIMEOUT_SECONDS))
    )
    top = args.top if args.top is not None else int(conf.get("top", 8))

    hw = hardware.detect_hardware()
    if args.hardware_only:
        print(hw.describe())
        return 0

    scan = endpoint.scan_endpoint(str(base_url), timeout=timeout)
    recs = make_recommendations(hw, scan.model_ids)

    if args.json:
        if scan.error:
            payload_recs: list = []
        else:
            visible = (
                recs
                if args.show_unfit
                else [
                    r
                    for r in recs
                    if recommend.TIER_PENALTY[r.fit_tier]
                    <= recommend.TIER_PENALTY[recommend.TIER_TIGHT]
                ]
            )
            payload_recs = [_recommendation_to_dict(r) for r in visible[:top]]
        payload = {
            "version": __version__,
            "hardware": {
                "total_ram_bytes": hw.total_ram_bytes,
                "available_ram_bytes": hw.available_ram_bytes,
                "cpu_count": hw.cpu_count,
                "cpu_model": hw.cpu_model,
                "gpus": [
                    {"name": g.name, "vram_bytes": g.vram_bytes, "source": g.source}
                    for g in hw.gpus
                ],
            },
            "endpoint": {
                "base_url": scan.base_url,
                "resolved_url": scan.resolved_url,
                "error": scan.error,
            },
            "recommendations": payload_recs,
        }
        print(jsonlib.dumps(payload, indent=2))
        return 0 if not scan.error else 1

    print_report(hw, recs, scan, top, args.show_unfit)
    return 0 if not scan.error else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
