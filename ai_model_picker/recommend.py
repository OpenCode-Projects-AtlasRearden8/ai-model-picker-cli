"""Rank models against the local machine's hardware, in plain English.

Copyright (c) 2026 Asdale Dynamics, LLC
"""

from __future__ import annotations

from dataclasses import dataclass

from . import models as model_sizing
from .hardware import GIB, GpuInfo, Hardware
from .models import ModelEstimate

# Fit tiers, best to worst. Lower penalty = better.
TIER_EXCELLENT = "excellent fit"
TIER_GOOD = "good fit"
TIER_TIGHT = "will run, but snug"
TIER_UNLIKELY = "probably too big"
TIER_NO = "will not fit here"

TIER_PENALTY = {
    TIER_EXCELLENT: 0,
    TIER_GOOD: 1,
    TIER_TIGHT: 2,
    TIER_UNLIKELY: 3,
    TIER_NO: 4,
}


def _gib(n: int) -> float:
    return n / GIB


@dataclass
class Recommendation:
    model_id: str
    estimate: ModelEstimate
    ram_needed_bytes: int
    fit_tier: str
    gpu_fits: bool
    reasons: list[str]


def _assess_gpu(gpus: list[GpuInfo], download_bytes: int) -> bool:
    """True if the whole model fits in combined VRAM with headroom."""
    vram = sum(g.vram_bytes for g in gpus)
    return vram > 0 and download_bytes <= int(vram * 0.92)


def _tier_for(
    estimate: ModelEstimate, ram_needed: int, hw: Hardware, gpu_fits: bool
) -> str:
    if gpu_fits:
        return TIER_EXCELLENT
    budget = hw.available_ram_bytes or hw.total_ram_bytes
    if budget <= 0:
        return TIER_UNLIKELY
    if ram_needed <= int(budget * 0.55):
        return TIER_GOOD
    if ram_needed <= int(budget * 0.75):
        return TIER_TIGHT
    if ram_needed <= budget:
        return TIER_TIGHT
    if ram_needed <= int(budget * 1.25):
        return TIER_UNLIKELY
    return TIER_NO


def _reasons(
    rec_model_id: str, est: ModelEstimate, ram_needed: int, hw: Hardware, gpu_fits: bool
) -> list[str]:
    size = _gib(est.download_bytes)
    need = _gib(ram_needed)
    reasons: list[str] = [
        f"~{est.params_billions:g} B parameters, estimated ~{size:.1f} GiB downloaded ({est.quant}-class)."
    ]
    if gpu_fits:
        vram = _gib(hw.total_vram_bytes())
        reasons.append(
            f"Fits entirely in your {vram:.0f} GiB of VRAM - fast, fully offloaded inference."
        )
    ram_line = f"{hw.total_ram_gib():.1f} GiB total"
    if hw.available_ram_bytes:
        ram_line += f" ({hw.available_ram_gib():.1f} GiB free)"
    if gpu_fits:
        reasons.append(
            f"Needs roughly {need:.1f} GiB RAM too, so it stays comfortable on {ram_line}."
        )
    elif need <= _gib(hw.available_ram_bytes or hw.total_ram_bytes):
        reasons.append(
            f"Needs roughly {need:.1f} GiB RAM; you have {ram_line}, so it should run on CPU."
        )
    elif need <= _gib(hw.available_ram_bytes or hw.total_ram_bytes) * 1.25:
        reasons.append(
            f"Needs ~{need:.1f} GiB RAM against {ram_line} - expect slowdowns if other apps are open."
        )
    else:
        reasons.append(
            f"Needs ~{need:.1f} GiB RAM, more than your {ram_line}; expect heavy swapping or failure."
        )
    return reasons


def recommend(model_ids: list[str], hw: Hardware) -> list[Recommendation]:
    """Turn endpoint model IDs into ranked recommendations."""
    recs: list[Recommendation] = []
    for mid in model_ids:
        est = model_sizing.estimate_model(mid)
        if est is None:
            continue
        needed = model_sizing.ram_needed_bytes(est.download_bytes)
        gpu_fits = _assess_gpu(hw.gpus, est.download_bytes)
        tier = _tier_for(est, needed, hw, gpu_fits)
        recs.append(
            Recommendation(
                model_id=mid,
                estimate=est,
                ram_needed_bytes=needed,
                fit_tier=tier,
                gpu_fits=gpu_fits,
                reasons=_reasons(mid, est, needed, hw, gpu_fits),
            )
        )
    return _rank(recs)


def _rank(recs: list[Recommendation]) -> list[Recommendation]:
    """Best fit first; within a tier, larger models (better quality)."""
    return sorted(
        recs,
        key=lambda r: (
            TIER_PENALTY[r.fit_tier],
            -r.estimate.params_billions,
            r.model_id,
        ),
    )


def top_recommendation(recs: list[Recommendation]) -> Recommendation | None:
    """Highest-ranked rec that actually fits, falling back to the closest."""
    fitting = [r for r in recs if TIER_PENALTY[r.fit_tier] <= TIER_PENALTY[TIER_TIGHT]]
    return fitting[0] if fitting else (recs[0] if recs else None)
