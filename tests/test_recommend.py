"""Tests for ranking logic with mocked hardware."""

from ai_model_picker import recommend


class TestRanking:
    def test_small_model_ranked_first(self, mid_tier_machine):
        ids = ["llama3.1:70b-q4_K_M", "qwen2.5:7b-instruct-q4_K_M", "llama3.2:3b"]
        recs = recommend.recommend(ids, mid_tier_machine)
        # 7B q4 fits entirely in the 8 GiB VRAM -> "excellent fit", and
        # within that tier bigger models come first.
        first = recs[0]
        assert first.estimate.params_billions == 7.0
        assert first.fit_tier == recommend.TIER_EXCELLENT
        assert recs[-1].model_id == "llama3.1:70b-q4_K_M"
        assert recs[-1].fit_tier == recommend.TIER_NO

    def test_does_not_fit_flagged(self, mid_tier_machine):
        recs = recommend.recommend(["llama3.1:70b-q4_K_M"], mid_tier_machine)
        assert recs[0].fit_tier == recommend.TIER_NO
        assert any("RAM" in r for r in recs[0].reasons)

    def test_same_tier_prefers_bigger_models(self, budget_machine):
        recs = recommend.recommend(["tinyllama:latest", "llama3.2:3b"], budget_machine)
        fits = [r for r in recs if r.fit_tier != recommend.TIER_NO]
        assert len(fits) >= 2
        # Within one tier, larger params should come first.
        tiers = {r.model_id: r.fit_tier for r in fits}
        if tiers["tinyllama:latest"] == tiers["llama3.2:3b"]:
            ids = [r.model_id for r in fits]
            assert ids.index("llama3.2:3b") < ids.index("tinyllama:latest")

    def test_gpu_fit_beats_ram_fit(self, mid_tier_machine):
        # 4B q4 fits in VRAM; 8B at q8 (~9 GiB) does not fit in 8 GiB VRAM.
        recs = recommend.recommend(["gemma3:4b", "llama3.1:8b-q8_0"], mid_tier_machine)
        gpu_rec = next(r for r in recs if r.gpu_fits)
        no_gpu_rec = next(r for r in recs if not r.gpu_fits)
        assert gpu_rec.fit_tier == recommend.TIER_EXCELLENT
        assert (
            recommend.TIER_PENALTY[gpu_rec.fit_tier]
            < recommend.TIER_PENALTY[no_gpu_rec.fit_tier]
        )

    def test_no_hardware_info_hedges(self, budget_machine):
        empty = budget_machine
        empty.total_ram_bytes = 0
        empty.available_ram_bytes = 0
        recs = recommend.recommend(["llama3.2:3b"], empty)
        assert recs[0].fit_tier == recommend.TIER_UNLIKELY


class TestTopRecommendation:
    def test_returns_fitting_model(self, mid_tier_machine):
        recs = recommend.recommend(
            ["llama3.2:3b", "llama3.1:70b-q4_K_M"], mid_tier_machine
        )
        top = recommend.top_recommendation(recs)
        assert top.model_id == "llama3.2:3b"

    def test_none_when_no_models(self, mid_tier_machine):
        assert recommend.top_recommendation([]) is None


class TestReasonsArePlainEnglish:
    def test_reasons_read_like_sentences(self, mid_tier_machine):
        recs = recommend.recommend(["qwen2.5:7b-instruct-q4_K_M"], mid_tier_machine)
        reasons = recs[0].reasons
        assert len(reasons) >= 2
        for reason in reasons:
            assert reason.endswith(".")
            assert "NaN" not in reason and "None" not in reason
