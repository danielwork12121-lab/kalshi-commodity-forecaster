import numpy as np
import pytest

from commodity_forecast.data import fetch_wti_daily
from commodity_forecast.forecast import fit_adaptive_naive_forecaster, fit_calibrated_naive_forecaster


def test_fit_calibrated_naive_forecaster_on_synthetic_series():
    """Sanity-checks the pipeline mechanics (shapes, diagnostic keys, basic
    bounds) against conformal-forecast's own synthetic generator, where the
    noise distribution is known and controlled -- isolating "does our
    wiring of conformal-forecast's split/window/conformal machinery work"
    from "is real commodity data well-behaved", which the next test covers.
    """
    from forecasting.data import generate_synthetic

    series = generate_synthetic(n=600, period=24, noise_std=1.0, seed=1)
    result = fit_calibrated_naive_forecaster(series, lookback=5, alpha=0.1)
    d = result.diagnostics

    assert d["n_train_windows"] > 0
    assert d["n_cal_windows"] > 0
    assert 0.0 <= d["empirical_coverage"] <= 1.0
    assert d["mean_interval_width"] > 0
    assert d["mae_usd_per_bbl"] >= 0


def test_fit_calibrated_naive_forecaster_on_real_wti_series():
    """Exercises the whole pipeline against the real, currently-cached WTI
    daily spot price series -- not synthetic data -- so the baseline and
    its calibration are checked against real market dynamics (volatility
    clustering, real jumps), not just a clean sinusoid-plus-noise.

    Requires `commodity_forecast/data.py`'s own module-level fetch (run as
    `python -m commodity_forecast.data`, or any prior call to
    fetch_wti_daily) to have already populated data/wti_spot_daily.csv;
    this test itself does not hit the network.
    """
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    result = fit_calibrated_naive_forecaster(series, lookback=5, alpha=0.1)
    d = result.diagnostics

    assert d["n_test"] > 100
    # Loose sanity bounds, not a tight regression check: real WTI daily
    # moves are usually well under $10/bbl day-over-day, so a naive
    # (today-predicts-tomorrow) baseline's MAE should sit well below the
    # price level itself (currently in the $60-$100/bbl range).
    assert 0 < d["mae_usd_per_bbl"] < 20
    assert 0.0 <= d["empirical_coverage"] <= 1.0
    assert d["mean_interval_width"] > 0


def test_fit_adaptive_naive_forecaster_rejects_unknown_variant():
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    with pytest.raises(ValueError, match="variant"):
        fit_adaptive_naive_forecaster(series, variant="nope")


@pytest.mark.parametrize("variant", ["fixed", "sliding"])
def test_fit_adaptive_naive_forecaster_on_synthetic_series(variant):
    """Sanity-checks ACI's wiring (shapes, diagnostic keys, basic bounds) on
    the same synthetic generator `test_fit_calibrated_naive_forecaster_on_synthetic_series`
    uses for the static baseline, for both ACI variants.
    """
    from forecasting.data import generate_synthetic

    series = generate_synthetic(n=600, period=24, noise_std=1.0, seed=1)
    result = fit_adaptive_naive_forecaster(series, lookback=5, alpha=0.1, variant=variant)
    d = result.diagnostics

    assert d["n_train_windows"] > 0
    assert d["n_cal_windows"] > 0
    assert 0.0 <= d["empirical_coverage"] <= 1.0
    assert d["mean_interval_width"] > 0
    assert d["mae_usd_per_bbl"] >= 0
    assert d["variant"] == variant
    # ACI's own long-run guarantee is on mean_error_rate -> alpha, not on
    # empirical_coverage == nominal_coverage exactly -- both should be in a
    # sane range, not pinned to a specific value on a short synthetic run.
    assert 0.0 <= d["mean_error_rate"] <= 1.0
    assert 0.0 < d["final_alpha_t"] < 1.0


@pytest.mark.parametrize("variant", ["fixed", "sliding"])
def test_fit_adaptive_naive_forecaster_on_real_wti_series(variant):
    """Exercises ACI (both variants) against the real, full WTI series --
    the actual question this test answers: does either variant meaningfully
    close the 80.9%-vs-90% coverage gap `fit_calibrated_naive_forecaster`
    found with static split conformal on this exact series, same model,
    same split, same alpha? This test only pins down sane bounds (ACI is
    genuinely a different result each time the underlying series is
    refreshed, since it's a sequential online process over real market
    data) -- the actual honest numbers are measured and reported in the
    README, not asserted as a fixed regression value here.
    """
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    result = fit_adaptive_naive_forecaster(series, lookback=5, alpha=0.1, variant=variant)
    d = result.diagnostics

    assert d["n_test"] > 100
    assert 0 < d["mae_usd_per_bbl"] < 20
    assert 0.0 <= d["empirical_coverage"] <= 1.0
    assert d["mean_interval_width"] > 0
    assert 0.0 <= d["mean_error_rate"] <= 1.0
    assert d["variant"] == variant
    # The actual question this feature exists to answer, pinned as a real
    # regression check now that it's been measured (see README "First real
    # result"): static split conformal measured 80.9% empirical coverage
    # against the 90% nominal target on this exact series/model/split.
    # Both ACI variants measurably close that gap (fixed: 87.4%, sliding/
    # unbounded: 88.4%) -- a loose ">0.85" threshold, not an exact pin, so
    # a future, legitimate refresh of the committed WTI CSV cache (new
    # real data appended) doesn't spuriously fail this test over a few
    # tenths of a percentage point of real measurement noise.
    assert d["empirical_coverage"] > 0.85


def test_fit_adaptive_naive_forecaster_sliding_window_bounds_pool_growth():
    """A bounded `window` on the sliding variant should keep the diagnostic
    pipeline working end to end (shape/bounds sanity) exactly like the
    unbounded default -- this is the parameter conformal-forecast's own
    SlidingWindowAdaptiveConformalForecaster added specifically so the pool
    can shrink back down, not just grow forever.
    """
    from forecasting.data import generate_synthetic

    series = generate_synthetic(n=600, period=24, noise_std=1.0, seed=1)
    result = fit_adaptive_naive_forecaster(series, lookback=5, alpha=0.1, variant="sliding", window=20)
    d = result.diagnostics

    assert d["window"] == 20
    assert 0.0 <= d["empirical_coverage"] <= 1.0
    assert d["mean_interval_width"] > 0


def test_sliding_window_aci_with_tuned_window_nearly_closes_the_real_coverage_gap():
    """The actual headline finding this run measured (see README "First
    real result, with ACI"): a bounded sliding-window pool (explored at
    window in {20, 30, 50, 75, 100, 150, 200, 300} against the real WTI
    series) gets empirical coverage to ~89.5-89.8% almost everywhere in
    that range -- a small, honest 0.2-0.5pp residual gap from the 90%
    nominal target, down from static split conformal's 9.1pp gap (80.9%).
    window=75 is pinned here (not asserted as the literal best -- the
    differences between 20/75/100 are within real measurement noise) as
    one concrete, reproducible point in that range. A loose threshold
    (">0.89"), not an exact float pin, for the same reason as the
    adjacent parametrized test.
    """
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    result = fit_adaptive_naive_forecaster(series, lookback=5, alpha=0.1, variant="sliding", window=75)
    assert result.diagnostics["empirical_coverage"] > 0.89
