import numpy as np

from commodity_forecast.data import fetch_wti_daily
from commodity_forecast.forecast import fit_calibrated_naive_forecaster


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
