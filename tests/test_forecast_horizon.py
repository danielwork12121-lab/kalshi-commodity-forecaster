import numpy as np
import pytest

from commodity_forecast.data import fetch_wti_daily
from commodity_forecast.forecast import (
    fit_calibrated_naive_forecaster,
    fit_calibrated_naive_forecaster_horizon,
    prepare_horizon_split,
)
from commodity_forecast.probability import backtest_threshold_calibration


def test_prepare_horizon_split_rejects_horizon_below_one():
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    with pytest.raises(ValueError, match="horizon"):
        prepare_horizon_split(series, horizon=0)


def test_horizon_one_reduces_exactly_to_the_existing_one_step_pipeline():
    """The whole point of `prepare_horizon_split`/`fit_calibrated_naive_forecaster_horizon`
    is to generalize the existing one-step pipeline, not replace or drift
    from it -- at horizon=1 the two must agree exactly (same split, same
    model, same math), confirmed here on the real WTI series rather than
    just asserted in a docstring.
    """
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    baseline = fit_calibrated_naive_forecaster(series, lookback=5, alpha=0.1)
    horizon_one = fit_calibrated_naive_forecaster_horizon(series, horizon=1, lookback=5, alpha=0.1)

    assert baseline.diagnostics["n_test"] == horizon_one.diagnostics["n_test"]
    assert baseline.diagnostics["n_train_windows"] == horizon_one.diagnostics["n_train_windows"]
    assert baseline.diagnostics["n_cal_windows"] == horizon_one.diagnostics["n_cal_windows"]
    assert baseline.diagnostics["mae_usd_per_bbl"] == pytest.approx(horizon_one.diagnostics["mae_usd_per_bbl"])
    assert baseline.diagnostics["empirical_coverage"] == pytest.approx(horizon_one.diagnostics["empirical_coverage"])
    assert baseline.diagnostics["mean_interval_width"] == pytest.approx(horizon_one.diagnostics["mean_interval_width"])
    np.testing.assert_allclose(
        np.sort(baseline.cal_residuals_normalized), np.sort(horizon_one.cal_residuals_normalized)
    )


def test_fit_calibrated_naive_forecaster_horizon_on_synthetic_series():
    """Sanity-checks the horizon-aware pipeline's mechanics (shapes,
    diagnostic keys, basic bounds) on a controlled synthetic series, the
    same split-mechanics-first approach every other forecaster function in
    this module's own test file uses before trusting it on real data.
    """
    from forecasting.data import generate_synthetic

    series = generate_synthetic(n=600, period=24, noise_std=1.0, seed=1)
    for horizon in (1, 3, 10):
        result = fit_calibrated_naive_forecaster_horizon(series, horizon=horizon, lookback=5, alpha=0.1)
        d = result.diagnostics
        assert d["horizon"] == horizon
        assert d["n_train_windows"] > 0
        assert d["n_cal_windows"] > 0
        assert 0.0 <= d["empirical_coverage"] <= 1.0
        assert d["mean_interval_width"] > 0
        assert d["mae_usd_per_bbl"] >= 0


@pytest.mark.parametrize("horizon", [1, 2, 3, 5, 10, 15, 21])
def test_fit_calibrated_naive_forecaster_horizon_on_real_wti_series(horizon):
    """Exercises every one of this project's own validated horizons
    (`edge.VALIDATED_FORECAST_HORIZONS_DAYS`) against the real, full WTI
    series. Only loose sanity bounds are pinned here -- the actual honest
    numbers (which degrade as horizon grows, as they should for a flat
    naive forecast on a series that keeps moving) are measured and
    reported in the README, not asserted as exact regression values, for
    the same reason every other real-data test in this project uses loose
    bounds: a legitimate future refresh of the committed WTI cache
    shouldn't spuriously fail this test over ordinary measurement noise.
    """
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    result = fit_calibrated_naive_forecaster_horizon(series, horizon=horizon, lookback=5, alpha=0.1)
    d = result.diagnostics

    assert d["n_test"] > 100
    assert d["horizon"] == horizon
    assert 0.0 <= d["empirical_coverage"] <= 1.0
    assert d["mean_interval_width"] > 0
    # MAE should grow (not shrink) as horizon grows, since a flat
    # "tomorrow = today" forecast is a worse approximation the further out
    # it's held fixed on a series that keeps moving -- a real, structural
    # property of this method, not a coincidence of one run's data.
    assert 0 < d["mae_usd_per_bbl"] < 20 * horizon


def test_mae_increases_monotonically_with_horizon_on_real_data():
    """The core honest property this whole feature rests on: further out
    should mean (at least on average, allowing the rare tie) a harder,
    not easier, forecasting problem for a method that holds today's price
    flat. If this weren't true, treating each horizon's own residual pool
    as a genuine calibration (rather than just a wider version of the same
    one-day pool) would be measuring noise, not a real horizon effect.
    """
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    horizons = [1, 2, 3, 5, 10, 15, 21]
    maes = [
        fit_calibrated_naive_forecaster_horizon(series, horizon=h, lookback=5, alpha=0.1).diagnostics[
            "mae_usd_per_bbl"
        ]
        for h in horizons
    ]
    assert maes == sorted(maes)


def test_backtest_threshold_calibration_works_unchanged_on_a_horizon_result():
    """`probability.backtest_threshold_calibration` was built against
    `fit_calibrated_naive_forecaster`'s one-step `CalibratedForecastResult`
    shape -- confirms it needs zero changes to work on a horizon-validated
    result too, exactly as `fit_calibrated_naive_forecaster_horizon`'s own
    docstring promises, since both return the same dataclass shape.
    """
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    horizon = 5
    X_train, y_train, X_cal, y_cal, X_test, y_test = prepare_horizon_split(series, horizon=horizon, lookback=5)[:6]
    result = fit_calibrated_naive_forecaster_horizon(series, horizon=horizon, lookback=5, alpha=0.1)

    bt = backtest_threshold_calibration(result, X_test, y_test)
    assert bt.n_test_points == len(X_test)
    assert 0.0 <= bt.brier_score <= 1.0
    assert 0.0 <= bt.mean_calibration_error <= 1.0
