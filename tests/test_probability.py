import numpy as np
import pytest

from commodity_forecast.data import fetch_wti_daily
from commodity_forecast.forecast import fit_calibrated_naive_forecaster
from commodity_forecast.probability import backtest_threshold_calibration, exceedance_probability
from forecasting.experiment import prepare_split_from_series


def test_exceedance_probability_at_point_forecast_uses_empirical_residual_cdf():
    """Hand-computed sanity check with a tiny, known residual sample."""
    residuals = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])

    # threshold == point forecast -> need the residual to exceed 0 -> 3/6.
    assert exceedance_probability(np.array([0.0]), residuals, 0.0)[0] == pytest.approx(0.5)
    # threshold 1 above a 0 point forecast -> only residuals > 1 (just "2.0") -> 1/6.
    assert exceedance_probability(np.array([0.0]), residuals, 1.0)[0] == pytest.approx(1 / 6)
    # a threshold far beyond every residual -> ~0; far below every residual -> ~1.
    assert exceedance_probability(np.array([0.0]), residuals, 100.0)[0] == 0.0
    assert exceedance_probability(np.array([0.0]), residuals, -100.0)[0] == 1.0


def test_exceedance_probability_is_monotonically_decreasing_in_threshold():
    """A real, non-degenerate residual sample -- probability of exceeding a
    higher bar can never go up as the bar itself rises."""
    rng = np.random.default_rng(7)
    residuals = rng.normal(loc=0.0, scale=1.0, size=500)
    thresholds = np.linspace(-3.0, 3.0, 25)
    probs = np.array([exceedance_probability(np.array([0.0]), residuals, t)[0] for t in thresholds])
    assert np.all(np.diff(probs) <= 0.0)


def test_exceedance_probability_broadcasts_per_point_forecast():
    """A vector of point forecasts against one shared threshold should give
    a different probability per point -- a higher point forecast is closer
    to clearing a fixed bar, so its exceedance probability should be higher."""
    residuals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    probs = exceedance_probability(np.array([-1.0, 0.0, 1.0]), residuals, 0.0)
    assert probs[0] < probs[1] < probs[2]


def test_backtest_threshold_calibration_on_real_wti_series():
    """The honest check this module exists for: across many real,
    historical, non-cherry-picked questions (every test-split day x every
    relative offset), does the predicted probability roughly match the
    actual exceedance rate? A loose bound, not an exact pin, since this is
    a real measured result on real data (and the cached CSV can legitimately
    be refreshed with new real observations later).
    """
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    result = fit_calibrated_naive_forecaster(series, lookback=5, alpha=0.1)
    (_X_train, _y_train, _X_cal, _y_cal, X_test, y_test, _normalizer, _period) = prepare_split_from_series(
        series, lookback=5, period=5, label="wti_spot"
    )

    bt = backtest_threshold_calibration(result, X_test, y_test)

    assert bt.n_test_points == len(X_test)
    assert bt.n_observations == bt.n_test_points * len(bt.relative_offsets)
    # A method with no real skill (e.g. always predicting 0.5) scores a
    # Brier score of 0.25; a reasonably well-calibrated real method should
    # clear that bar by a real margin, not just barely.
    assert bt.brier_score < 0.15
    # Mean calibration error (count-weighted |predicted - actual| across
    # reliability bins) should be small, not exact -- some bins have very
    # few observations (a threshold far in the tail) and are noisier.
    assert bt.mean_calibration_error < 0.08
    assert len(bt.reliability_bins) >= 5
    for b in bt.reliability_bins:
        assert 0.0 <= b["mean_predicted_probability"] <= 1.0
        assert 0.0 <= b["empirical_exceedance_rate"] <= 1.0
        assert b["count"] > 0


def test_backtest_threshold_calibration_respects_custom_offsets():
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    result = fit_calibrated_naive_forecaster(series, lookback=5, alpha=0.1)
    (_X_train, _y_train, _X_cal, _y_cal, X_test, y_test, _normalizer, _period) = prepare_split_from_series(
        series, lookback=5, period=5, label="wti_spot"
    )

    bt = backtest_threshold_calibration(result, X_test, y_test, relative_offsets=(0.0, 0.03))
    assert bt.relative_offsets == (0.0, 0.03)
    assert bt.n_observations == bt.n_test_points * 2
