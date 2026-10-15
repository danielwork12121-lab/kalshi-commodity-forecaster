import numpy as np
import pytest

from commodity_forecast.data import fetch_wti_daily
from commodity_forecast.forecast import CalibratedForecastResult, fit_calibrated_naive_forecaster
from commodity_forecast.probability import (
    backtest_threshold_calibration,
    backtest_threshold_calibration_adaptive,
    exceedance_probability,
)
from forecasting.conformal import SplitConformalForecaster
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


class _ConstantModel:
    """A tiny stand-in model: always predicts the same value, for a fully
    hand-computable adaptive-pool scenario below. Not NaiveForecaster
    itself, specifically so the point forecast for every step is a known
    constant rather than depending on the preceding real observation."""

    def __init__(self, value: float):
        self.value = value

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ConstantModel":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), 1), self.value)


def _fake_conformal_result(seed_residuals: np.ndarray, value: float = 0.0) -> CalibratedForecastResult:
    """Build a real (not duck-typed) CalibratedForecastResult around a
    _ConstantModel, wrapped in a genuine SplitConformalForecaster -- so both
    `backtest_threshold_calibration` (which calls `.conformal_forecaster
    .predict(...)`) and `backtest_threshold_calibration_adaptive` (which
    only reads `.conformal_forecaster.model`) work against the exact same
    fixture without a hand-rolled mock drifting out of sync with the real
    interface either function actually relies on.
    """
    conformal = SplitConformalForecaster(model=_ConstantModel(value), alpha=0.1)
    # Trivial 1-row fit data -- q_hat itself is never used by either
    # function under test, only conformal.model and conformal.predict()'s
    # .point are.
    conformal.fit(
        X_train=np.zeros((1, 1)),
        y_train=np.array([[value]]),
        X_cal=np.zeros((1, 1)),
        y_cal=np.array([[value]]),
    )
    return CalibratedForecastResult(
        conformal_forecaster=conformal,
        normalizer_mean=0.0,
        normalizer_std=1.0,
        diagnostics={},
        cal_residuals_normalized=seed_residuals,
    )


def test_backtest_threshold_calibration_adaptive_hand_computed_step_ordering():
    """Fully hand-computed, 2-step scenario pinning down the exact
    step-ordering contract: step t is scored against the pool as it stood
    *before* step t, and only afterward does step t's own signed residual
    join the pool for step t+1 onward.

    Seed pool (6 values): [-2, -1, -0.5, 0.5, 1, 2]. Point forecast is a
    constant 0.0 at every step (see _ConstantModel), threshold offset 0.0
    -> "needed excess" is just 0 at every step, so each step's predicted
    probability is P(pool > 0).

    Step 0: pool is still just the 6-value seed -> P(pool > 0) = 3/6 = 0.5
    exactly (the values 0.5, 1, 2 clear the bar). True y[0] = 10.0 (>
    threshold 0) -> actually exceeded.
    After step 0, residual (y[0] - point[0]) = 10.0 joins the pool -> 7
    values: [-2, -1, -0.5, 0.5, 1, 2, 10].
    Step 1: P(pool > 0) = 4/7 (now 0.5, 1, 2, 10 clear the bar). True
    y[1] = 0.0 (not > threshold 0) -> not exceeded.
    """
    seed_residuals = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    fake_result = _fake_conformal_result(seed_residuals)
    X_test = np.zeros((2, 1))  # content irrelevant to _ConstantModel beyond its length
    y_test = np.array([10.0, 0.0])

    bt = backtest_threshold_calibration_adaptive(
        fake_result, X_test, y_test, relative_offsets=(0.0,), window=None
    )

    expected_predicted = np.array([0.5, 4.0 / 7.0])
    expected_actual = np.array([1.0, 0.0])
    expected_brier = float(np.mean((expected_predicted - expected_actual) ** 2))
    assert bt.brier_score == pytest.approx(expected_brier)
    assert bt.n_observations == 2

    # The static backtest uses the frozen seed pool for *both* steps
    # (never sees step 0's own residual), so its step-1 predicted
    # probability stays at the seed pool's 3/6 = 0.5 instead of the
    # adaptive version's updated 4/7 -- the two must genuinely differ,
    # not just be computed by different code paths that happen to agree.
    static_bt = backtest_threshold_calibration(fake_result, X_test, y_test, relative_offsets=(0.0,))
    expected_static_brier = float(np.mean((np.array([0.5, 0.5]) - expected_actual) ** 2))
    assert static_bt.brier_score == pytest.approx(expected_static_brier)
    assert bt.brier_score != pytest.approx(static_bt.brier_score)


def test_backtest_threshold_calibration_adaptive_rejects_mismatched_lengths():
    fake_result = _fake_conformal_result(np.array([-1.0, 0.0, 1.0]))
    X_test = np.zeros((3, 1))
    y_test = np.array([0.0, 0.0])  # length 2, mismatched with X_test's 3 rows
    with pytest.raises(ValueError):
        backtest_threshold_calibration_adaptive(fake_result, X_test, y_test)


def test_backtest_threshold_calibration_adaptive_bounded_window_truncates_seed_pool():
    # Same hand-computed scenario as the unbounded test above, but with a
    # window (3) smaller than the 6-value seed pool -- the seed itself must
    # be truncated to the most recent 3 values immediately (see the
    # function's own docstring for why: otherwise a bounded window never
    # actually binds on a pool that starts above it).
    seed_residuals = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    fake_result = _fake_conformal_result(seed_residuals)
    X_test = np.zeros((1, 1))
    y_test = np.array([10.0])

    bt = backtest_threshold_calibration_adaptive(
        fake_result, X_test, y_test, relative_offsets=(0.0,), window=3
    )
    # window=3 truncates the 6-value seed to its last 3 entries:
    # [-2, -1, -0.5, 0.5, 1, 2][-3:] == [0.5, 1.0, 2.0] -- all strictly
    # greater than 0, so P(pool > 0) = 3/3 = 1.0 for this single step.
    # True y[0] = 10.0 (> threshold 0) -> actually exceeded too, so
    # predicted and actual agree exactly and the Brier score is 0.
    assert bt.brier_score == pytest.approx(0.0)


def test_backtest_threshold_calibration_adaptive_on_real_wti_series():
    """The real, honest check: does letting the residual pool adapt online
    change (for better or worse -- measured, not assumed) the probability
    calibration on real WTI history, relative to the static version?"""
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    result = fit_calibrated_naive_forecaster(series, lookback=5, alpha=0.1)
    (_X_train, _y_train, _X_cal, _y_cal, X_test, y_test, _normalizer, _period) = prepare_split_from_series(
        series, lookback=5, period=5, label="wti_spot"
    )

    bt = backtest_threshold_calibration_adaptive(result, X_test, y_test, window=75)

    assert bt.n_test_points == len(X_test)
    assert bt.n_observations == bt.n_test_points * 7  # default 7 relative_offsets
    # Same loose-bound discipline as the static version's own real-data
    # test: a real measured result on real data, not an exact pin, and not
    # assumed to be *better* than the static version just because it's
    # fancier -- only that it stays well inside a sane, working range.
    assert bt.brier_score < 0.15
    assert bt.mean_calibration_error < 0.10
    for b in bt.reliability_bins:
        assert 0.0 <= b["mean_predicted_probability"] <= 1.0
        assert 0.0 <= b["empirical_exceedance_rate"] <= 1.0
        assert b["count"] > 0
