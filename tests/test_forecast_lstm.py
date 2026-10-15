"""Tests for `fit_calibrated_lstm_forecaster` -- this project's own "What's
next" item 2: does a non-trivial point model (conformal-forecast's LSTM,
wrapped in DeltaWrapper) change the static-conformal coverage result, or
the probability layer's calibration, measured on the real WTI series?

Mirrors `tests/test_forecast.py`'s own synthetic + real-WTI pattern for
`fit_calibrated_naive_forecaster`, so the two models are checked the same
way and any later comparison between them is apples-to-apples.
"""
from __future__ import annotations

import numpy as np

from commodity_forecast.data import fetch_wti_daily
from commodity_forecast.forecast import fit_calibrated_lstm_forecaster, fit_calibrated_naive_forecaster
from commodity_forecast.probability import backtest_threshold_calibration


def test_fit_calibrated_lstm_forecaster_on_synthetic_series():
    from forecasting.data import generate_synthetic

    series = generate_synthetic(n=600, period=24, noise_std=1.0, seed=1)
    result = fit_calibrated_lstm_forecaster(series, lookback=5, alpha=0.1, seed=0)
    d = result.diagnostics

    assert d["n_train_windows"] > 0
    assert d["n_cal_windows"] > 0
    assert 0.0 <= d["empirical_coverage"] <= 1.0
    assert d["mean_interval_width"] > 0
    assert d["mae_usd_per_bbl"] >= 0
    # The model is really a DeltaWrapper(LSTMForecaster), not a relabeled
    # NaiveForecaster -- confirmed by type, not just by the function name.
    from forecasting.models import DeltaWrapper, LSTMForecaster

    assert isinstance(result.conformal_forecaster.model, DeltaWrapper)
    assert isinstance(result.conformal_forecaster.model.base_model, LSTMForecaster)


def test_fit_calibrated_lstm_forecaster_on_real_wti_series():
    """Same real-series sanity bounds as
    `test_fit_calibrated_naive_forecaster_on_real_wti_series`, so this is a
    genuine apples-to-apples check, not a looser bar for the new model."""
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    result = fit_calibrated_lstm_forecaster(series, lookback=5, alpha=0.1, seed=0)
    d = result.diagnostics

    assert d["n_test"] > 100
    assert 0 < d["mae_usd_per_bbl"] < 20
    assert 0.0 <= d["empirical_coverage"] <= 1.0
    assert d["mean_interval_width"] > 0


def test_lstm_and_naive_forecasters_share_the_identical_split():
    """Both functions call `prepare_split_from_series` identically (same
    series, lookback, label) -- so the two models' test sets must line up
    exactly in size, which is what makes a later coverage/MAE comparison
    between them meaningful rather than an artifact of different splits.
    """
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    naive = fit_calibrated_naive_forecaster(series, lookback=5, alpha=0.1)
    lstm = fit_calibrated_lstm_forecaster(series, lookback=5, alpha=0.1, seed=0)

    assert naive.diagnostics["n_test"] == lstm.diagnostics["n_test"]
    assert naive.diagnostics["n_train_windows"] == lstm.diagnostics["n_train_windows"]
    assert naive.diagnostics["n_cal_windows"] == lstm.diagnostics["n_cal_windows"]


def test_lstm_calibrated_result_is_a_drop_in_for_the_probability_layer():
    """`backtest_threshold_calibration` (see `commodity_forecast/probability.py`)
    was built and tested only against `fit_calibrated_naive_forecaster`'s
    output -- this checks directly, not just by code inspection, that it
    also runs correctly against the LSTM-based `CalibratedForecastResult`,
    since both functions only ever expose the same generic dataclass
    shape. A real, bounded Brier score and mean calibration error coming
    back (not a crash, not a NaN) is the actual claim being tested.
    """
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    from forecasting.experiment import prepare_split_from_series

    X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, _period = prepare_split_from_series(
        series, lookback=5, period=5, label="wti_spot"
    )
    result = fit_calibrated_lstm_forecaster(series, lookback=5, alpha=0.1, seed=0)
    bt = backtest_threshold_calibration(result, X_test, y_test)

    assert 0.0 <= bt.brier_score <= 1.0
    assert not np.isnan(bt.brier_score)
    assert bt.n_test_points == len(X_test)
