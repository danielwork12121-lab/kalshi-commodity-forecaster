"""Tests for `select_adaptive_pool_window`/`format_adaptive_pool_window_selection`
(v0.23) -- this project's own "What's next" item 3: a principled,
held-out-data way to pick the adaptive probability pool's `window`,
mirroring conformal-forecast's `auto-window`/`cv-window-select` for ACI's
interval case. Covers the new `_make_rolling_folds`/`_select_best_window`
fold-boundary helpers (pure, no model dependency) and the real-data
integration path, same two-layer split conformal-forecast's own CV tests
use.
"""
from __future__ import annotations

import numpy as np
import pytest

from commodity_forecast.data import fetch_wti_daily
from commodity_forecast.forecast import fit_calibrated_naive_forecaster
from commodity_forecast.probability import (
    AdaptivePoolWindowSelectionResult,
    _make_rolling_folds,
    _select_best_window,
    select_adaptive_pool_window,
)
from forecasting.experiment import prepare_split_from_series

# ---------------------------------------------------------------------------
# Layer 1: pure fold-boundary / selection-rule logic, no model or data
# dependency -- same split conformal-forecast's own test file uses for its
# analogous helpers.
# ---------------------------------------------------------------------------


def test_make_rolling_folds_cover_range_exactly_with_no_gap_or_overlap():
    folds = _make_rolling_folds(n=1000, n_folds=5, min_initial=300, min_fold_size=100)
    assert folds[0][0] == 300
    assert folds[-1][1] == 1000
    for (_, end_a), (start_b, _) in zip(folds, folds[1:]):
        assert end_a == start_b


def test_make_rolling_folds_too_small_returns_none():
    assert _make_rolling_folds(n=10, n_folds=4, min_initial=8, min_fold_size=5) is None


def test_make_rolling_folds_fewer_achievable_folds_than_requested():
    folds = _make_rolling_folds(n=20, n_folds=6, min_initial=5, min_fold_size=5)
    assert folds is not None
    assert len(folds) == 3


def test_make_rolling_folds_invalid_args_raise():
    with pytest.raises(ValueError):
        _make_rolling_folds(n=20, n_folds=0, min_initial=5, min_fold_size=5)
    with pytest.raises(ValueError):
        _make_rolling_folds(n=20, n_folds=2, min_initial=0, min_fold_size=5)


def test_select_best_window_picks_smallest_score():
    windows = [5, 10, None]
    score = {5: 0.05, 10: 0.01, None: 0.08}
    tiebreak = {5: 0.1, 10: 0.1, None: 0.1}
    assert _select_best_window(windows, score, tiebreak) == 10


def test_select_best_window_tiebreaks_by_smaller_secondary_score():
    windows = [5, 10]
    score = {5: 0.02, 10: 0.02}  # tied on primary
    tiebreak = {5: 0.09, 10: 0.03}
    assert _select_best_window(windows, score, tiebreak) == 10


def test_select_best_window_falls_back_to_larger_window():
    windows = [5, 10, None]
    score = {5: 0.02, 10: 0.02, None: 0.02}  # all tied
    tiebreak = {5: 0.1, 10: 0.1, None: 0.1}  # still tied
    assert _select_best_window(windows, score, tiebreak) is None  # unbounded wins the final tiebreak


def test_select_best_window_requires_at_least_one_candidate():
    with pytest.raises(ValueError):
        _select_best_window([], {}, {})


# ---------------------------------------------------------------------------
# Layer 2: real-data integration, via select_adaptive_pool_window. Uses
# NaiveForecaster (via fit_calibrated_naive_forecaster) rather than the
# LSTM -- this feature only cares about the residual-pool/fold machinery,
# which is model-agnostic, and NaiveForecaster keeps the suite fast.
# ---------------------------------------------------------------------------

CANDIDATES = [20, 75, 200, None]


@pytest.fixture(scope="module")
def wti_split():
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    result = fit_calibrated_naive_forecaster(series, lookback=5, alpha=0.1)
    X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, _period = prepare_split_from_series(
        series, lookback=5, period=5, label="wti_spot"
    )
    return result, X_cal, y_cal, X_test, y_test


def test_runs_and_reports_every_candidate(wti_split):
    result, X_cal, y_cal, X_test, y_test = wti_split
    selection = select_adaptive_pool_window(result, X_cal, y_cal, X_test, y_test, windows=CANDIDATES, n_folds=5)

    assert isinstance(selection, AdaptivePoolWindowSelectionResult)
    assert [c.window for c in selection.candidates] == CANDIDATES
    assert selection.selected_window in CANDIDATES
    assert selection.best_test_window in CANDIDATES
    assert selection.n_folds >= 1
    assert len(selection.fold_bounds) == selection.n_folds
    for c in selection.candidates:
        assert c.mean_fold_calibration_error >= 0.0
        assert c.mean_fold_brier >= 0.0
        assert 0.0 <= c.test_calibration_error
        assert 0.0 <= c.test_brier


def test_folds_partition_the_calibration_set(wti_split):
    result, X_cal, y_cal, X_test, y_test = wti_split
    selection = select_adaptive_pool_window(result, X_cal, y_cal, X_test, y_test, windows=[75, None], n_folds=5)
    assert selection.fold_bounds[0][0] >= 1
    for (_, end_a), (start_b, _) in zip(selection.fold_bounds, selection.fold_bounds[1:]):
        assert end_a == start_b


def test_is_deterministic():
    series = fetch_wti_daily(refresh=False).to_numpy(dtype=np.float64)
    result = fit_calibrated_naive_forecaster(series, lookback=5, alpha=0.1)
    X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, _period = prepare_split_from_series(
        series, lookback=5, period=5, label="wti_spot"
    )
    a = select_adaptive_pool_window(result, X_cal, y_cal, X_test, y_test, windows=CANDIDATES, n_folds=5)
    b = select_adaptive_pool_window(result, X_cal, y_cal, X_test, y_test, windows=CANDIDATES, n_folds=5)
    assert a.selected_window == b.selected_window
    assert [c.mean_fold_calibration_error for c in a.candidates] == [c.mean_fold_calibration_error for c in b.candidates]


def test_selection_never_uses_the_test_set(wti_split):
    """The selection criterion (mean_fold_calibration_error/mean_fold_brier)
    is computed entirely from X_cal/y_cal -- shrinking the test set passed
    in must not change those numbers at all, only the honest post-hoc
    `test_*` fields that are reported alongside but never used to select.
    """
    result, X_cal, y_cal, X_test, y_test = wti_split
    full = select_adaptive_pool_window(result, X_cal, y_cal, X_test, y_test, windows=[75, None], n_folds=5)
    half = select_adaptive_pool_window(
        result, X_cal, y_cal, X_test[: len(X_test) // 2], y_test[: len(y_test) // 2], windows=[75, None], n_folds=5
    )
    assert [c.mean_fold_calibration_error for c in full.candidates] == [
        c.mean_fold_calibration_error for c in half.candidates
    ]
    assert [c.mean_fold_brier for c in full.candidates] == [c.mean_fold_brier for c in half.candidates]
    assert full.selected_window == half.selected_window


def test_too_small_calibration_set_raises():
    # A tiny synthetic calibration set, reserving 90% of it as the first
    # fold's seed, can't support even one fold at min_fold_frac=0.5.
    class _ConstantModel:
        def predict(self, X):
            return np.zeros((len(X), 1))

    class _FakeConformal:
        model = _ConstantModel()

    from commodity_forecast.forecast import CalibratedForecastResult

    fake_result = CalibratedForecastResult(
        conformal_forecaster=_FakeConformal(),
        normalizer_mean=0.0,
        normalizer_std=1.0,
        diagnostics={},
        cal_residuals_normalized=np.zeros(10),
    )
    X_cal, y_cal = np.zeros((10, 1)), np.zeros(10)
    X_test, y_test = np.zeros((5, 1)), np.zeros(5)
    with pytest.raises(ValueError):
        select_adaptive_pool_window(
            fake_result, X_cal, y_cal, X_test, y_test, windows=[3], min_initial_frac=0.9, min_fold_frac=0.5
        )


def test_invalid_frac_args_raise(wti_split):
    result, X_cal, y_cal, X_test, y_test = wti_split
    with pytest.raises(ValueError):
        select_adaptive_pool_window(result, X_cal, y_cal, X_test, y_test, windows=[75], min_initial_frac=0.0)
    with pytest.raises(ValueError):
        select_adaptive_pool_window(result, X_cal, y_cal, X_test, y_test, windows=[75], min_fold_frac=1.0)


def test_selected_test_result_matches_a_direct_call(wti_split):
    """The honest post-hoc check must report the exact same real test-set
    numbers `backtest_threshold_calibration_adaptive` would for the
    selected window, called directly -- not a recomputed or approximated
    version.
    """
    from commodity_forecast.probability import backtest_threshold_calibration_adaptive

    result, X_cal, y_cal, X_test, y_test = wti_split
    selection = select_adaptive_pool_window(result, X_cal, y_cal, X_test, y_test, windows=CANDIDATES, n_folds=5)
    direct = backtest_threshold_calibration_adaptive(result, X_test, y_test, window=selection.selected_window)
    assert selection.selected_test_result.mean_calibration_error == direct.mean_calibration_error
    assert selection.selected_test_result.brier_score == direct.brier_score


def test_format_includes_every_window_and_the_selection_marker(wti_split):
    from commodity_forecast.probability import format_adaptive_pool_window_selection

    result, X_cal, y_cal, X_test, y_test = wti_split
    selection = select_adaptive_pool_window(result, X_cal, y_cal, X_test, y_test, windows=[75, None], n_folds=5)
    text = format_adaptive_pool_window_selection(selection)
    assert "Adaptive-pool rolling-origin CV window selection" in text
    assert "| 75" in text
    assert "unbounded (growing buffer)" in text
    assert "<- selected" in text
