"""Converting a calibrated forecast into a Kalshi-style threshold probability.

`forecast.py` produces a point forecast plus a calibrated *interval* with a
measured (not just asserted) coverage guarantee. A Kalshi-style market asks
a sharper question than an interval answers: not "is the true price in
[lower, upper] with ~90% probability?" but "what is P(price > K) for some
specific strike K, by some specific date?" -- a single number between 0 and
1, for a threshold the interval alone doesn't speak to directly.

This module answers that question without inventing a parametric
distribution (e.g. "assume the forecast error is Gaussian") the data has no
reason to support. Instead it reuses the one piece of real, held-out signal
`fit_calibrated_naive_forecaster` already computes: the model's *signed*
residuals on the calibration split (`CalibratedForecastResult
.cal_residuals_normalized`) -- the same split conformal prediction's own
coverage guarantee rests on, just not thrown away as raw numbers this time.
Treating those residuals as a Monte-Carlo sample from "the forecast error
distribution" (an approximation, exactly as sound or unsound as split
conformal's own exchangeability assumption -- see `forecasting.conformal`'s
own honesty note) gives:

    P(y > K) ≈ P(point_forecast + residual > K)
             = P(residual > K - point_forecast)
             = (fraction of calibration residuals exceeding K - point_forecast)

i.e. the empirical CDF of the calibration residuals, evaluated at one point
per test-set forecast. This is cheap, has no extra hyperparameters, and --
critically -- is backtestable: for the same question asked over and over
across real history, did "70% probability" events actually happen about
70% of the time? `backtest_threshold_calibration` below measures exactly
that, honestly, including where it's wrong.

Known limitation, stated plainly: this treats the calibration-residual
sample as the *entire* predictive distribution for every test point, which
assumes the residual distribution doesn't itself depend on the forecast
level or on time -- a real simplification for a non-stationary 40-year
commodity series (the same simplification split conformal's own fixed
q_hat already makes for the interval case; see `forecasting/adaptive.py`'s
online alternative, not yet wired into this module -- see README "What's
next").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from commodity_forecast.forecast import CalibratedForecastResult


def exceedance_probability(
    point_forecast_normalized: np.ndarray,
    cal_residuals_normalized: np.ndarray,
    threshold_normalized: np.ndarray | float,
) -> np.ndarray:
    """P(true value > threshold) for each point forecast, in normalized units.

    `point_forecast_normalized` and `threshold_normalized` broadcast against
    each other (pass a scalar threshold for "the same K for every test
    point", or an array the same length as `point_forecast_normalized` for
    a different, e.g. forecast-relative, K per point). `cal_residuals_normalized`
    is the empirical sample the probability is estimated from -- see the
    module docstring for the exact formula and its honesty caveats.
    """
    point_forecast_normalized = np.asarray(point_forecast_normalized, dtype=np.float64).reshape(-1)
    threshold_normalized = np.asarray(threshold_normalized, dtype=np.float64)
    needed_excess = threshold_normalized - point_forecast_normalized  # residual must exceed this
    residuals = np.asarray(cal_residuals_normalized, dtype=np.float64).reshape(1, -1)
    # (n_points, n_cal_residuals) comparison, averaged over the residual axis.
    return np.mean(residuals > needed_excess.reshape(-1, 1), axis=1)


@dataclass
class ThresholdBacktestResult:
    relative_offsets: tuple[float, ...]
    n_test_points: int
    n_observations: int  # n_test_points * len(relative_offsets)
    brier_score: float
    # Reliability bins: predicted probability bucketed into deciles, each
    # bucket's mean predicted probability vs. its actual exceedance rate --
    # a well-calibrated method has these nearly equal in every bucket with
    # enough observations to be meaningful.
    reliability_bins: list[dict]
    mean_calibration_error: float  # observation-count-weighted mean |predicted - actual| across bins


def backtest_threshold_calibration(
    result: CalibratedForecastResult,
    X_test: np.ndarray,
    y_test: np.ndarray,
    relative_offsets: tuple[float, ...] = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10),
    n_bins: int = 10,
) -> ThresholdBacktestResult:
    """Backtest `exceedance_probability` against real, held-out history.

    For every point in the test split and every relative offset in
    `relative_offsets` (e.g. 0.05 = "5% above this day's own point
    forecast"), computes the predicted P(true price > that threshold) and
    checks it against what actually happened on that real historical day --
    this is the honest way to check a probability-conversion method: not
    "does it look plausible" but "across many real, independent
    questions, did events predicted at ~p% probability happen about p% of
    the time". Uses forecast-relative thresholds (rather than a handful of
    fixed dollar strikes) so every one of the ~1,900 real test-split days
    contributes a genuine, non-cherry-picked question, at a range of
    difficulties (a threshold just above today's forecast is closer to a
    coin flip; 10% above is a real tail bet).

    `X_test`/`y_test` must be the *same* test split `result` was evaluated
    against (i.e. the `X_test`/`y_test` `fit_calibrated_naive_forecaster`
    used internally) -- this function takes them explicitly rather than
    re-deriving them, since `CalibratedForecastResult` doesn't itself keep
    a reference to the split arrays.
    """
    prediction = result.conformal_forecaster.predict(X_test)
    point_normalized = prediction.point.reshape(-1)
    y_test_normalized = np.asarray(y_test, dtype=np.float64).reshape(-1)
    point_usd = point_normalized * result.normalizer_std + result.normalizer_mean
    y_test_usd = y_test_normalized * result.normalizer_std + result.normalizer_mean

    all_predicted = []
    all_actual = []
    for offset in relative_offsets:
        threshold_usd = point_usd * (1.0 + offset)
        threshold_normalized = (threshold_usd - result.normalizer_mean) / result.normalizer_std
        predicted_prob = exceedance_probability(
            point_normalized, result.cal_residuals_normalized, threshold_normalized
        )
        actual_exceeded = (y_test_usd > threshold_usd).astype(np.float64)
        all_predicted.append(predicted_prob)
        all_actual.append(actual_exceeded)

    predicted = np.concatenate(all_predicted)
    actual = np.concatenate(all_actual)

    brier = float(np.mean((predicted - actual) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(predicted, bin_edges[1:-1]), 0, n_bins - 1)
    reliability_bins = []
    weighted_abs_error_sum = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted = float(predicted[mask].mean())
        mean_actual = float(actual[mask].mean())
        reliability_bins.append(
            {
                "bin_range": (float(bin_edges[b]), float(bin_edges[b + 1])),
                "count": count,
                "mean_predicted_probability": mean_predicted,
                "empirical_exceedance_rate": mean_actual,
            }
        )
        weighted_abs_error_sum += count * abs(mean_predicted - mean_actual)

    mean_calibration_error = weighted_abs_error_sum / len(predicted)

    return ThresholdBacktestResult(
        relative_offsets=relative_offsets,
        n_test_points=int(len(point_normalized)),
        n_observations=int(len(predicted)),
        brier_score=brier,
        reliability_bins=reliability_bins,
        mean_calibration_error=mean_calibration_error,
    )
