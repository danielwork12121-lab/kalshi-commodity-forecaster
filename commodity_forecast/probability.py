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

Known limitation, stated plainly: `backtest_threshold_calibration` below
treats the calibration-residual sample as the *entire* predictive
distribution for every test point, which assumes the residual distribution
doesn't itself depend on the forecast level or on time -- a real
simplification for a non-stationary 40-year commodity series (the same
simplification split conformal's own fixed q_hat already makes for the
interval case).

`backtest_threshold_calibration_adaptive` (since 2026-10-15) answers the
obvious follow-up question `forecasting/adaptive.py`'s own module docstring
raises for the interval case: does letting the residual pool itself adapt
online -- the same idea that measurably closed conformal-forecast's airline
coverage gap (see that repo's README) -- also improve the *probability*
estimate, not just the interval? It's wired in directly here (not by
reusing `forecasting.adaptive.SlidingWindowAdaptiveConformalForecaster`,
which only ever tracks *absolute* residuals internally -- enough for a
symmetric interval, not enough to recover the signed empirical CDF a
threshold probability needs), replaying the same strict step-by-step
ordering discipline that class uses: score step t's questions against the
pool *as it stood before* step t, then fold step t's own signed residual in
only afterward, so no step ever sees its own answer early.
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


def _score_predicted_vs_actual(
    predicted: np.ndarray,
    actual: np.ndarray,
    relative_offsets: tuple[float, ...],
    n_bins: int,
) -> "ThresholdBacktestResult":
    """Shared scoring tail for both the static and the online/adaptive
    backtest below: Brier score + count-weighted reliability bins, from a
    flat array of (predicted_probability, actually_exceeded) pairs. Pulled
    out so the two backtests can't silently drift apart on how they define
    "well calibrated" -- only how the predictions themselves are produced
    differs between them.
    """
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
        n_test_points=0,  # filled in by the caller, which knows the real value
        n_observations=int(len(predicted)),
        brier_score=brier,
        reliability_bins=reliability_bins,
        mean_calibration_error=mean_calibration_error,
    )


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

    bt = _score_predicted_vs_actual(predicted, actual, relative_offsets, n_bins)
    bt.n_test_points = int(len(point_normalized))
    return bt


def backtest_threshold_calibration_adaptive(
    result: CalibratedForecastResult,
    X_test: np.ndarray,
    y_test: np.ndarray,
    relative_offsets: tuple[float, ...] = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10),
    window: int | None = None,
    n_bins: int = 10,
) -> ThresholdBacktestResult:
    """Same backtest as `backtest_threshold_calibration`, but with an
    *online*, ACI-style signed-residual pool instead of the frozen
    calibration sample -- the probability-estimation analogue of
    `forecasting.adaptive.SlidingWindowAdaptiveConformalForecaster`.

    `window=None` (default): an unbounded growing buffer -- every observed
    signed residual (calibration, then every test step seen so far) stays
    in the pool forever. `window=k`: a true bounded sliding window of the
    most recent `k` signed residuals, same semantics as the interval-ACI
    class's own `window` parameter, including the same "seed pool larger
    than window" truncation it needs (see that class's docstring for the
    real bug this guards against).

    Strict step ordering, matching `SlidingWindowAdaptiveConformalForecaster
    .predict_sequential` exactly: for each test step t, (1) score every
    relative-offset question for step t against the pool *as it stood
    before* step t, (2) only afterward fold step t's own signed residual
    into the pool for step t+1 onward. No step ever has access to its own
    answer when it's scored.

    `X_test`/`y_test` must be the same test split `result` was evaluated
    against, same contract as `backtest_threshold_calibration`.
    """
    model = result.conformal_forecaster.model
    point_normalized = np.asarray(model.predict(X_test), dtype=np.float64).reshape(-1)
    y_test_normalized = np.asarray(y_test, dtype=np.float64).reshape(-1)
    if len(point_normalized) != len(y_test_normalized):
        raise ValueError("X_test and y_test must have the same length")

    # Mutable working copy -- never mutates result.cal_residuals_normalized,
    # so re-running this (e.g. from a test) always starts from the same
    # seed pool. Same seed-truncation rule as the interval-ACI class: if
    # window is smaller than the calibration pool itself, a pool that
    # starts above `window` must be truncated down to it immediately, or a
    # bounded window never actually binds (append one, evict one, net zero
    # size change every step).
    pool = list(np.asarray(result.cal_residuals_normalized, dtype=np.float64))
    if window is not None and len(pool) > window:
        pool = pool[-window:]

    all_predicted = []
    all_actual = []
    for t in range(len(point_normalized)):
        point_t = point_normalized[t]
        point_t_usd = point_t * result.normalizer_std + result.normalizer_mean
        y_t_usd = y_test_normalized[t] * result.normalizer_std + result.normalizer_mean

        pool_arr = np.asarray(pool, dtype=np.float64)
        for offset in relative_offsets:
            threshold_usd = point_t_usd * (1.0 + offset)
            threshold_normalized = (threshold_usd - result.normalizer_mean) / result.normalizer_std
            needed_excess = threshold_normalized - point_t
            predicted_prob = float(np.mean(pool_arr > needed_excess))
            actual_exceeded = 1.0 if y_t_usd > threshold_usd else 0.0
            all_predicted.append(predicted_prob)
            all_actual.append(actual_exceeded)

        # Fold this step's own signed residual in only now, after every
        # threshold question for step t has already been scored against
        # the pool as it stood *before* this step.
        pool.append(float(y_test_normalized[t] - point_t))
        if window is not None and len(pool) > window:
            pool.pop(0)

    predicted = np.array(all_predicted)
    actual = np.array(all_actual)

    bt = _score_predicted_vs_actual(predicted, actual, relative_offsets, n_bins)
    bt.n_test_points = int(len(point_normalized))
    return bt
