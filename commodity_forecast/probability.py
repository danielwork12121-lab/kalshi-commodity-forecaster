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


def _run_adaptive_pool_backtest(
    point_normalized: np.ndarray,
    y_normalized: np.ndarray,
    seed_pool: list[float],
    window: int | None,
    normalizer_mean: float,
    normalizer_std: float,
    relative_offsets: tuple[float, ...],
    n_bins: int,
) -> ThresholdBacktestResult:
    """Shared online-residual-pool backtest loop, used both by
    `backtest_threshold_calibration_adaptive` (real test set, seeded from
    the full calibration sample) and `select_adaptive_pool_window` (v0.23 --
    a rolling-origin CV fold *within* the calibration set, seeded from only
    that fold's own prefix) -- pulled out so the two can never silently
    diverge on the strict step-ordering discipline that makes this backtest
    honest: score step t against the pool *as it stood before* step t, and
    only afterward fold step t's own signed residual in for step t+1
    onward. No step ever has access to its own answer when it's scored.

    `seed_pool` is the caller's own already-decided starting pool (the full
    calibration residual sample for the real backtest; a fold-specific
    prefix of it for CV) -- this function only ever appends to / truncates
    a *copy* of it (`list(seed_pool)`), never the caller's own list.
    """
    point_normalized = np.asarray(point_normalized, dtype=np.float64).reshape(-1)
    y_normalized = np.asarray(y_normalized, dtype=np.float64).reshape(-1)
    if len(point_normalized) != len(y_normalized):
        raise ValueError("point_normalized and y_normalized must have the same length")

    # Same seed-truncation rule as the interval-ACI class this mirrors: if
    # window is smaller than the seed pool, the seed itself must be
    # truncated down to it immediately, or a bounded window never actually
    # binds (append one, evict one, net zero size change every step).
    pool = list(seed_pool)
    if window is not None and len(pool) > window:
        pool = pool[-window:]

    all_predicted = []
    all_actual = []
    for t in range(len(point_normalized)):
        point_t = point_normalized[t]
        point_t_usd = point_t * normalizer_std + normalizer_mean
        y_t_usd = y_normalized[t] * normalizer_std + normalizer_mean

        pool_arr = np.asarray(pool, dtype=np.float64)
        for offset in relative_offsets:
            threshold_usd = point_t_usd * (1.0 + offset)
            threshold_normalized = (threshold_usd - normalizer_mean) / normalizer_std
            needed_excess = threshold_normalized - point_t
            predicted_prob = float(np.mean(pool_arr > needed_excess))
            actual_exceeded = 1.0 if y_t_usd > threshold_usd else 0.0
            all_predicted.append(predicted_prob)
            all_actual.append(actual_exceeded)

        # Fold this step's own signed residual in only now, after every
        # threshold question for step t has already been scored against
        # the pool as it stood *before* this step.
        pool.append(float(y_normalized[t] - point_t))
        if window is not None and len(pool) > window:
            pool.pop(0)

    predicted = np.array(all_predicted)
    actual = np.array(all_actual)

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
    class's own `window` parameter.

    `X_test`/`y_test` must be the same test split `result` was evaluated
    against, same contract as `backtest_threshold_calibration`. The actual
    online-pool loop (step ordering, truncation) is shared with
    `select_adaptive_pool_window`'s CV folds via `_run_adaptive_pool_backtest`.
    """
    model = result.conformal_forecaster.model
    point_normalized = np.asarray(model.predict(X_test), dtype=np.float64).reshape(-1)
    return _run_adaptive_pool_backtest(
        point_normalized,
        y_test,
        seed_pool=list(np.asarray(result.cal_residuals_normalized, dtype=np.float64)),
        window=window,
        normalizer_mean=result.normalizer_mean,
        normalizer_std=result.normalizer_std,
        relative_offsets=relative_offsets,
        n_bins=n_bins,
    )


# ---------------------------------------------------------------------------
# Adaptive-pool window selection (v0.23) -- this project's own "What's next"
# item 3: "a principled way to pick the probability pool's own window (e.g.
# a held-out validation slice), mirroring conformal-forecast's own
# auto-window/cv-window-select for ACI's interval case" -- the obvious
# follow-up now that `backtest_threshold_calibration_adaptive` (item 5) and
# a stronger point model (item 2) are both shipped. Mirrors
# conformal-forecast's `run_cv_window_selection` method (rolling-origin CV
# over held-out data, never touching the real test set to select) using
# this project's own calibration set and its own online-pool backtest loop,
# rather than importing conformal-forecast's private experiment-module
# helpers directly (this project only ever depends on conformal-forecast's
# public API -- see `forecast.py`'s own module docstring).
# ---------------------------------------------------------------------------


def _make_rolling_folds(n: int, n_folds: int, min_initial: int, min_fold_size: int) -> list[tuple[int, int]] | None:
    """Expanding-window ("blocked", chronological) folds over indices
    `[0, n)` -- the same fold-boundary logic conformal-forecast's own
    `run_cv_window_selection` uses (reimplemented here, not imported,
    since it's a private helper in that project's experiment module and
    this project only depends on conformal-forecast's public API).

    Fold 0's seed is `[0, min_initial)`, its validation chunk is the next
    `fold_size` points; fold 1's seed extends to cover fold 0's validation
    chunk too, and so on -- every index from `min_initial` to `n` is
    covered by exactly one fold's validation chunk, no gap, no overlap.
    Returns `None` if `n` is too small to form even one fold of at least
    `min_fold_size` validation points after reserving `min_initial` points
    for the first seed. `n_folds` is a request, not a guarantee.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be a positive integer")
    if min_initial < 1 or min_fold_size < 1:
        raise ValueError("min_initial and min_fold_size must be positive integers")

    usable = n - min_initial
    if usable < min_fold_size:
        return None

    actual_folds = min(n_folds, usable // min_fold_size)
    fold_size = usable // actual_folds

    folds = []
    start = min_initial
    for k in range(actual_folds):
        end = start + fold_size if k < actual_folds - 1 else n
        folds.append((start, end))
        start = end
    return folds


def _select_best_window(windows: list[int | None], score_by_window: dict, tiebreak_by_window: dict) -> int | None:
    """Pick the window minimizing `score_by_window` (rounded, to treat
    near-ties as ties rather than letting floating-point noise pick a
    winner), tie-broken by the smaller `tiebreak_by_window`, and -- failing
    that -- the larger window (fewer residuals makes for a noisier
    probability estimate, so prefer the option that discards less
    information; `None`, unbounded, is "infinitely large" here). Same
    three-level tie-break structure as conformal-forecast's own
    `_select_best_window`, adapted to this project's two probability
    metrics (mean calibration error primary, Brier score secondary)
    instead of that project's (coverage gap, interval width).
    """
    if not windows:
        raise ValueError("Need at least one candidate window to select from")

    def key(w):
        size = float("inf") if w is None else w
        return (round(score_by_window[w], 10), tiebreak_by_window[w], -size)

    return min(windows, key=key)


@dataclass
class AdaptivePoolWindowCandidate:
    window: int | None  # None = unbounded growing buffer
    mean_fold_calibration_error: float  # mean across folds -- primary selection criterion
    mean_fold_brier: float  # mean across folds -- tie-break criterion
    test_calibration_error: float  # real test-set result, honest post-hoc check only
    test_brier: float  # real test-set result, honest post-hoc check only


@dataclass
class AdaptivePoolWindowSelectionResult:
    n_folds: int
    fold_bounds: list[tuple[int, int]]
    relative_offsets: tuple[float, ...]
    candidates: list[AdaptivePoolWindowCandidate]
    selected_window: int | None  # chosen using the fold columns only
    best_test_window: int | None  # oracle: smallest real test-set calibration error -- comparison only
    static_test: ThresholdBacktestResult  # reference, real test set, frozen calibration pool
    selected_test_result: ThresholdBacktestResult  # real test-set result for `selected_window`


def select_adaptive_pool_window(
    result: CalibratedForecastResult,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    windows: list[int | None],
    relative_offsets: tuple[float, ...] = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10),
    n_folds: int = 5,
    min_initial_frac: float = 0.3,
    min_fold_frac: float = 0.1,
    n_bins: int = 10,
) -> AdaptivePoolWindowSelectionResult:
    """Select `backtest_threshold_calibration_adaptive`'s own `window` via
    rolling-origin cross-validation over the calibration set, instead of
    sweeping it by eye the way `active-project-kalshi.md`'s progress log
    did for the adaptive-pool feature itself.

    Method, mirroring conformal-forecast's `run_cv_window_selection`
    exactly: `X_cal`/`y_cal` are partitioned into folds via
    `_make_rolling_folds`. For each candidate window, each fold's own
    online-pool backtest (`_run_adaptive_pool_backtest`, the same loop
    `backtest_threshold_calibration_adaptive` uses) is seeded from an
    *expanding* prefix of the calibration residuals (`[0:train_end)` --
    this project's calibration set is large enough, at ~1,900 windows,
    that a fixed expanding seed is a reasonable default; conformal-forecast's
    own `sliding`-seed alternative is not reproduced here since this
    project has not measured whether it would help on this series, unlike
    that project's own README comparison) and scored against that fold's
    own validation chunk -- never the real test set. The per-window score
    is the MEAN of `mean_calibration_error` across folds, tie-broken by
    the mean Brier score and then the larger window.

    `X_test`/`y_test` are used only afterward, via the already-tested
    `backtest_threshold_calibration_adaptive`, to report each candidate's
    real test-set result as an honest post-hoc check -- the selection
    itself never sees them, same discipline as conformal-forecast's own
    window-selection functions.

    Raises ValueError if the calibration set is too small to form even one
    fold of `min_fold_frac * n_cal` points after reserving
    `min_initial_frac * n_cal` points for the first fold's seed.
    """
    if not (0.0 < min_initial_frac < 1.0) or not (0.0 < min_fold_frac < 1.0):
        raise ValueError("min_initial_frac and min_fold_frac must be in (0, 1)")

    model = result.conformal_forecaster.model
    point_cal_normalized = np.asarray(model.predict(X_cal), dtype=np.float64).reshape(-1)
    y_cal_normalized = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    cal_residuals_normalized = y_cal_normalized - point_cal_normalized
    n_cal = len(point_cal_normalized)

    min_initial = max(20, round(n_cal * min_initial_frac))
    min_fold_size = max(10, round(n_cal * min_fold_frac))
    fold_bounds = _make_rolling_folds(n_cal, n_folds, min_initial, min_fold_size)
    if fold_bounds is None:
        raise ValueError(
            f"Calibration set too small for rolling-origin CV: n_cal={n_cal}, "
            f"min_initial={min_initial}, min_fold_size={min_fold_size} (from min_initial_frac="
            f"{min_initial_frac}, min_fold_frac={min_fold_frac}) -- try a smaller n_folds / "
            "min_initial_frac / min_fold_frac, or more calibration data"
        )

    mean_fold_calibration_error_by_window: dict = {}
    mean_fold_brier_by_window: dict = {}
    for window in windows:
        fold_errors, fold_briers = [], []
        for train_end, test_end in fold_bounds:
            seed_pool = list(cal_residuals_normalized[:train_end])
            fold_point = point_cal_normalized[train_end:test_end]
            fold_y = y_cal_normalized[train_end:test_end]
            bt = _run_adaptive_pool_backtest(
                fold_point,
                fold_y,
                seed_pool=seed_pool,
                window=window,
                normalizer_mean=result.normalizer_mean,
                normalizer_std=result.normalizer_std,
                relative_offsets=relative_offsets,
                n_bins=n_bins,
            )
            fold_errors.append(bt.mean_calibration_error)
            fold_briers.append(bt.brier_score)
        mean_fold_calibration_error_by_window[window] = float(np.mean(fold_errors))
        mean_fold_brier_by_window[window] = float(np.mean(fold_briers))

    selected_window = _select_best_window(
        windows, mean_fold_calibration_error_by_window, mean_fold_brier_by_window
    )

    # Honest post-hoc check: the real test-set result for every candidate,
    # via the already-tested real-backtest function -- never used to select.
    static_test = backtest_threshold_calibration(result, X_test, y_test, relative_offsets=relative_offsets, n_bins=n_bins)
    test_result_by_window = {
        w: backtest_threshold_calibration_adaptive(
            result, X_test, y_test, relative_offsets=relative_offsets, window=w, n_bins=n_bins
        )
        for w in windows
    }
    test_error_by_window = {w: r.mean_calibration_error for w, r in test_result_by_window.items()}
    test_brier_by_window = {w: r.brier_score for w, r in test_result_by_window.items()}
    best_test_window = _select_best_window(windows, test_error_by_window, test_brier_by_window)

    candidates = [
        AdaptivePoolWindowCandidate(
            window=w,
            mean_fold_calibration_error=mean_fold_calibration_error_by_window[w],
            mean_fold_brier=mean_fold_brier_by_window[w],
            test_calibration_error=test_error_by_window[w],
            test_brier=test_brier_by_window[w],
        )
        for w in windows
    ]

    return AdaptivePoolWindowSelectionResult(
        n_folds=len(fold_bounds),
        fold_bounds=fold_bounds,
        relative_offsets=relative_offsets,
        candidates=candidates,
        selected_window=selected_window,
        best_test_window=best_test_window,
        static_test=static_test,
        selected_test_result=test_result_by_window[selected_window],
    )


def format_adaptive_pool_window_selection(result: AdaptivePoolWindowSelectionResult) -> str:
    fold_desc = ", ".join(f"[{a}:{b})" for a, b in result.fold_bounds)
    lines = [
        f"### Adaptive-pool rolling-origin CV window selection (n_folds={result.n_folds}, folds={fold_desc})",
        "",
        "| Window | Mean fold calibration error (used to select) | Mean fold Brier | Real test calibration error (post-hoc) | Real test Brier (post-hoc) |",
        "|---|---|---|---|---|",
    ]
    for c in result.candidates:
        wlabel = "unbounded (growing buffer)" if c.window is None else str(c.window)
        marker = ""
        if c.window == result.selected_window:
            marker += " **<- selected**"
        if c.window == result.best_test_window:
            marker += " *(best in hindsight)*"
        lines.append(
            f"| {wlabel}{marker} | {c.mean_fold_calibration_error:.4f} | {c.mean_fold_brier:.4f} | "
            f"{c.test_calibration_error:.4f} | {c.test_brier:.4f} |"
        )
    s = result.static_test
    lines += [
        "",
        f"*Reference, real test set, static (frozen) pool:* calibration error {s.mean_calibration_error:.4f}, "
        f"Brier {s.brier_score:.4f}.",
        "",
        f"**Selected window: {('unbounded' if result.selected_window is None else result.selected_window)}** "
        f"(chosen from the {result.n_folds} fold columns only) -> real test-set calibration error "
        f"{result.selected_test_result.mean_calibration_error:.4f}, Brier {result.selected_test_result.brier_score:.4f}.",
    ]
    return "\n".join(lines)
