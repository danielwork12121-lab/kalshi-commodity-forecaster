"""Calibrated commodity-price forecasting, built on conformal-forecast.

This module deliberately does NOT reimplement any splitting, windowing,
normalization, or conformal-calibration logic. It depends on the
`forecasting` package from
https://github.com/danielwork12121-lab/conformal-forecast (installed as a
real pip dependency -- see pyproject.toml) and reuses its own
`prepare_split_from_series` (the public entry point to the exact
split/window/normalize/validate pipeline conformal-forecast validates its
own coverage guarantees against, added in conformal-forecast v0.16
specifically so a downstream project like this one can depend on it rather
than reimplementing a second, possibly-drifting version), its
`SplitConformalForecaster`, and its baseline models.

Honest scope, as of this slice: point forecast + calibrated interval on one
real series (WTI daily spot price), using the simplest baseline model, plus
(since 2026-10-15, see `commodity_forecast/probability.py`) converting that
calibrated output into a Kalshi-style threshold probability, P(price > K),
backtested against real history. Trying a stronger model than the naive
baseline is explicitly NOT done yet -- see the project README's "What's
next."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecasting.adaptive import (
    AdaptiveConformalForecaster,
    SlidingWindowAdaptiveConformalForecaster,
    evaluate_adaptive_coverage,
)
from forecasting.conformal import SplitConformalForecaster, evaluate_coverage
from forecasting.experiment import prepare_split_from_series
from forecasting.models import DeltaWrapper, LSTMForecaster, NaiveForecaster


@dataclass
class CalibratedForecastResult:
    conformal_forecaster: SplitConformalForecaster
    normalizer_mean: float
    normalizer_std: float
    diagnostics: dict
    # Signed (not absolute) calibration-split residuals, in normalized
    # (z-score) units: y_cal - model.predict(X_cal). `SplitConformalForecaster`
    # itself only ever keeps the *absolute* residuals' quantile (`q_hat`) --
    # enough for a symmetric +/- interval, not enough to convert a forecast
    # into a threshold probability, which needs the actual shape (and sign)
    # of the residual distribution. Added 2026-10-15 for
    # `commodity_forecast/probability.py`; unused by anything built before
    # that date.
    cal_residuals_normalized: np.ndarray


@dataclass
class AdaptiveCalibratedForecastResult:
    adaptive_forecaster: AdaptiveConformalForecaster
    normalizer_mean: float
    normalizer_std: float
    diagnostics: dict


def fit_calibrated_naive_forecaster(
    series_values: np.ndarray,
    lookback: int = 5,
    alpha: float = 0.1,
) -> CalibratedForecastResult:
    """Fit a naive-forecaster baseline wrapped in split conformal calibration
    on a real commodity price series, using conformal-forecast's own
    chronological split/window/normalize pipeline end to end.

    Deliberately starts with conformal-forecast's simplest model
    (NaiveForecaster: predicts "tomorrow = today, in normalized units")
    rather than its LSTM, for this first slice -- the point right now is
    proving the real dependency + real data + real calibration pipeline
    holds together end-to-end, not chasing forecast accuracy yet. A
    stronger model is exactly the kind of next step that belongs in a
    later, separately verified run (see README "What's next."), not folded
    in here unverified.

    Returns a CalibratedForecastResult with the fitted forecaster plus
    diagnostics measured on a held-out test split the model and the
    calibration step never saw -- so the coverage claim is measured, not
    just asserted. `period` is passed through to `prepare_split_from_series`
    only for API-shape compatibility with conformal-forecast's own
    bundled-dataset path; this module does not currently use any seasonal
    model that would act on it.
    """
    (
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        normalizer,
        _period,
    ) = prepare_split_from_series(series_values, lookback=lookback, period=lookback, label="wti_spot")

    model = NaiveForecaster()
    conformal = SplitConformalForecaster(model=model, alpha=alpha)
    conformal.fit(X_train, y_train, X_cal, y_cal)

    prediction = conformal.predict(X_test)
    coverage = evaluate_coverage(prediction, y_test)

    # De-normalize for a human-readable MAE in real dollars/barrel, since
    # evaluate_coverage's own numbers are in normalized (z-score) units.
    point_denorm = normalizer.inverse_transform(prediction.point)
    y_test_denorm = normalizer.inverse_transform(y_test)
    mae_usd = float(np.mean(np.abs(point_denorm - y_test_denorm)))

    # Signed calibration residuals, normalized units -- the model is already
    # fit (above), so this is a free extra prediction call on the held-out
    # calibration split, never the test split.
    cal_residuals_normalized = (y_cal - conformal.model.predict(X_cal)).reshape(-1)

    diagnostics = {
        **coverage,
        "mae_usd_per_bbl": mae_usd,
        "n_train_windows": int(len(X_train)),
        "n_cal_windows": int(len(X_cal)),
        "lookback": lookback,
    }

    return CalibratedForecastResult(
        conformal_forecaster=conformal,
        normalizer_mean=normalizer.mean_,
        cal_residuals_normalized=cal_residuals_normalized,
        normalizer_std=normalizer.std_,
        diagnostics=diagnostics,
    )


def fit_calibrated_lstm_forecaster(
    series_values: np.ndarray,
    lookback: int = 5,
    alpha: float = 0.1,
    seed: int = 0,
) -> CalibratedForecastResult:
    """Same calibrated-forecast pipeline as `fit_calibrated_naive_forecaster`
    above, but with conformal-forecast's own from-scratch LSTM instead of
    `NaiveForecaster` -- this project's own "What's next" item 2: *"A
    stronger model than the naive baseline (conformal-forecast's LSTM, and
    now its Transformer too, are already available as a dependency) --
    does ACI's benefit, or the probability layer's calibration, hold up
    with a non-trivial point model?"*

    Wrapped in `DeltaWrapper` (predicting tomorrow's *change* rather than
    its absolute level), the same convention conformal-forecast's own
    `forecasting/experiment.py` uses for every neural model on every
    dataset it benchmarks -- for a near-random-walk daily commodity price
    series, "predict the delta" is a far easier, better-conditioned
    learning target than "predict the level directly," and using anything
    else here would not be a fair test of whether a stronger *model*
    helps, since a bare (non-delta) LSTM would be fighting a harder
    problem than `NaiveForecaster`'s implicit delta-of-zero assumption
    already handles for free.

    Uses the identical train/cal/test split, `lookback`, and `alpha` as
    `fit_calibrated_naive_forecaster` (both call `prepare_split_from_series`
    identically) -- so any difference between the two functions' results
    is attributable to the model alone, not a different split or a
    different amount of held-out data.

    `backtest_threshold_calibration`/`backtest_threshold_calibration_adaptive`
    in `commodity_forecast/probability.py` need no changes to work with
    this function's output: both already only ever read
    `result.conformal_forecaster.{model,predict}`,
    `result.cal_residuals_normalized`, and `result.normalizer_{mean,std}`
    off the generic `CalibratedForecastResult` -- never anything
    `NaiveForecaster`-specific -- so this is a genuinely drop-in stronger
    point model for the whole probability layer, not just the interval
    case.
    """
    (
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        normalizer,
        _period,
    ) = prepare_split_from_series(series_values, lookback=lookback, period=lookback, label="wti_spot")

    model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    conformal = SplitConformalForecaster(model=model, alpha=alpha)
    conformal.fit(X_train, y_train, X_cal, y_cal)

    prediction = conformal.predict(X_test)
    coverage = evaluate_coverage(prediction, y_test)

    point_denorm = normalizer.inverse_transform(prediction.point)
    y_test_denorm = normalizer.inverse_transform(y_test)
    mae_usd = float(np.mean(np.abs(point_denorm - y_test_denorm)))

    cal_residuals_normalized = (y_cal - conformal.model.predict(X_cal)).reshape(-1)

    diagnostics = {
        **coverage,
        "mae_usd_per_bbl": mae_usd,
        "n_train_windows": int(len(X_train)),
        "n_cal_windows": int(len(X_cal)),
        "lookback": lookback,
        "seed": seed,
    }

    return CalibratedForecastResult(
        conformal_forecaster=conformal,
        normalizer_mean=normalizer.mean_,
        cal_residuals_normalized=cal_residuals_normalized,
        normalizer_std=normalizer.std_,
        diagnostics=diagnostics,
    )


def fit_adaptive_naive_forecaster(
    series_values: np.ndarray,
    lookback: int = 5,
    alpha: float = 0.1,
    gamma: float = 0.05,
    variant: str = "sliding",
    window: int | None = None,
) -> AdaptiveCalibratedForecastResult:
    """Fit the same NaiveForecaster baseline as `fit_calibrated_naive_forecaster`,
    but calibrated with conformal-forecast's Adaptive Conformal Inference (ACI)
    instead of static split conformal, to see whether ACI closes the real,
    measured 80.9%-vs-90% coverage gap `fit_calibrated_naive_forecaster` found
    on the real WTI series (see the project README's "First real result").

    `variant="fixed"` uses `AdaptiveConformalForecaster` -- only the
    miscoverage rate (alpha_t) adapts online; the residual pool stays frozen
    at the calibration window, so the interval can never exceed the
    calibration pool's own maximum residual.

    `variant="sliding"` (default) uses `SlidingWindowAdaptiveConformalForecaster`
    -- the residual pool itself is updated online with each step's own
    observed residual (via `window`: `None` for an unbounded growing buffer,
    or a positive int for a true bounded sliding window), removing that
    ceiling. This is the variant conformal-forecast's own README recommends
    trying first when a fixed-pool ceiling is suspected (which is exactly
    the WTI series' situation: the calibration window is a tiny ~5-point
    lookback slice of a 9,500+ point, highly non-stationary series).

    Uses the *same* NaiveForecaster baseline, the same lookback/alpha, and
    the same train/cal/test split as `fit_calibrated_naive_forecaster` (both
    call `prepare_split_from_series` identically), so a comparison against
    that function's diagnostics is a fair, apples-to-apples test of the
    calibration method only -- not a different model or a different split.

    Unlike static split conformal, ACI needs the true y values during
    prediction (it's inherently sequential: alpha_t for step t+1 depends on
    whether step t's interval covered the truth) -- this models a realistic
    deployment where today's real price is known before tomorrow's forecast
    is due, which is true for this series.
    """
    if variant not in ("fixed", "sliding"):
        raise ValueError("variant must be 'fixed' or 'sliding'")

    (
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        normalizer,
        _period,
    ) = prepare_split_from_series(series_values, lookback=lookback, period=lookback, label="wti_spot")

    model = NaiveForecaster()
    if variant == "fixed":
        adaptive: AdaptiveConformalForecaster = AdaptiveConformalForecaster(model, alpha=alpha, gamma=gamma)
    else:
        adaptive = SlidingWindowAdaptiveConformalForecaster(model, alpha=alpha, gamma=gamma, window=window)

    adaptive.fit(X_train, y_train, X_cal, y_cal)
    prediction = adaptive.predict_sequential(X_test, y_test.reshape(-1))
    coverage = evaluate_adaptive_coverage(prediction, y_test)

    # De-normalize for a human-readable MAE in real dollars/barrel, same
    # convention as fit_calibrated_naive_forecaster.
    point_denorm = normalizer.inverse_transform(prediction.point)
    y_test_denorm = normalizer.inverse_transform(y_test)
    mae_usd = float(np.mean(np.abs(point_denorm - y_test_denorm)))

    diagnostics = {
        **coverage,
        "mae_usd_per_bbl": mae_usd,
        "n_train_windows": int(len(X_train)),
        "n_cal_windows": int(len(X_cal)),
        "lookback": lookback,
        "variant": variant,
        "window": window,
    }

    return AdaptiveCalibratedForecastResult(
        adaptive_forecaster=adaptive,
        normalizer_mean=normalizer.mean_,
        normalizer_std=normalizer.std_,
        diagnostics=diagnostics,
    )
