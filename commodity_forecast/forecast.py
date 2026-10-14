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

Honest scope, as of this first slice: point forecast + calibrated interval
on one real series (WTI daily spot price), using the simplest baseline
model. Converting an interval into a Kalshi-style threshold probability
(e.g. P(price > K)), backtesting that conversion against real history, and
trying a stronger model than the naive baseline are explicitly NOT done yet
-- see the project README's "What's next."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecasting.conformal import SplitConformalForecaster, evaluate_coverage
from forecasting.experiment import prepare_split_from_series
from forecasting.models import NaiveForecaster


@dataclass
class CalibratedForecastResult:
    conformal_forecaster: SplitConformalForecaster
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
        normalizer_std=normalizer.std_,
        diagnostics=diagnostics,
    )
