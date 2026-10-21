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
from forecasting.data import Normalizer, chronological_split, make_windows
from forecasting.experiment import prepare_split_from_series
from forecasting.models import DeltaWrapper, LSTMForecaster, NaiveForecaster, TransformerForecaster


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


def prepare_horizon_split(
    series_values: np.ndarray,
    horizon: int,
    lookback: int = 5,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
):
    """Chronological train/cal/test split + windowing for a genuine
    `horizon`-steps-ahead forecast target.

    Added 2026-10-17 to answer this project's own standing "What's next"
    item: every number in this project's README up to this point (coverage,
    calibration error, the edge-vs-market comparison) was validated exactly
    one trading day ahead, while `edge.py`'s own real investigation found
    live Kalshi `KXWTI` markets closing two or more weeks out -- naively
    reusing the one-day residual pool that far out produced the
    overconfident, almost-certainly-spurious 60-95-point "edges" documented
    in that module's docstring. This function is the honest fix: a forecast
    target genuinely `horizon` steps ahead, with its own, separately
    measured calibration residual pool -- not the one-day pool reused past
    where it was ever validated.

    Built directly on conformal-forecast's own already-tested, UNMODIFIED
    primitives (`chronological_split`, `Normalizer`, `make_windows` --
    `make_windows` has supported a `horizon` parameter since conformal-forecast
    v0.1; it was simply never threaded through conformal-forecast's own
    public `prepare_split_from_series` wrapper, which hardcodes horizon=1
    internally). Deliberately does NOT modify conformal-forecast itself
    (changing a function two projects share is a bigger, riskier unilateral
    edit than this project composing conformal-forecast's own lower-level,
    already-public primitives a different way) -- this mirrors
    `_prepare_split_from_series`'s own index arithmetic, generalized for
    `horizon > 1`, not a reimplementation of any conformal-calibration math.

    Window i's target is `series[i + lookback + horizon - 1]` -- the single
    price exactly `horizon` steps after window i's own lookback context
    ends, not the whole intervening path `make_windows` would otherwise
    return (a "will price be above K in `horizon` days" question only ever
    needs that one future value, not every day in between).

    At `horizon=1` this reduces exactly to `_prepare_split_from_series`'s
    own split (same `chronological_split`/`Normalizer`/`make_windows` calls,
    same index arithmetic with `horizon=1` substituted in) -- confirmed by a
    dedicated test comparing this function's horizon=1 output against
    `fit_calibrated_naive_forecaster`'s own internal split byte-for-byte.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    split = chronological_split(series_values, train_frac=train_frac, cal_frac=cal_frac)
    normalizer = Normalizer().fit(split.train)
    norm_series = normalizer.transform(split.full())

    X, y_path = make_windows(norm_series, lookback=lookback, horizon=horizon)
    y = y_path[:, -1:]  # only the exact horizon-ahead value, not the full intervening path

    n_train_raw, n_cal_raw = len(split.train), len(split.calibration)
    n_train_windows = max(0, n_train_raw - lookback - horizon + 1)
    n_cal_windows = n_cal_raw

    X_train, y_train = X[:n_train_windows], y[:n_train_windows]
    X_cal, y_cal = X[n_train_windows : n_train_windows + n_cal_windows], y[
        n_train_windows : n_train_windows + n_cal_windows
    ]
    X_test, y_test = X[n_train_windows + n_cal_windows :], y[n_train_windows + n_cal_windows :]

    if len(X_train) < 5 or len(X_cal) < 5 or len(X_test) < 5:
        raise ValueError(
            f"Not enough windows for horizon={horizon} lookback={lookback}: "
            f"train={len(X_train)} cal={len(X_cal)} test={len(X_test)}"
        )

    return X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer


def fit_calibrated_naive_forecaster_horizon(
    series_values: np.ndarray,
    horizon: int,
    lookback: int = 5,
    alpha: float = 0.1,
) -> CalibratedForecastResult:
    """Same calibrated-forecast pipeline as `fit_calibrated_naive_forecaster`,
    but genuinely validated `horizon` steps ahead instead of always one --
    see `prepare_horizon_split`'s own docstring for why this exists and the
    real caveat it answers (`edge.py`'s forecast-horizon-mismatch finding).

    Returns a plain `CalibratedForecastResult`, same shape every other
    function in this module returns, specifically so
    `commodity_forecast.probability.backtest_threshold_calibration` (and
    every other downstream consumer of a `CalibratedForecastResult`) works
    on a horizon-validated result completely unchanged -- no new backtest
    machinery needed, just a result whose residual pool was honestly
    measured at the horizon it's about to be used for, instead of borrowed
    from the one-day pool.
    """
    (
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        normalizer,
    ) = prepare_horizon_split(series_values, horizon=horizon, lookback=lookback)

    model = NaiveForecaster()
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
        "horizon": horizon,
    }

    return CalibratedForecastResult(
        conformal_forecaster=conformal,
        normalizer_mean=normalizer.mean_,
        cal_residuals_normalized=cal_residuals_normalized,
        normalizer_std=normalizer.std_,
        diagnostics=diagnostics,
    )


MODEL_NAMES = ("naive", "lstm", "transformer")


def _build_model(model_name: str, seed: int):
    """Construct a fresh, untrained model for `fit_calibrated_forecaster_horizon`.

    'naive' returns conformal-forecast's own `NaiveForecaster` directly
    (no `DeltaWrapper` -- it's already a "tomorrow = today" delta model in
    effect, and wrapping it again would double-apply the same identity).
    'lstm'/'transformer' return conformal-forecast's own from-scratch
    PyTorch models, `DeltaWrapper`-wrapped with the same discipline that
    library's own benchmark uses throughout (predict the *change* from the
    last observed value, not the absolute level -- see `DeltaWrapper`'s own
    docstring for why this matters for a trending series).
    """
    if model_name == "naive":
        return NaiveForecaster()
    if model_name == "lstm":
        return DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    if model_name == "transformer":
        return DeltaWrapper(TransformerForecaster(seed=seed))
    raise ValueError(f"Unknown model_name {model_name!r}; must be one of {MODEL_NAMES}")


def fit_calibrated_forecaster_horizon(
    series_values: np.ndarray,
    horizon: int,
    model_name: str = "naive",
    lookback: int = 5,
    alpha: float = 0.1,
    seed: int = 0,
) -> CalibratedForecastResult:
    """Generalizes `fit_calibrated_naive_forecaster_horizon` to any of
    `MODEL_NAMES`, instead of always `NaiveForecaster`.

    Added to answer this project's own README "What's next" item 7's
    still-open candidate: "worth re-trying conformal-forecast's own
    LSTM/Transformer/N-BEATS model families at this horizon specifically,
    not just at horizon=1 as already checked" -- every prior
    stronger-point-model comparison in this project (`forecast.py`'s
    earlier one-horizon LSTM check, see the README's "Does a stronger
    point model help?" section) only ever ran at `horizon=1`, the one
    this project has since found is not what real, live Kalshi markets
    actually need (see the Horizon-caveat resolution above).

    Deliberately does NOT replace `fit_calibrated_naive_forecaster_horizon`
    (every existing caller -- `edge.py`, `edge_survey.py`, and this
    project's own tests -- keeps using the unchanged naive-only function,
    same "don't touch what's already shipped and depended on" discipline
    `prepare_horizon_split`'s own docstring already follows for
    conformal-forecast itself). `model_name="naive"` here is confirmed,
    not just assumed, to produce byte-identical results to that function
    (same split, same model construction, same everything) -- see
    `tests/test_horizon_model_comparison.py`.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name {model_name!r}; must be one of {MODEL_NAMES}")

    (
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        normalizer,
    ) = prepare_horizon_split(series_values, horizon=horizon, lookback=lookback)

    model = _build_model(model_name, seed=seed)
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
        "horizon": horizon,
        "model_name": model_name,
    }

    return CalibratedForecastResult(
        conformal_forecaster=conformal,
        normalizer_mean=normalizer.mean_,
        cal_residuals_normalized=cal_residuals_normalized,
        normalizer_std=normalizer.std_,
        diagnostics=diagnostics,
    )
