"""Does a stronger point model help at the *validated* forecast horizon?
(v0.7, 2026-10-21)

Every prior stronger-point-model check in this project (`forecast.py`'s
earlier LSTM comparison, written up in the README's "Does a stronger point
model help?" section) only ever ran at `horizon=1` -- the one this project
has since found is not representative of what a real, live Kalshi `KXWTI`
market actually needs (most open markets close several trading days to a
few weeks out; see the "Horizon" caveat resolution). This module repeats
that same honest comparison, but at each of the seven horizons this
project has actually validated a residual pool for
(`edge.VALIDATED_FORECAST_HORIZONS_DAYS`), using
`forecast.fit_calibrated_forecaster_horizon`'s new `model_name` option
(v0.7) instead of the naive-only function every other module here still
uses.

Honest framing up front: the original horizon=1 check found WTI's
near-random-walk behavior means the LSTM converges to essentially the same
zero-delta forecast `NaiveForecaster` already makes, so this isn't expected
to suddenly diverge just because the horizon is longer -- but whether that
actually holds at every validated horizon, or breaks down somewhere (a
longer horizon gives a model more room to find a real non-trivial pattern,
or conversely more room for a neural model's own extrapolation error to
compound), has never been checked. This module checks it, honestly, rather
than assuming the horizon=1 finding generalizes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from commodity_forecast.forecast import fit_calibrated_forecaster_horizon, prepare_horizon_split
from commodity_forecast.probability import backtest_threshold_calibration

DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 15, 21)
DEFAULT_MODEL_NAMES = ("naive", "lstm", "transformer")


@dataclass
class HorizonModelCell:
    horizon: int
    model_name: str
    n_test: int
    mae_usd_per_bbl: float
    empirical_coverage: float
    coverage_gap: float
    brier_score: float
    mean_calibration_error: float


@dataclass
class HorizonModelComparison:
    horizons: tuple[int, ...]
    model_names: tuple[str, ...]
    lookback: int
    alpha: float
    seed: int
    cells: list[HorizonModelCell]  # horizon-major order


def run_horizon_model_comparison(
    series_values: np.ndarray,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    model_names: tuple[str, ...] = DEFAULT_MODEL_NAMES,
    lookback: int = 5,
    alpha: float = 0.1,
    seed: int = 0,
) -> HorizonModelComparison:
    """Real comparison: for every (horizon, model) pair, fit the
    horizon-validated calibrated forecaster, measure real MAE/coverage on
    the real held-out test split, and backtest the probability conversion
    (Brier score, mean calibration error) the exact same honest way every
    other probability result in this project is measured -- never read off
    training data, always the held-out test split.

    Each `(horizon, model_name)` pair gets its own full, independent fit
    (the model's own training, not just how an already-fitted model's
    output is read, changes with both horizon -- a different target -- and
    model choice).
    """
    cells = []
    for horizon in horizons:
        _, _, _, _, X_test, y_test, _ = prepare_horizon_split(series_values, horizon=horizon, lookback=lookback)
        for model_name in model_names:
            result = fit_calibrated_forecaster_horizon(
                series_values, horizon=horizon, model_name=model_name, lookback=lookback, alpha=alpha, seed=seed
            )
            backtest = backtest_threshold_calibration(result, X_test, y_test)
            cells.append(
                HorizonModelCell(
                    horizon=horizon,
                    model_name=model_name,
                    n_test=result.diagnostics["n_test"],
                    mae_usd_per_bbl=result.diagnostics["mae_usd_per_bbl"],
                    empirical_coverage=result.diagnostics["empirical_coverage"],
                    coverage_gap=result.diagnostics["coverage_gap"],
                    brier_score=backtest.brier_score,
                    mean_calibration_error=backtest.mean_calibration_error,
                )
            )

    return HorizonModelComparison(
        horizons=tuple(horizons),
        model_names=tuple(model_names),
        lookback=lookback,
        alpha=alpha,
        seed=seed,
        cells=cells,
    )


def format_horizon_model_comparison(comparison: HorizonModelComparison) -> str:
    nominal_pct = int(round((1 - comparison.alpha) * 100))
    by_key = {(c.horizon, c.model_name): c for c in comparison.cells}
    lines = [
        f"### Does a stronger point model help, at every validated horizon? "
        f"(nominal {nominal_pct}%, lookback={comparison.lookback}, seed={comparison.seed})",
        "",
        "| Horizon | Model | MAE ($/bbl) | Coverage | Brier | Mean calib. error |",
        "|---|---|---|---|---|---|",
    ]
    for horizon in comparison.horizons:
        best_mae_model = min(comparison.model_names, key=lambda m: by_key[(horizon, m)].mae_usd_per_bbl)
        for model_name in comparison.model_names:
            c = by_key[(horizon, model_name)]
            marker = " **<- best MAE this horizon**" if model_name == best_mae_model else ""
            lines.append(
                f"| {horizon} | {model_name}{marker} | ${c.mae_usd_per_bbl:.2f} | "
                f"{c.empirical_coverage * 100:.1f}% ({c.coverage_gap * 100:+.1f}pp) | "
                f"{c.brier_score:.4f} | {c.mean_calibration_error:.4f} |"
            )
    return "\n".join(lines)
