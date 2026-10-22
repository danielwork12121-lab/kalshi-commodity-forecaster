"""The real WTI market family: `KXWTI`, scored via this project's own
EIA-spot-trained, horizon-validated forecaster (`forecast.py`), basis-
corrected against the real ICE-futures settlement benchmark `KXWTI`
actually settles against (`benchmark.py`) -- see `edge.py`'s own module
docstring for the full honesty discussion (settlement-benchmark mismatch,
illiquidity, forecast-horizon mismatch).

## Why this is its own module now (v0.9.0)

This is the exact scoring logic that lived inline in
`edge_survey.survey_live_edges` through v0.8.0 -- ported here, with **no
change to its real, already-tested behavior**, as this project's first
concrete `MarketFamily` implementation (see `markets/base.py` for why:
`active-project-kalshi.md`'s 2026-10-21 direction-update item 2 calls for
a multi-market framework, starting with temperature-vs-NWS next, rather
than re-measuring the WTI ladder forever). `edge_survey.survey_live_edges`
is now a thin backward-compatible wrapper:
`run_family_survey(WTIMarketFamily(...))`. Every real, measured behavior
(skip reasons, horizon selection, basis correction, liquidity caveats,
the `model_name`/`seed` parameters added in v0.7.2/v0.8.0) is unchanged --
only the module it lives in moved.
"""

from __future__ import annotations

import numpy as np

from commodity_forecast.benchmark import (
    basis_adjusted_strike,
    describe_basis_correction,
    fetch_settlement_brackets,
    measure_basis,
    recent_basis_estimate,
)
from commodity_forecast.data import fetch_wti_daily
from commodity_forecast.edge import days_to_close_from_last_observation, pick_validated_horizon
from commodity_forecast.forecast import MODEL_NAMES, fit_calibrated_forecaster_horizon
from commodity_forecast.kalshi_client import KalshiMarketQuote, fetch_markets
from commodity_forecast.markets.base import MarketFamily, MarketNotScorable, ProbabilityEstimate
from commodity_forecast.probability import exceedance_probability


class WTIMarketFamily(MarketFamily):
    """Scores the real, live `KXWTI` ladder. All construction-time work
    (fetching/pinning the real spot series, fetching real settlement
    brackets, computing the real recent-window basis estimate) happens
    once in `__init__` -- exactly when `survey_live_edges` used to do it,
    before the per-market scoring loop -- not lazily per-market, so a
    single `WTIMarketFamily` instance fits at most one model per distinct
    real horizon the ladder needs, same as before.
    """

    def __init__(
        self,
        series_ticker: str = "KXWTI",
        lookback: int = 5,
        alpha: float = 0.1,
        refresh_spot: bool = False,
        series=None,
        model_name: str = "transformer",
        seed: int = 0,
    ) -> None:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model_name {model_name!r}; must be one of {MODEL_NAMES}")

        self.series_ticker = series_ticker
        self.lookback = lookback
        self.alpha = alpha
        self.model_name = model_name
        self.seed = seed

        self._series = series if series is not None else fetch_wti_daily(refresh=refresh_spot)
        self._last_observed_date = self._series.index.max()
        self._last_observed_price = float(self._series.iloc[-1])

        brackets = fetch_settlement_brackets(series_ticker)
        spot_by_date = {d.date().isoformat(): float(v) for d, v in self._series.items()}
        measurements = measure_basis(brackets, spot_by_date)
        basis_est = recent_basis_estimate(measurements)
        if basis_est is not None:
            self._basis_mean, basis_stdev, basis_n = basis_est
            self._basis_trustworthy, self._basis_note = describe_basis_correction(
                self._basis_mean, basis_stdev, basis_n
            )
        else:
            self._basis_mean, self._basis_trustworthy = 0.0, False
            self._basis_note = "no real settlement brackets available to measure a basis this run"

        self._fitted_by_horizon: dict[int, object] = {}

    def list_open_markets(self) -> list[KalshiMarketQuote]:
        return fetch_markets(self.series_ticker, status="open")

    def estimate_probability(self, market: KalshiMarketQuote) -> ProbabilityEstimate:
        if market.strike_type != "greater" or market.floor_strike is None:
            raise MarketNotScorable(
                f"strike_type={market.strike_type!r} is not a simple 'above $X' question this "
                "survey knows how to score, skipped"
            )

        days_to_close = days_to_close_from_last_observation(market.close_time, self._last_observed_date)
        if days_to_close < 0:
            raise MarketNotScorable(
                "close_time is before this project's own last observed spot date "
                f"({self._last_observed_date.date().isoformat()}), skipped"
            )

        horizon = pick_validated_horizon(days_to_close)
        if horizon is None:
            raise MarketNotScorable(
                f"real days_to_close={days_to_close:.1f} exceeds every horizon this project has "
                "validated -- skipped rather than extrapolated past where it's trustworthy"
            )

        if horizon not in self._fitted_by_horizon:
            self._fitted_by_horizon[horizon] = fit_calibrated_forecaster_horizon(
                self._series.values,
                horizon=horizon,
                model_name=self.model_name,
                lookback=self.lookback,
                alpha=self.alpha,
                seed=self.seed,
            )
        result = self._fitted_by_horizon[horizon]

        last_window_normalized = (
            (np.asarray(self._series.values[-self.lookback :], dtype=np.float64) - result.normalizer_mean)
            / result.normalizer_std
        ).reshape(1, -1)
        point_forecast_normalized = result.conformal_forecaster.model.predict(last_window_normalized).reshape(-1)

        adjusted_strike = (
            basis_adjusted_strike(market.floor_strike, self._basis_mean)
            if self._basis_trustworthy
            else market.floor_strike
        )
        threshold_normalized = (adjusted_strike - result.normalizer_mean) / result.normalizer_std

        our_probability = float(
            exceedance_probability(
                point_forecast_normalized, result.cal_residuals_normalized, threshold_normalized
            )[0]
        )

        return ProbabilityEstimate(
            our_probability=our_probability,
            adjusted_strike=adjusted_strike,
            settlement_benchmark_matches_forecast=self._basis_trustworthy,
            forecast_horizon_days=horizon,
            days_to_close=days_to_close,
            reference_price=self._last_observed_price,
            basis_note=self._basis_note,
        )
