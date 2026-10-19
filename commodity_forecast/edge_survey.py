"""A real, cross-sectional edge survey across the live `KXWTI` ladder.

## Why this exists

`active-project-kalshi.md`'s "What's next" item 3 (2026-10-19, after both
the settlement-benchmark and forecast-horizon caveats were resolved) asks a
real follow-up question honestly, rather than stopping at the ladder-wide
average: the corrected edge does not show a trustworthy *average*
profitable signal (see `edge.py`'s module docstring), but does **any real
subset** of the live ladder show a notably different edge than that
average -- e.g. markets near the current spot price vs. far out-of-the-money
ones, short-horizon vs. long-horizon questions, or liquid vs. illiquid
strikes?

## Honest scope, stated plainly up front

This module answers that question from **one real, live snapshot** --
whatever real open markets exist on a series (default `KXWTI`) at the
moment `survey_live_edges` is called -- broken down by real dimensions
available on that one snapshot (basis-adjusted strike distance from the
live spot price, real forecast horizon, and real liquidity). This is
**NOT** a persistence claim across multiple points in time: a subset that
shows a larger edge on one day's ladder could easily be noise, not a real,
repeatable pattern, and this project has no repeated-snapshot history yet
to tell the difference. See `summarize_edge_subsets`'s own docstring and
the README's "Edge-subset survey" section for how this is reported --
honestly, as a single-snapshot cross-section, never oversold as evidence
of a *persistent* edge. The natural next step (not yet attempted) is
running this same survey on successive scheduled runs and checking whether
any one subset's edge sign/magnitude actually holds up over time, which is
the only way a "persistent" claim could honestly be supported.

Every real market this survey cannot honestly score (an unsupported
strike shape, a horizon beyond what this project has validated) is
reported in `skipped` with a plain-English reason -- never silently
dropped, same discipline as every other module in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from commodity_forecast.benchmark import (
    basis_adjusted_strike,
    describe_basis_correction,
    fetch_settlement_brackets,
    measure_basis,
    recent_basis_estimate,
)
from commodity_forecast.data import fetch_wti_daily
from commodity_forecast.edge import (
    EdgeEstimate,
    compute_edge,
    days_to_close_from_last_observation,
    pick_validated_horizon,
)
from commodity_forecast.forecast import fit_calibrated_naive_forecaster_horizon
from commodity_forecast.kalshi_client import fetch_markets
from commodity_forecast.probability import exceedance_probability


@dataclass
class SurveyedEdge:
    """One real `EdgeEstimate` plus the real cross-sectional metadata this
    survey groups by. Kept separate from `EdgeEstimate` itself (a single-
    market concept `edge.py` already owns and that other callers construct
    directly) so this module can add survey-only context without changing
    that dataclass's shape or its existing tests/callers.
    """

    estimate: EdgeEstimate
    days_to_close: float
    horizon_used: int
    strike_distance_usd: float  # basis-adjusted strike minus last observed spot price; >0 = strike above spot


def survey_live_edges(
    series_ticker: str = "KXWTI",
    lookback: int = 5,
    alpha: float = 0.1,
    min_volume_for_trust: float = 1.0,
    refresh_spot: bool = False,
) -> tuple[list[SurveyedEdge], list[str]]:
    """Real cross-sectional edge survey across every real, currently-open
    market in `series_ticker`. See the module docstring for the honest
    scope (one live snapshot, not a persistence claim).

    Returns `(surveyed, skipped)`: `surveyed` holds one `SurveyedEdge` per
    real open market this project could honestly score; `skipped` holds a
    plain-English reason string for each real open market it could NOT
    score (never silently dropped). Fits at most one horizon-validated
    forecaster per distinct real horizon the live ladder actually needs
    (never refit per-market), via the same already-tested
    `fit_calibrated_naive_forecaster_horizon` the Horizon-caveat resolution
    already validated.
    """
    quotes = fetch_markets(series_ticker, status="open")
    series = fetch_wti_daily(refresh=refresh_spot)
    last_observed_date = series.index.max()
    last_observed_price = float(series.iloc[-1])

    brackets = fetch_settlement_brackets(series_ticker)
    spot_by_date = {d.date().isoformat(): float(v) for d, v in series.items()}
    measurements = measure_basis(brackets, spot_by_date)
    basis_est = recent_basis_estimate(measurements)
    if basis_est is not None:
        basis_mean, basis_stdev, basis_n = basis_est
        basis_trustworthy, basis_note = describe_basis_correction(basis_mean, basis_stdev, basis_n)
    else:
        basis_mean, basis_trustworthy = 0.0, False
        basis_note = "no real settlement brackets available to measure a basis this run"

    fitted_by_horizon: dict[int, object] = {}
    surveyed: list[SurveyedEdge] = []
    skipped: list[str] = []

    for quote in quotes:
        if quote.strike_type != "greater" or quote.floor_strike is None:
            skipped.append(
                f"{quote.ticker}: strike_type={quote.strike_type!r} is not a simple 'above $X' "
                "question this survey knows how to score, skipped"
            )
            continue

        days_to_close = days_to_close_from_last_observation(quote.close_time, last_observed_date)
        if days_to_close < 0:
            skipped.append(
                f"{quote.ticker}: close_time is before this project's own last observed spot date "
                f"({last_observed_date.date().isoformat()}), skipped"
            )
            continue

        horizon = pick_validated_horizon(days_to_close)
        if horizon is None:
            skipped.append(
                f"{quote.ticker}: real days_to_close={days_to_close:.1f} exceeds every horizon this "
                "project has validated -- skipped rather than extrapolated past where it's trustworthy"
            )
            continue

        if horizon not in fitted_by_horizon:
            fitted_by_horizon[horizon] = fit_calibrated_naive_forecaster_horizon(
                series.values, horizon=horizon, lookback=lookback, alpha=alpha
            )
        result = fitted_by_horizon[horizon]

        last_window_normalized = (
            (np.asarray(series.values[-lookback:], dtype=np.float64) - result.normalizer_mean)
            / result.normalizer_std
        ).reshape(1, -1)
        point_forecast_normalized = result.conformal_forecaster.model.predict(last_window_normalized).reshape(-1)

        adjusted_strike = (
            basis_adjusted_strike(quote.floor_strike, basis_mean) if basis_trustworthy else quote.floor_strike
        )
        threshold_normalized = (adjusted_strike - result.normalizer_mean) / result.normalizer_std

        our_probability = float(
            exceedance_probability(point_forecast_normalized, result.cal_residuals_normalized, threshold_normalized)[0]
        )

        estimate = compute_edge(
            our_probability=our_probability,
            quote=quote,
            strike_usd=adjusted_strike,
            settlement_benchmark_matches_forecast=basis_trustworthy,
            min_volume_for_trust=min_volume_for_trust,
            forecast_horizon_days=horizon,
            days_to_close=days_to_close,
            basis_note=basis_note,
        )
        surveyed.append(
            SurveyedEdge(
                estimate=estimate,
                days_to_close=days_to_close,
                horizon_used=horizon,
                strike_distance_usd=adjusted_strike - last_observed_price,
            )
        )

    return surveyed, skipped


@dataclass
class EdgeSubsetSummary:
    """A real, honest summary statistic for one real subset of a survey --
    never a p-value or a significance claim, since a handful of markets
    from one snapshot is nowhere near enough real data to support one.
    """

    label: str
    n: int
    mean_edge: float
    mean_abs_edge: float
    edges: list[float] = field(default_factory=list)


def _summary(label: str, edges: list[float]) -> EdgeSubsetSummary:
    n = len(edges)
    mean_edge = float(sum(edges) / n) if n else 0.0
    mean_abs_edge = float(sum(abs(e) for e in edges) / n) if n else 0.0
    return EdgeSubsetSummary(label=label, n=n, mean_edge=mean_edge, mean_abs_edge=mean_abs_edge, edges=list(edges))


def summarize_edge_subsets(
    surveyed: list[SurveyedEdge],
    trustworthy_only: bool = True,
    near_money_usd: float = 2.0,
    short_horizon_days: int = 5,
) -> list[EdgeSubsetSummary]:
    """Break a real survey down into real, named subsets and report each
    one's real mean edge -- the actual answer to "does any subset show a
    notably different edge than the ladder-wide average," computed, never
    asserted.

    `trustworthy_only=True` (the default) restricts every subset to
    estimates with `EdgeEstimate.trustworthy is True` -- i.e. real,
    basis-corrected, liquid, horizon-valid estimates only. This is the
    honest default: averaging in estimates this project's own code already
    flags as untrustworthy would silently launder a caveated number into an
    uncaveated-looking summary statistic. Pass `False` only to inspect the
    raw, uncorrected distribution for diagnostic purposes.

    Subsets, each a real, pre-registered dimension (not picked after
    looking at the data to manufacture a win):
    - "all" -- every scored estimate (the ladder-wide average).
    - "near-the-money" / "far-out-of-the-money" -- split on
      `abs(strike_distance_usd) <= near_money_usd`.
    - "short-horizon" / "long-horizon" -- split on `horizon_used <= short_horizon_days`.
    - "liquid" / "illiquid" -- split on `estimate.market_volume > 0`.
    """
    pool = [s for s in surveyed if (s.estimate.trustworthy or not trustworthy_only)]

    def edges_where(pred) -> list[float]:
        return [s.estimate.edge for s in pool if pred(s)]

    return [
        _summary("all", edges_where(lambda s: True)),
        _summary("near-the-money", edges_where(lambda s: abs(s.strike_distance_usd) <= near_money_usd)),
        _summary("far-out-of-the-money", edges_where(lambda s: abs(s.strike_distance_usd) > near_money_usd)),
        _summary("short-horizon", edges_where(lambda s: s.horizon_used <= short_horizon_days)),
        _summary("long-horizon", edges_where(lambda s: s.horizon_used > short_horizon_days)),
        _summary("liquid", edges_where(lambda s: s.estimate.market_volume > 0)),
        _summary("illiquid", edges_where(lambda s: s.estimate.market_volume <= 0)),
    ]


def format_edge_subset_summary(summaries: list[EdgeSubsetSummary]) -> str:
    """A plain-text table for the README -- real numbers only, `n=0`
    subsets reported as such (not hidden), never a significance claim."""
    lines = [f"{'subset':<22} {'n':>4} {'mean edge':>11} {'mean |edge|':>13}"]
    for s in summaries:
        if s.n == 0:
            lines.append(f"{s.label:<22} {0:>4} {'n/a':>11} {'n/a':>13}")
        else:
            lines.append(f"{s.label:<22} {s.n:>4} {s.mean_edge:>+11.4f} {s.mean_abs_edge:>13.4f}")
    return "\n".join(lines)
