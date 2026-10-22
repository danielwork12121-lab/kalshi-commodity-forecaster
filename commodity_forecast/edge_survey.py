"""A real, cross-sectional edge survey across a live Kalshi market ladder.

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
whatever real open markets exist on a series at the moment
`survey_live_edges` is called -- broken down by real dimensions available
on that one snapshot (basis-adjusted strike distance from the live
reference price, real forecast horizon, and real liquidity). This is
**NOT** a persistence claim across multiple points in time: a subset that
shows a larger edge on one day's ladder could easily be noise, not a real,
repeatable pattern -- see `edge_persistence.py` for the machinery that
actually checks persistence across repeated snapshots. See
`summarize_edge_subsets`'s own docstring and the README's "Edge-subset
survey" section for how this is reported -- honestly, as a single-snapshot
cross-section, never oversold as evidence of a *persistent* edge.

Every real market this survey cannot honestly score (an unsupported
strike shape, a horizon beyond what this project has validated) is
reported in `skipped` with a plain-English reason -- never silently
dropped, same discipline as every other module in this project.

## v0.9.0: the WTI-specific scoring logic moved to `markets/wti.py`

Through v0.8.0, this module did everything inline: fetch the real open
WTI ladder, fetch the real spot series and settlement brackets, compute
the real basis correction, fit a horizon-validated forecaster, score each
market, and call `edge.compute_edge`. `active-project-kalshi.md`'s
2026-10-21 direction-update item 2 calls for growing this into a genuine
multi-market framework (temperature-vs-NWS next) rather than re-measuring
the WTI ladder forever, so that WTI-specific logic is now
`markets.wti.WTIMarketFamily`, the first implementation of the common
`markets.base.MarketFamily` interface -- **ported with no change to its
real, already-tested behavior.** `survey_live_edges` below is now a thin
backward-compatible wrapper: build a `WTIMarketFamily` from the same
parameters it always took, and call the new market-agnostic
`markets.base.run_family_survey`. `SurveyedEdge` itself moved to
`markets/base.py` (so `run_family_survey` can build one without an import
cycle back into this module) and is re-exported here unchanged for every
existing import site.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from commodity_forecast.markets.base import SurveyedEdge, run_family_survey
from commodity_forecast.markets.wti import WTIMarketFamily

__all__ = [
    "SurveyedEdge",
    "EdgeSubsetSummary",
    "survey_live_edges",
    "summarize_edge_subsets",
    "format_edge_subset_summary",
]


def survey_live_edges(
    series_ticker: str = "KXWTI",
    lookback: int = 5,
    alpha: float = 0.1,
    min_volume_for_trust: float = 1.0,
    refresh_spot: bool = False,
    series=None,
    model_name: str = "transformer",
    seed: int = 0,
) -> tuple[list[SurveyedEdge], list[str]]:
    """Real cross-sectional edge survey across every real, currently-open
    market in `series_ticker`. See the module docstring for the honest
    scope (one live snapshot, not a persistence claim).

    Since v0.9.0, this is a thin wrapper: build a
    `markets.wti.WTIMarketFamily` from these exact parameters (unchanged
    meaning from v0.8.0 -- see that class's own docstring), then run the
    market-agnostic `markets.base.run_family_survey` over it. Every real,
    measured behavior (skip reasons, horizon selection, basis correction,
    liquidity caveats, the live default `model_name="transformer"`) is
    unchanged from v0.8.0; only the module the scoring logic itself lives
    in moved (`markets.wti`, not here) -- see that module's docstring for
    the full account.

    `series` defaults to a real fetch (`fetch_wti_daily(refresh=refresh_spot)`)
    -- pass an explicit one (e.g. in tests) to pin what "last observed
    date" this call uses, the same escape hatch `track_record.py`'s own
    `resolve_pending_predictions` already offers for the same reason.

    `model_name`: which of `forecast.MODEL_NAMES` to fit via
    `fit_calibrated_forecaster_horizon` for every horizon this ladder
    actually needs. Defaults to `"transformer"` (v0.8.0) on the strength
    of v0.7.0's real, measured finding that the Transformer point model
    has a consistently better MAE and a smaller coverage gap than naive
    from horizon 3 onward. Pass `model_name="naive"` to reproduce the
    exact pre-0.8.0 behavior (used by this project's own frozen-fixture
    regression tests). `seed` is forwarded to the model constructor for
    `"lstm"`/`"transformer"` (unused by `"naive"`).

    Returns `(surveyed, skipped)`: `surveyed` holds one `SurveyedEdge` per
    real open market this project could honestly score; `skipped` holds a
    plain-English reason string for each real open market it could NOT
    score (never silently dropped). Fits at most one horizon-validated
    forecaster per distinct real horizon the live ladder actually needs
    (never refit per-market).
    """
    family = WTIMarketFamily(
        series_ticker=series_ticker,
        lookback=lookback,
        alpha=alpha,
        refresh_spot=refresh_spot,
        series=series,
        model_name=model_name,
        seed=seed,
    )
    return run_family_survey(family, min_volume_for_trust=min_volume_for_trust)


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
