"""Common interface for one Kalshi market *family* -- a group of real
markets answering the same kind of real-world threshold question (e.g.
"is WTI spot above $X on date Y", or, for a future family not yet built,
"is the NWS-forecast high temperature above X degrees on date Y") -- plus
the one market-agnostic driving loop (`run_family_survey`) every family
reuses rather than reimplementing.

## Why this exists

`active-project-kalshi.md`'s 2026-10-21 direction-update item 2: six real
naive-model snapshots and a first real Transformer-model snapshot all
agree the live `KXWTI` ladder shows no trustworthy edge once the
settlement-benchmark and liquidity caveats are honestly applied.
Re-measuring the same single market forever is not the same as making
progress -- the project should grow into a genuine multi-market edge
framework, starting with temperature-vs-NWS markets (not yet built; see
`wti.py`'s own module docstring and the README's "What's next" for the
plan). This module is the shared interface every family implements, so
adding a new one is a matter of writing one new `MarketFamily` subclass,
not touching the survey/reporting machinery (`run_family_survey` here,
`summarize_edge_subsets`/`format_edge_subset_summary` in `edge_survey.py`)
again.

## What moved, and what didn't

Before v0.9.0, `edge_survey.survey_live_edges` did everything inline: fetch
the real open ladder, fetch the real spot series and settlement brackets,
compute the real basis correction, fit a horizon-validated forecaster per
distinct horizon the ladder needed, score each market, and call
`edge.compute_edge`. All of that WTI-specific logic (series, basis,
forecaster fitting) is now `wti.WTIMarketFamily`, with **no change to its
real, already-tested scoring behavior** -- the three frozen-fixture
end-to-end tests in `tests/test_edge_survey.py` still pass unchanged in
substance (only their monkeypatch targets moved, since the real code they
patch now lives in `markets.wti`, not `edge_survey`). The one, genuinely
new, market-agnostic piece is `run_family_survey` below: given any
`MarketFamily`, it fetches that family's own open markets, asks the
family to score each one, calls the *same* `edge.compute_edge` every
family will ever call (that module was already market-agnostic -- see its
own docstring), and builds the same `SurveyedEdge` list
`edge_survey.summarize_edge_subsets` already knows how to break down.
`edge_survey.survey_live_edges` is now a thin backward-compatible wrapper:
`WTIMarketFamily(...)` + `run_family_survey(...)`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from commodity_forecast.edge import EdgeEstimate, compute_edge
from commodity_forecast.kalshi_client import KalshiMarketQuote


@dataclass(frozen=True)
class ProbabilityEstimate:
    """One family's own real calibrated probability for one real open
    market, plus whatever family-specific context `edge.compute_edge`
    needs in order to attach the right honesty caveats -- kept generic
    across families so `run_family_survey` never needs to know a family's
    own internals (which forecaster it fit, what its basis correction
    looked like, etc.).

    `adjusted_strike`: the real threshold this probability was actually
    computed against -- may differ from the market's own raw
    `floor_strike` (e.g. WTI's basis-corrected strike; a family with no
    such correction can just pass the market's own strike through
    unchanged).

    `reference_price`: this family's own current real "reference" value
    for computing `SurveyedEdge.strike_distance_usd` (how far a market's
    threshold sits from where this family's own forecast currently is) --
    WTI's is the last observed real spot price; a future family could use
    its own forecast's point estimate, a current observed reading, or
    whatever reference makes "near/far from the money" a meaningful split
    for that family's own real-world question. `None` when no such
    reference is meaningful for a given estimate (the resulting
    `strike_distance_usd` is then `float("nan")`, never silently zero).
    """

    our_probability: float
    adjusted_strike: float | None
    settlement_benchmark_matches_forecast: bool
    forecast_horizon_days: int | None
    days_to_close: float | None
    reference_price: float | None = None
    basis_note: str | None = None


class MarketNotScorable(Exception):
    """Raised by `MarketFamily.estimate_probability` for one real market
    that family's own method cannot honestly score (an unsupported strike
    shape, a horizon past what's validated, etc.) -- carries the plain-
    English reason as its message. `run_family_survey` catches this
    specific exception (never a bare `Exception`) so a real, unrelated bug
    in a family's own code still surfaces as a real error instead of being
    silently swallowed into `skipped`.
    """


class MarketFamily(ABC):
    """One real market family: a real Kalshi series plus everything needed
    to turn its real, live open ladder into real, caveated `EdgeEstimate`s.
    Implement `list_open_markets` and `estimate_probability` once per
    family (`wti.WTIMarketFamily` is the first); `run_family_survey` below
    is written once and reused by every family that exists now or gets
    added later.
    """

    series_ticker: str

    @abstractmethod
    def list_open_markets(self) -> list[KalshiMarketQuote]:
        """Real, currently-open markets in this family's own series.
        Implementations may return every open market and let
        `estimate_probability` raise `MarketNotScorable` per-market for any
        shape they can't handle (this is what `WTIMarketFamily` does,
        since Kalshi's own `status="open"` filter is already the natural
        real-world scope), or filter up front here instead -- whichever
        fits a given family's own real API access pattern best.
        """

    @abstractmethod
    def estimate_probability(self, market: KalshiMarketQuote) -> ProbabilityEstimate:
        """This family's own real calibrated probability estimate for one
        real open market, plus the context `edge.compute_edge` needs to
        attach the right honesty caveats. Raise `MarketNotScorable` for a
        market this family's own method cannot honestly score -- never
        return a low-confidence guess instead, and never silently drop the
        market without a reason (that's what `MarketNotScorable`'s message
        is for; `run_family_survey` records it in `skipped`).
        """

    def settle(self, market: KalshiMarketQuote) -> KalshiMarketQuote:
        """The real, current state of `market`, re-fetched fresh from
        Kalshi's own public API. A ticker's settlement lookup doesn't
        depend on which family scored it, so this default implementation
        (a plain `kalshi_client.fetch_market` call) is generic across every
        family and should rarely need overriding -- it exists mainly so a
        future cross-family paper-bet/prediction-resolution step (mirroring
        `edge.resolve_pending_paper_bets`) can work the same way regardless
        of which family logged the original bet, without hardcoding WTI's
        own series anywhere in that resolution code.
        """
        from commodity_forecast.kalshi_client import fetch_market

        return fetch_market(market.ticker)


@dataclass
class SurveyedEdge:
    """One real `EdgeEstimate` plus the real cross-sectional metadata
    `edge_survey.summarize_edge_subsets` groups by -- unchanged from its
    pre-v0.9.0 shape (previously defined in `edge_survey.py`; moved here
    only so `run_family_survey` can build one without an import cycle back
    into `edge_survey`). `edge_survey.py` re-exports this name for every
    existing import site.
    """

    estimate: EdgeEstimate
    days_to_close: float
    horizon_used: int
    strike_distance_usd: float


def run_family_survey(
    family: MarketFamily,
    min_volume_for_trust: float = 1.0,
    retry_if_empty: bool = True,
    empty_retry_delay: float = 2.0,
) -> tuple[list[SurveyedEdge], list[str]]:
    """The market-agnostic survey loop every `MarketFamily` shares: fetch
    `family`'s own real open markets, ask the family to score each one,
    compare against the real market price via the already-market-agnostic
    `edge.compute_edge`, and collect the result into the same
    `(surveyed, skipped)` shape `edge_survey.survey_live_edges` has always
    returned. A market the family raises `MarketNotScorable` for is
    recorded in `skipped` with a plain-English reason (`"{ticker}: {reason}"`)
    and never raises out of this function; any other exception is a real
    bug in the family's own code and is allowed to propagate.

    `retry_if_empty` (new 2026-10-27): a real, observed Kalshi-API behavior
    -- `list_open_markets()` can come back with a genuinely empty list on a
    transient basis, for a series that normally has dozens of open markets,
    via a real `200 OK` with an empty `markets` array rather than an HTTP
    429/5xx `kalshi_client._get_json` would already retry on its own. This
    was first caught on 2026-10-27 when a scheduled run's real survey
    recorded zero open WTI-transformer and zero open Miami-temperature
    markets, and a same-session read-only re-check moments later found 135
    and 12 real open markets respectively -- the live data was there the
    whole time, the first fetch just came back empty. When `retry_if_empty`
    is true (the default), an initial empty result triggers exactly one
    retry after `empty_retry_delay` seconds before being accepted; a
    genuinely-empty series still comes back empty on the retry and nothing
    about the returned `(surveyed, skipped)` changes, so this is a no-op
    safety net when zero really is correct and a real bug fix when it
    isn't. It does not retry a non-empty result, and it does not retry
    more than once -- this guards against a single transient empty blip,
    not a sustained real outage.
    """
    surveyed: list[SurveyedEdge] = []
    skipped: list[str] = []

    open_markets = family.list_open_markets()
    if retry_if_empty and not open_markets:
        time.sleep(empty_retry_delay)
        open_markets = family.list_open_markets()

    for market in open_markets:
        try:
            pe = family.estimate_probability(market)
        except MarketNotScorable as exc:
            skipped.append(f"{market.ticker}: {exc}")
            continue

        estimate = compute_edge(
            our_probability=pe.our_probability,
            quote=market,
            strike_usd=pe.adjusted_strike,
            settlement_benchmark_matches_forecast=pe.settlement_benchmark_matches_forecast,
            min_volume_for_trust=min_volume_for_trust,
            forecast_horizon_days=pe.forecast_horizon_days,
            days_to_close=pe.days_to_close,
            basis_note=pe.basis_note,
        )

        if pe.adjusted_strike is not None and pe.reference_price is not None:
            strike_distance = pe.adjusted_strike - pe.reference_price
        else:
            strike_distance = float("nan")

        surveyed.append(
            SurveyedEdge(
                estimate=estimate,
                days_to_close=pe.days_to_close if pe.days_to_close is not None else float("nan"),
                horizon_used=pe.forecast_horizon_days if pe.forecast_horizon_days is not None else -1,
                strike_distance_usd=strike_distance,
            )
        )

    return surveyed, skipped
