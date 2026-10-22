"""New in v0.9.0: tests for the market-agnostic `markets.base` machinery
itself, exercised against a tiny, fully-synthetic `MarketFamily` -- the
actual point of splitting the interface out of `edge_survey.py` was to
make this machinery testable in isolation from any real WTI-specific
logic (which keeps its own, separately-tested coverage in
`tests/test_edge_survey.py`, now patched at `commodity_forecast.markets.wti`).
"""

from __future__ import annotations

import pytest

from commodity_forecast.kalshi_client import KalshiMarketQuote
from commodity_forecast.markets.base import (
    MarketFamily,
    MarketNotScorable,
    ProbabilityEstimate,
    SurveyedEdge,
    run_family_survey,
)


def _quote(
    ticker: str = "TOY-1",
    volume: float = 0.0,
    floor_strike: float | None = 100.0,
    yes_bid: float = 0.4,
    yes_ask: float = 0.6,
) -> KalshiMarketQuote:
    return KalshiMarketQuote(
        ticker=ticker,
        event_ticker="TOY",
        title="toy market",
        status="active",
        strike_type="greater",
        floor_strike=floor_strike,
        close_time="2026-11-01T00:00:00Z",
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=round(1.0 - yes_ask, 4),
        no_ask=round(1.0 - yes_bid, 4),
        volume=volume,
        open_interest=volume,
        result="",
    )


def test_market_family_is_abstract_and_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        MarketFamily()  # type: ignore[abstract]


class _ToyFamily(MarketFamily):
    """A tiny, fully-synthetic family used only to exercise
    `run_family_survey` in isolation -- no real forecaster, no real basis
    correction, no network calls of any kind."""

    series_ticker = "TOY"

    def __init__(self, quotes, unscorable_tickers=(), reference_price: float | None = 95.0):
        self._quotes = quotes
        self._unscorable = set(unscorable_tickers)
        self._reference_price = reference_price

    def list_open_markets(self):
        return self._quotes

    def estimate_probability(self, market: KalshiMarketQuote) -> ProbabilityEstimate:
        if market.ticker in self._unscorable:
            raise MarketNotScorable("synthetic: cannot score this one")
        return ProbabilityEstimate(
            our_probability=0.7,
            adjusted_strike=market.floor_strike,
            settlement_benchmark_matches_forecast=True,
            forecast_horizon_days=3,
            days_to_close=2.0,
            reference_price=self._reference_price,
        )


def test_run_family_survey_scores_every_real_market_the_family_can_handle():
    quotes = [_quote("TOY-1", volume=50.0), _quote("TOY-2", volume=0.0, floor_strike=120.0)]
    surveyed, skipped = run_family_survey(_ToyFamily(quotes))

    assert skipped == []
    assert len(surveyed) == 2
    assert all(isinstance(s, SurveyedEdge) for s in surveyed)

    first = surveyed[0]
    assert first.estimate.our_probability == pytest.approx(0.7)
    assert first.estimate.market_probability == pytest.approx(0.5)  # (0.4 + 0.6) / 2
    assert first.estimate.edge == pytest.approx(0.2)  # 0.7 - 0.5
    assert first.strike_distance_usd == pytest.approx(100.0 - 95.0)
    assert first.horizon_used == 3
    assert first.days_to_close == pytest.approx(2.0)

    second = surveyed[1]
    assert second.strike_distance_usd == pytest.approx(120.0 - 95.0)


def test_run_family_survey_records_a_plain_english_skip_reason_never_raises():
    quotes = [_quote("TOY-OK"), _quote("TOY-BAD")]
    surveyed, skipped = run_family_survey(_ToyFamily(quotes, unscorable_tickers=["TOY-BAD"]))

    assert len(surveyed) == 1
    assert surveyed[0].estimate.kalshi_ticker == "TOY-OK"
    assert len(skipped) == 1
    assert "TOY-BAD" in skipped[0]
    assert "synthetic: cannot score this one" in skipped[0]


def test_run_family_survey_marks_illiquid_market_untrustworthy_via_min_volume_for_trust():
    quotes = [_quote("TOY-ILLIQUID", volume=0.0)]
    surveyed, skipped = run_family_survey(_ToyFamily(quotes), min_volume_for_trust=1.0)

    assert skipped == []
    assert surveyed[0].estimate.trustworthy is False
    assert any("liquidity" in c for c in surveyed[0].estimate.caveats)


def test_run_family_survey_an_unexpected_exception_is_not_swallowed_as_a_skip():
    """`run_family_survey` only catches `MarketNotScorable` -- a real bug
    in a family's own `estimate_probability` (here, a bare `ValueError`)
    must still propagate as a real error, not get silently folded into
    `skipped` as if it were an honestly-declined market."""

    class _BuggyFamily(MarketFamily):
        series_ticker = "TOY"

        def list_open_markets(self):
            return [_quote("TOY-BUGGY")]

        def estimate_probability(self, market):
            raise ValueError("not a MarketNotScorable -- a real bug")

    with pytest.raises(ValueError, match="real bug"):
        run_family_survey(_BuggyFamily())


def test_strike_distance_is_nan_not_silently_zero_when_no_reference_price():
    """A family with no meaningful reference price (`reference_price=None`)
    must not have `run_family_survey` quietly treat that as a reference of
    0.0 -- "no reference" and "the market sits right at the money" are
    different, real facts, and the strike_distance_usd field must not
    conflate them."""
    quotes = [_quote("TOY-NO-REF")]
    surveyed, _ = run_family_survey(_ToyFamily(quotes, reference_price=None))

    assert len(surveyed) == 1
    assert surveyed[0].strike_distance_usd != surveyed[0].strike_distance_usd  # NaN != NaN


def test_settle_default_implementation_delegates_to_the_real_kalshi_client(monkeypatch):
    """`MarketFamily.settle`'s default implementation is generic across
    every family -- a ticker's settlement lookup doesn't depend on which
    family scored it -- so confirm it really calls through to
    `kalshi_client.fetch_market` rather than, say, silently returning the
    same (possibly stale) quote it was given."""
    seen_tickers = []

    def _fake_fetch_market(ticker):
        seen_tickers.append(ticker)
        return _quote(ticker, volume=999.0)

    monkeypatch.setattr("commodity_forecast.kalshi_client.fetch_market", _fake_fetch_market)

    family = _ToyFamily([])
    result = family.settle(_quote("TOY-TO-SETTLE"))

    assert seen_tickers == ["TOY-TO-SETTLE"]
    assert result.volume == 999.0
