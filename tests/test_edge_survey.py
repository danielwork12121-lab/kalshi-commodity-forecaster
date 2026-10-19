import json
import pathlib

import pytest

from commodity_forecast.benchmark import bracket_settlements
from commodity_forecast.edge import EdgeEstimate
from commodity_forecast.edge_survey import (
    EdgeSubsetSummary,
    SurveyedEdge,
    format_edge_subset_summary,
    summarize_edge_subsets,
    survey_live_edges,
)
from commodity_forecast.kalshi_client import KalshiMarketQuote, _parse_market

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load_open_ladder_quotes():
    """The real, captured, currently-open KXWTI daily ladder from
    2026-10-16 (30 real markets, all zero-volume -- this project's own
    documented real example of a freshly-listed, illiquid ladder)."""
    raw = json.load(open(FIXTURES / "kxwti_markets_live_2026-10-16.json"))
    return [_parse_market(m) for m in raw["markets"]]


def _load_settlement_history_quotes():
    raw = json.load(open(FIXTURES / "kxwti_settlement_history_2026-10-19.json"))
    return [_parse_market(m) for m in raw["markets"]]


def _estimate(edge: float, trustworthy: bool, volume: float = 100.0, **overrides) -> EdgeEstimate:
    defaults = dict(
        computed_at_utc="2026-10-19T00:00:00+00:00",
        kalshi_ticker="KXWTI-TEST",
        kalshi_title="test",
        strike_usd=100.0,
        kalshi_close_time="2026-10-19T18:30:00Z",
        our_probability=0.5 + edge / 2,
        market_probability=0.5 - edge / 2,
        market_yes_bid=0.49,
        market_yes_ask=0.51,
        market_volume=volume,
        market_open_interest=volume,
        edge=edge,
        caveats=[] if trustworthy else ["synthetic caveat for this test"],
    )
    defaults.update(overrides)
    return EdgeEstimate(**defaults)


def _surveyed(edge: float, trustworthy: bool, strike_distance_usd: float, horizon_used: int, volume: float = 100.0) -> SurveyedEdge:
    return SurveyedEdge(
        estimate=_estimate(edge, trustworthy, volume=volume),
        days_to_close=float(horizon_used),
        horizon_used=horizon_used,
        strike_distance_usd=strike_distance_usd,
    )


# --- summarize_edge_subsets / format_edge_subset_summary: pure unit tests, no network ---


def test_summarize_edge_subsets_splits_on_strike_distance():
    surveyed = [
        _surveyed(edge=0.10, trustworthy=True, strike_distance_usd=0.5, horizon_used=1),
        _surveyed(edge=0.30, trustworthy=True, strike_distance_usd=5.0, horizon_used=1),
    ]
    summaries = {s.label: s for s in summarize_edge_subsets(surveyed)}
    assert summaries["near-the-money"].n == 1
    assert summaries["near-the-money"].mean_edge == pytest.approx(0.10)
    assert summaries["far-out-of-the-money"].n == 1
    assert summaries["far-out-of-the-money"].mean_edge == pytest.approx(0.30)
    assert summaries["all"].n == 2
    assert summaries["all"].mean_edge == pytest.approx(0.20)


def test_summarize_edge_subsets_splits_on_horizon():
    surveyed = [
        _surveyed(edge=0.05, trustworthy=True, strike_distance_usd=0.0, horizon_used=1),
        _surveyed(edge=-0.05, trustworthy=True, strike_distance_usd=0.0, horizon_used=21),
    ]
    summaries = {s.label: s for s in summarize_edge_subsets(surveyed)}
    assert summaries["short-horizon"].n == 1
    assert summaries["short-horizon"].mean_edge == pytest.approx(0.05)
    assert summaries["long-horizon"].n == 1
    assert summaries["long-horizon"].mean_edge == pytest.approx(-0.05)


def test_summarize_edge_subsets_splits_on_liquidity():
    surveyed = [
        _surveyed(edge=0.10, trustworthy=True, strike_distance_usd=0.0, horizon_used=1, volume=500.0),
        _surveyed(edge=0.20, trustworthy=True, strike_distance_usd=0.0, horizon_used=1, volume=0.0),
    ]
    summaries = {s.label: s for s in summarize_edge_subsets(surveyed)}
    assert summaries["liquid"].n == 1
    assert summaries["liquid"].mean_edge == pytest.approx(0.10)
    assert summaries["illiquid"].n == 1
    assert summaries["illiquid"].mean_edge == pytest.approx(0.20)


def test_summarize_edge_subsets_excludes_untrustworthy_by_default():
    surveyed = [
        _surveyed(edge=0.90, trustworthy=False, strike_distance_usd=0.0, horizon_used=1),
    ]
    summaries = {s.label: s for s in summarize_edge_subsets(surveyed)}
    assert summaries["all"].n == 0
    # The untrustworthy estimate is recoverable with trustworthy_only=False,
    # never silently discarded -- it's just excluded from the honest default.
    summaries_raw = {s.label: s for s in summarize_edge_subsets(surveyed, trustworthy_only=False)}
    assert summaries_raw["all"].n == 1
    assert summaries_raw["all"].mean_edge == pytest.approx(0.90)


def test_format_edge_subset_summary_reports_empty_subsets_as_na_not_hidden():
    summaries = [EdgeSubsetSummary(label="empty-subset", n=0, mean_edge=0.0, mean_abs_edge=0.0)]
    text = format_edge_subset_summary(summaries)
    assert "empty-subset" in text
    assert "n/a" in text


# --- survey_live_edges: a real end-to-end run against real, captured fixture data ---


def test_survey_live_edges_scores_the_real_captured_open_ladder(monkeypatch):
    """Runs the real scoring pipeline (horizon-validated forecaster fit,
    basis correction, compute_edge) against a real, captured, currently-
    open KXWTI ladder (2026-10-16) and a real settlement-history fixture
    for the basis estimate -- only the live network calls are replaced,
    never the scoring logic itself."""
    open_quotes = _load_open_ladder_quotes()
    settlement_quotes = _load_settlement_history_quotes()

    monkeypatch.setattr("commodity_forecast.edge_survey.fetch_markets", lambda *a, **kw: open_quotes)
    monkeypatch.setattr(
        "commodity_forecast.edge_survey.fetch_settlement_brackets",
        lambda *a, **kw: bracket_settlements(settlement_quotes),
    )

    surveyed, skipped = survey_live_edges(refresh_spot=False)

    # All 30 real markets in this fixture are "greater than $X" questions
    # closing on a real, validatable horizon (this project's own cached
    # spot series ends 2026-10-14; this ladder closes 2026-10-19, ~5.8
    # real days out -- within the 1..21-day validated range) -- nothing
    # should be skipped here.
    assert skipped == []
    assert len(surveyed) == 30

    # This fixture is this project's own documented real example of a
    # freshly-listed, zero-volume ladder -- every real estimate here must
    # carry the honest low/no-liquidity caveat, so none are "trustworthy".
    assert all(not s.estimate.trustworthy for s in surveyed)
    assert all("low/no liquidity" in c for s in surveyed for c in s.estimate.caveats if "liquidity" in c)
    assert all(s.horizon_used == 10 for s in surveyed)  # ceil(5.8) -> smallest validated horizon >= 5.8

    # The real strikes in this fixture span a real range around the real
    # last-observed spot price ($101.91, 2026-10-14) -- confirm the basis-
    # adjusted strike distance is computed from real numbers, not zeroed out.
    distances = [s.strike_distance_usd for s in surveyed]
    assert len(set(round(d, 2) for d in distances)) > 1  # genuinely different strikes, not a constant


def test_survey_live_edges_skips_a_market_with_an_unsupported_strike_type(monkeypatch):
    open_quotes = _load_open_ladder_quotes()
    unsupported = KalshiMarketQuote(
        ticker="KXWTI-TEST-RANGE",
        event_ticker="KXWTI-TEST",
        title="range market",
        status="active",
        strike_type="between",
        floor_strike=None,
        close_time="2026-10-19T18:30:00Z",
        yes_bid=0.5,
        yes_ask=0.5,
        no_bid=0.5,
        no_ask=0.5,
        volume=0.0,
        open_interest=0.0,
        result="",
    )
    settlement_quotes = _load_settlement_history_quotes()

    monkeypatch.setattr(
        "commodity_forecast.edge_survey.fetch_markets", lambda *a, **kw: open_quotes + [unsupported]
    )
    monkeypatch.setattr(
        "commodity_forecast.edge_survey.fetch_settlement_brackets",
        lambda *a, **kw: bracket_settlements(settlement_quotes),
    )

    surveyed, skipped = survey_live_edges(refresh_spot=False)
    assert len(surveyed) == 30
    assert len(skipped) == 1
    assert "KXWTI-TEST-RANGE" in skipped[0]
    assert "not a simple" in skipped[0]


def test_survey_live_edges_skips_a_market_beyond_every_validated_horizon(monkeypatch):
    open_quotes = _load_open_ladder_quotes()
    too_far_out = KalshiMarketQuote(
        ticker="KXWTI-TEST-FAROUT",
        event_ticker="KXWTI-TEST",
        title="far-out market",
        status="active",
        strike_type="greater",
        floor_strike=100.0,
        close_time="2026-12-19T18:30:00Z",  # ~66 real days out -- beyond the 21-day validated max
        yes_bid=0.5,
        yes_ask=0.5,
        no_bid=0.5,
        no_ask=0.5,
        volume=10.0,
        open_interest=10.0,
        result="",
    )
    settlement_quotes = _load_settlement_history_quotes()

    monkeypatch.setattr(
        "commodity_forecast.edge_survey.fetch_markets", lambda *a, **kw: open_quotes + [too_far_out]
    )
    monkeypatch.setattr(
        "commodity_forecast.edge_survey.fetch_settlement_brackets",
        lambda *a, **kw: bracket_settlements(settlement_quotes),
    )

    surveyed, skipped = survey_live_edges(refresh_spot=False)
    assert len(surveyed) == 30
    assert len(skipped) == 1
    assert "KXWTI-TEST-FAROUT" in skipped[0]
    assert "exceeds every horizon" in skipped[0]
