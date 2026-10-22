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

# v0.9.0: the real WTI scoring logic (and the module-level `fetch_markets` /
# `fetch_settlement_brackets` / `fetch_wti_daily` / `fit_calibrated_forecaster_horizon`
# names these tests patch) moved to `commodity_forecast.markets.wti` -- see
# that module's own docstring and `edge_survey.py`'s updated module
# docstring for the full account. `survey_live_edges` itself is unchanged
# as a public, backward-compatible entry point, so every assertion below
# (skip reasons, counts, horizon picked, liquidity caveats, model_name
# behavior) is unchanged from before the refactor -- only the monkeypatch
# targets moved to match where the real code now lives.


def _load_open_ladder_quotes():
    """The real, captured, currently-open KXWTI daily ladder from
    2026-10-16 (30 real markets, all zero-volume -- this project's own
    documented real example of a freshly-listed, illiquid ladder)."""
    raw = json.load(open(FIXTURES / "kxwti_markets_live_2026-10-16.json"))
    return [_parse_market(m) for m in raw["markets"]]


def _load_settlement_history_quotes():
    raw = json.load(open(FIXTURES / "kxwti_settlement_history_2026-10-19.json"))
    return [_parse_market(m) for m in raw["markets"]]


def _frozen_spot_series_through_2026_10_14():
    """A frozen copy of this project's own real `data/wti_spot_daily.csv`
    cache as it stood through 2026-10-14 -- the exact state the three
    tests below were originally written and verified against (their own
    docstrings say so explicitly: "this project's own cached spot series
    ends 2026-10-14"). Loaded via the real `fetch_wti_daily`, not
    hand-built, so these tests exercise the real loading path -- just
    pointed at a fixture file instead of the live, evolving committed
    cache.

    Why this exists (v0.7.2): a real EIA data refresh (new real
    observations through 2026-10-20, after a week-plus publication gap)
    moved `data/wti_spot_daily.csv`'s own last-observed date forward,
    which silently broke all three tests below -- their fixture ladder
    closes 2026-10-19, so once the live cache's last-observed date passed
    that close time, every market in it looked like it had already closed
    and got skipped as "before the last observed spot date," collapsing
    `surveyed` from 30 to 0. Not a bug in `survey_live_edges` itself (it
    did exactly what it's supposed to: skip a market that's already
    closed) -- a real test-design fragility, now fixed by freezing the
    one input (the spot series) these tests actually need held constant,
    via `survey_live_edges`'s own `series=` parameter."""
    from commodity_forecast.data import fetch_wti_daily

    return fetch_wti_daily(cache_path=FIXTURES / "wti_spot_daily_through_2026-10-14.csv", refresh=False)


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
# (v0.9.0: the real scoring logic lives in `commodity_forecast.markets.wti`
# now, so the live network calls replaced below are patched there, not on
# `commodity_forecast.edge_survey` -- see this file's own top-of-file note.)


def test_survey_live_edges_scores_the_real_captured_open_ladder(monkeypatch):
    """Runs the real scoring pipeline (horizon-validated forecaster fit,
    basis correction, compute_edge) against a real, captured, currently-
    open KXWTI ladder (2026-10-16) and a real settlement-history fixture
    for the basis estimate -- only the live network calls are replaced,
    never the scoring logic itself.

    Pins `model_name="naive"` (v0.8.0): this test's assertions (horizon
    picked, liquidity caveat, count skipped) are all independent of which
    point model was fit, but pinning the pre-0.8.0 default here keeps this
    test's own intent -- a frozen-fixture regression check -- honest and
    decoupled from `survey_live_edges`'s own live default, which is
    `"transformer"`. See
    `test_survey_live_edges_defaults_to_the_transformer_model` below for a
    dedicated check of the new default itself."""
    open_quotes = _load_open_ladder_quotes()
    settlement_quotes = _load_settlement_history_quotes()

    monkeypatch.setattr("commodity_forecast.markets.wti.fetch_markets", lambda *a, **kw: open_quotes)
    monkeypatch.setattr(
        "commodity_forecast.markets.wti.fetch_settlement_brackets",
        lambda *a, **kw: bracket_settlements(settlement_quotes),
    )

    surveyed, skipped = survey_live_edges(series=_frozen_spot_series_through_2026_10_14(), model_name="naive")

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
        "commodity_forecast.markets.wti.fetch_markets", lambda *a, **kw: open_quotes + [unsupported]
    )
    monkeypatch.setattr(
        "commodity_forecast.markets.wti.fetch_settlement_brackets",
        lambda *a, **kw: bracket_settlements(settlement_quotes),
    )

    surveyed, skipped = survey_live_edges(series=_frozen_spot_series_through_2026_10_14(), model_name="naive")
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
        "commodity_forecast.markets.wti.fetch_markets", lambda *a, **kw: open_quotes + [too_far_out]
    )
    monkeypatch.setattr(
        "commodity_forecast.markets.wti.fetch_settlement_brackets",
        lambda *a, **kw: bracket_settlements(settlement_quotes),
    )

    surveyed, skipped = survey_live_edges(series=_frozen_spot_series_through_2026_10_14(), model_name="naive")
    assert len(surveyed) == 30
    assert len(skipped) == 1
    assert "KXWTI-TEST-FAROUT" in skipped[0]
    assert "exceeds every horizon" in skipped[0]


def test_survey_live_edges_passing_series_never_calls_fetch_wti_daily(monkeypatch):
    """The actual regression test for v0.7.2's fix: passing `series=`
    explicitly must bypass `fetch_wti_daily` entirely, not just override
    its result -- otherwise a test (or any other caller pinning the input
    deliberately) would still be silently coupled to whatever the real,
    live-refreshed cache currently contains, which is exactly the
    fragility this parameter exists to remove."""
    open_quotes = _load_open_ladder_quotes()
    settlement_quotes = _load_settlement_history_quotes()

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("fetch_wti_daily should not be called when series= is given")

    monkeypatch.setattr("commodity_forecast.markets.wti.fetch_markets", lambda *a, **kw: open_quotes)
    monkeypatch.setattr(
        "commodity_forecast.markets.wti.fetch_settlement_brackets",
        lambda *a, **kw: bracket_settlements(settlement_quotes),
    )
    monkeypatch.setattr("commodity_forecast.markets.wti.fetch_wti_daily", _must_not_be_called)

    surveyed, skipped = survey_live_edges(series=_frozen_spot_series_through_2026_10_14(), model_name="naive")
    assert skipped == []
    assert len(surveyed) == 30


# --- v0.8.0: the live default switched from always-naive to "transformer" ---


def test_survey_live_edges_rejects_unknown_model_name():
    """`model_name` is validated up front, the same discipline
    `forecast.fit_calibrated_forecaster_horizon` already uses for the same
    parameter -- a typo here should fail loudly, immediately, not after a
    real (possibly slow) model fit is already underway. Since v0.9.0 this
    validation happens in `WTIMarketFamily.__init__`, reached as soon as
    `survey_live_edges` constructs one."""
    with pytest.raises(ValueError, match="model_name"):
        survey_live_edges(series=_frozen_spot_series_through_2026_10_14(), model_name="bogus")


def test_survey_live_edges_defaults_to_the_transformer_model(monkeypatch):
    """Direct regression guard for the v0.8.0 default change: calling
    `survey_live_edges` with no `model_name` argument at all must fit the
    `"transformer"` model, not silently keep the old always-naive
    behavior. Spies on the real `fit_calibrated_forecaster_horizon` call
    (rather than asserting on output numbers, which are independent of the
    model for this fixture's caveats) to check the exact `model_name` it
    was actually called with."""
    open_quotes = _load_open_ladder_quotes()
    settlement_quotes = _load_settlement_history_quotes()
    seen_model_names = []

    import commodity_forecast.markets.wti as wti_module

    real_fit = wti_module.fit_calibrated_forecaster_horizon

    def _spy(*args, **kwargs):
        seen_model_names.append(kwargs.get("model_name"))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr("commodity_forecast.markets.wti.fetch_markets", lambda *a, **kw: open_quotes)
    monkeypatch.setattr(
        "commodity_forecast.markets.wti.fetch_settlement_brackets",
        lambda *a, **kw: bracket_settlements(settlement_quotes),
    )
    monkeypatch.setattr("commodity_forecast.markets.wti.fit_calibrated_forecaster_horizon", _spy)

    surveyed, skipped = survey_live_edges(series=_frozen_spot_series_through_2026_10_14())

    assert skipped == []
    assert len(surveyed) == 30
    # Exactly one real fit (one distinct horizon across this fixture's 30
    # markets -- see the horizon=10 assertion above), and it used the new
    # default.
    assert seen_model_names == ["transformer"]


def test_survey_live_edges_naive_and_transformer_give_real_different_probabilities(monkeypatch):
    """Confirms the model switch is real, not a silent no-op: running the
    same real fixture ladder through `model_name="naive"` vs. the new
    `model_name="transformer"` default must produce at least one real,
    different `our_probability` -- the same kind of direct regression
    guard `test_horizon_model_comparison.py`'s own
    `test_lstm_and_transformer_produce_real_different_predictions_than_naive`
    already uses for `fit_calibrated_forecaster_horizon` directly."""
    open_quotes = _load_open_ladder_quotes()
    settlement_quotes = _load_settlement_history_quotes()

    monkeypatch.setattr("commodity_forecast.markets.wti.fetch_markets", lambda *a, **kw: open_quotes)
    monkeypatch.setattr(
        "commodity_forecast.markets.wti.fetch_settlement_brackets",
        lambda *a, **kw: bracket_settlements(settlement_quotes),
    )

    naive_surveyed, _ = survey_live_edges(
        series=_frozen_spot_series_through_2026_10_14(), model_name="naive"
    )
    transformer_surveyed, _ = survey_live_edges(
        series=_frozen_spot_series_through_2026_10_14(), model_name="transformer"
    )

    naive_probs = [s.estimate.our_probability for s in naive_surveyed]
    transformer_probs = [s.estimate.our_probability for s in transformer_surveyed]
    assert len(naive_probs) == len(transformer_probs) == 30
    assert naive_probs != transformer_probs
