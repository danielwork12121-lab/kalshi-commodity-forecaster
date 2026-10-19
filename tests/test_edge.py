import pytest

from commodity_forecast.edge import (
    VALIDATED_FORECAST_HORIZONS_DAYS,
    EdgeEstimate,
    compute_edge,
    days_to_close_from_last_observation,
    load_paper_bets,
    market_implied_probability,
    pick_validated_horizon,
    resolve_pending_paper_bets,
    size_paper_bet,
    append_paper_bet,
    _kelly_fraction,
)
from commodity_forecast.kalshi_client import KalshiMarketQuote


def _quote(**overrides) -> KalshiMarketQuote:
    defaults = dict(
        ticker="KXWTI-26OCT1914-T95.49",
        event_ticker="KXWTI-26OCT1914",
        title="Will WTI settle above $95.49 on Oct 19, 2026?",
        status="active",
        strike_type="greater",
        floor_strike=95.49,
        close_time="2026-10-19T18:30:00Z",
        yes_bid=0.44,
        yes_ask=0.48,
        no_bid=0.52,
        no_ask=0.56,
        volume=0.0,
        open_interest=0.0,
        result="",
    )
    defaults.update(overrides)
    return KalshiMarketQuote(**defaults)


def test_market_implied_probability_is_the_bid_ask_midpoint():
    q = _quote(yes_bid=0.40, yes_ask=0.50)
    assert market_implied_probability(q) == pytest.approx(0.45)


def test_compute_edge_flags_settlement_mismatch_by_default():
    q = _quote(volume=1000.0, open_interest=500.0)  # liquid -- isolate the benchmark-mismatch caveat
    est = compute_edge(our_probability=0.60, quote=q)
    assert est.edge == pytest.approx(0.60 - 0.46)  # mid of 0.44/0.48
    assert est.trustworthy is False
    assert any("settlement-benchmark mismatch" in c for c in est.caveats)
    assert not any("liquidity" in c for c in est.caveats)


def test_compute_edge_flags_illiquidity_separately():
    q = _quote(volume=0.0, open_interest=0.0)
    est = compute_edge(
        our_probability=0.60, quote=q, settlement_benchmark_matches_forecast=True
    )
    assert est.trustworthy is False
    assert any("low/no liquidity" in c for c in est.caveats)
    assert not any("settlement-benchmark mismatch" in c for c in est.caveats)


def test_compute_edge_trustworthy_when_benchmark_matches_and_liquid():
    q = _quote(volume=1000.0, open_interest=500.0)
    est = compute_edge(
        our_probability=0.60, quote=q, settlement_benchmark_matches_forecast=True, min_volume_for_trust=1.0
    )
    assert est.trustworthy is True
    assert est.caveats == []


def test_compute_edge_flags_horizon_mismatch_when_close_is_farther_than_validated():
    q = _quote(volume=1000.0, open_interest=500.0)
    est = compute_edge(
        our_probability=0.95,
        quote=q,
        settlement_benchmark_matches_forecast=True,
        forecast_horizon_days=1,
        days_to_close=15,
    )
    assert est.trustworthy is False
    assert any("forecast-horizon mismatch" in c for c in est.caveats)


def test_compute_edge_does_not_flag_horizon_mismatch_within_validated_horizon():
    q = _quote(volume=1000.0, open_interest=500.0)
    est = compute_edge(
        our_probability=0.60,
        quote=q,
        settlement_benchmark_matches_forecast=True,
        forecast_horizon_days=1,
        days_to_close=1,
    )
    assert est.trustworthy is True
    assert not any("horizon" in c for c in est.caveats)


def test_compute_edge_skips_horizon_check_when_either_value_is_none():
    q = _quote(volume=1000.0, open_interest=500.0)
    est = compute_edge(
        our_probability=0.60,
        quote=q,
        settlement_benchmark_matches_forecast=True,
        forecast_horizon_days=1,
        days_to_close=None,
    )
    assert est.trustworthy is True
    assert not any("horizon" in c for c in est.caveats)


def test_kelly_fraction_zero_when_no_edge_or_at_boundary_prices():
    assert _kelly_fraction(true_probability=0.5, price_usd=0.5) == pytest.approx(0.0)
    assert _kelly_fraction(true_probability=0.3, price_usd=0.5) == pytest.approx(0.0)  # negative edge -> clipped to 0
    assert _kelly_fraction(true_probability=0.9, price_usd=0.0) == pytest.approx(0.0)
    assert _kelly_fraction(true_probability=0.9, price_usd=1.0) == pytest.approx(0.0)


def test_kelly_fraction_matches_hand_computed_value():
    # f* = (p - c) / (1 - c) = (0.7 - 0.4) / (1 - 0.4) = 0.3/0.6 = 0.5
    assert _kelly_fraction(true_probability=0.7, price_usd=0.4) == pytest.approx(0.5)


def test_size_paper_bet_returns_none_when_not_trustworthy():
    q = _quote(volume=0.0, open_interest=0.0)
    est = compute_edge(our_probability=0.60, quote=q)  # untrustworthy by default (mismatch + illiquid)
    assert size_paper_bet(est) is None


def test_size_paper_bet_returns_none_when_no_real_edge_either_side():
    q = _quote(yes_bid=0.44, yes_ask=0.48, volume=1000.0, open_interest=500.0)
    # our_probability right at the mid -- essentially no edge on either side
    # once the real bid-ask spread (buy YES at ask, buy NO at 1-bid) is paid.
    est = compute_edge(our_probability=0.46, quote=q, settlement_benchmark_matches_forecast=True)
    assert est.trustworthy is True
    bet = size_paper_bet(est)
    assert bet is None


def test_size_paper_bet_sizes_a_real_yes_edge_and_caps_it():
    q = _quote(yes_bid=0.10, yes_ask=0.14, volume=1000.0, open_interest=500.0)
    est = compute_edge(our_probability=0.90, quote=q, settlement_benchmark_matches_forecast=True)
    assert est.trustworthy is True

    bet = size_paper_bet(est, kelly_shrinkage=0.25, fraction_cap=0.05)
    assert bet is not None
    assert bet.side == "yes"
    assert bet.entry_price_usd == pytest.approx(0.14)
    # raw kelly = (0.90 - 0.14) / (1 - 0.14) = 0.76/0.86 ~= 0.8837
    assert bet.kelly_fraction_raw == pytest.approx(0.8837, abs=1e-3)
    # quarter-Kelly would be ~0.2209, well above the 0.05 cap -> capped.
    assert bet.size_fraction == pytest.approx(0.05)
    assert bet.outcome == "pending"
    assert bet.resolved_at_utc is None


def test_size_paper_bet_sizes_a_no_bet_when_our_probability_is_low():
    q = _quote(yes_bid=0.80, yes_ask=0.84, volume=1000.0, open_interest=500.0)
    est = compute_edge(our_probability=0.10, quote=q, settlement_benchmark_matches_forecast=True)
    bet = size_paper_bet(est, kelly_shrinkage=0.25, fraction_cap=0.5)
    assert bet is not None
    assert bet.side == "no"
    # Buying NO costs 1 - yes_bid = 0.20; true NO probability = 1 - 0.10 = 0.90
    assert bet.entry_price_usd == pytest.approx(0.20)
    assert bet.our_probability == pytest.approx(0.90)


def test_append_and_load_paper_bets_roundtrip(tmp_path):
    path = tmp_path / "paper_bets.jsonl"
    q = _quote(yes_bid=0.10, yes_ask=0.14, volume=1000.0, open_interest=500.0)
    est = compute_edge(our_probability=0.90, quote=q, settlement_benchmark_matches_forecast=True)
    bet = size_paper_bet(est)
    append_paper_bet(bet, path=path)

    loaded = load_paper_bets(path=path)
    assert len(loaded) == 1
    assert loaded[0] == bet


def test_load_paper_bets_on_missing_file_returns_empty_list(tmp_path):
    assert load_paper_bets(path=tmp_path / "does_not_exist.jsonl") == []


def test_resolve_pending_paper_bets_resolves_won_and_lost(tmp_path):
    path = tmp_path / "paper_bets.jsonl"
    q = _quote(yes_bid=0.10, yes_ask=0.14, volume=1000.0, open_interest=500.0, ticker="T-YES-WINS")
    est = compute_edge(our_probability=0.90, quote=q, settlement_benchmark_matches_forecast=True)
    yes_bet = size_paper_bet(est)
    append_paper_bet(yes_bet, path=path)

    q2 = _quote(
        yes_bid=0.80, yes_ask=0.84, volume=1000.0, open_interest=500.0, ticker="T-NO-LOSES"
    )
    est2 = compute_edge(our_probability=0.10, quote=q2, settlement_benchmark_matches_forecast=True)
    no_bet = size_paper_bet(est2)
    append_paper_bet(no_bet, path=path)

    def fake_fetch_market(ticker):
        if ticker == "T-YES-WINS":
            return _quote(ticker=ticker, result="yes")
        if ticker == "T-NO-LOSES":
            return _quote(ticker=ticker, result="yes")  # the NO bet loses since YES actually happened
        raise AssertionError(f"unexpected ticker {ticker}")

    resolved = resolve_pending_paper_bets(fake_fetch_market, path=path)
    assert resolved[0].outcome == "won"
    assert resolved[0].resolved_kalshi_result == "yes"
    assert resolved[1].outcome == "lost"
    assert resolved[1].resolved_kalshi_result == "yes"

    reloaded = load_paper_bets(path=path)
    assert reloaded[0].outcome == "won"
    assert reloaded[1].outcome == "lost"


def test_resolve_pending_paper_bets_leaves_unsettled_markets_pending(tmp_path):
    path = tmp_path / "paper_bets.jsonl"
    q = _quote(yes_bid=0.10, yes_ask=0.14, volume=1000.0, open_interest=500.0, ticker="T-STILL-OPEN")
    est = compute_edge(our_probability=0.90, quote=q, settlement_benchmark_matches_forecast=True)
    bet = size_paper_bet(est)
    append_paper_bet(bet, path=path)

    def fake_fetch_market(ticker):
        return _quote(ticker=ticker, result="")  # not yet settled

    resolved = resolve_pending_paper_bets(fake_fetch_market, path=path)
    assert resolved[0].outcome == "pending"
    assert resolved[0].resolved_at_utc is None


def test_resolve_pending_paper_bets_never_retouches_already_resolved(tmp_path):
    path = tmp_path / "paper_bets.jsonl"
    q = _quote(yes_bid=0.10, yes_ask=0.14, volume=1000.0, open_interest=500.0, ticker="T-ALREADY-DONE")
    est = compute_edge(our_probability=0.90, quote=q, settlement_benchmark_matches_forecast=True)
    bet = size_paper_bet(est)
    bet.outcome = "won"
    bet.resolved_at_utc = "2026-10-01T00:00:00+00:00"
    bet.resolved_kalshi_result = "yes"
    append_paper_bet(bet, path=path)

    def should_not_be_called(ticker):
        raise AssertionError("fetch_market_fn should not be called for an already-resolved bet")

    resolved = resolve_pending_paper_bets(should_not_be_called, path=path)
    assert resolved[0] == bet


def test_pick_validated_horizon_rounds_up_to_the_nearest_validated_value():
    assert pick_validated_horizon(1) == 1
    assert pick_validated_horizon(2.65) == 3  # a real days_to_close value seen live on 2026-10-17
    assert pick_validated_horizon(4) == 5
    assert pick_validated_horizon(21) == 21


def test_pick_validated_horizon_returns_none_beyond_every_validated_value():
    assert pick_validated_horizon(22) is None
    assert pick_validated_horizon(90) is None


def test_pick_validated_horizon_uses_the_real_module_constant_by_default():
    # Every validated horizon this project actually ships should round-trip
    # to itself -- if it didn't, the constant and the rounding logic would
    # have silently drifted apart.
    for h in VALIDATED_FORECAST_HORIZONS_DAYS:
        assert pick_validated_horizon(h) == h


def test_compute_edge_horizon_caveat_clears_once_days_to_close_is_within_a_validated_horizon():
    """The real, concrete improvement this run's work makes: a market whose
    `days_to_close` is covered by `pick_validated_horizon` should no longer
    carry the forecast-horizon-mismatch caveat (the settlement-benchmark
    caveat is a separate, still-open issue and correctly still fires with
    its own default)."""
    q = _quote(close_time="2026-10-19T18:30:00Z", volume=1000.0, open_interest=500.0)
    days_to_close = 2.65
    horizon = pick_validated_horizon(days_to_close)
    assert horizon == 3

    est = compute_edge(
        our_probability=0.6,
        quote=q,
        forecast_horizon_days=horizon,
        days_to_close=days_to_close,
    )
    assert not any("forecast-horizon mismatch" in c for c in est.caveats)
    # settlement-benchmark mismatch is untouched by this change and should
    # still fire with compute_edge's own honest default.
    assert any("settlement-benchmark mismatch" in c for c in est.caveats)
    assert est.trustworthy is False


def test_compute_edge_horizon_caveat_still_fires_beyond_every_validated_horizon():
    q = _quote(close_time="2026-11-20T18:30:00Z", volume=1000.0, open_interest=500.0)
    days_to_close = 34.0
    horizon = pick_validated_horizon(days_to_close)
    assert horizon is None

    est = compute_edge(
        our_probability=0.6,
        quote=q,
        forecast_horizon_days=1,  # the only honest value left to pass when nothing validated covers it
        days_to_close=days_to_close,
    )
    assert any("forecast-horizon mismatch" in c for c in est.caveats)


def test_compute_edge_notes_default_empty_and_do_not_affect_trustworthy():
    q = _quote(volume=1000.0, open_interest=500.0)
    est = compute_edge(our_probability=0.5, quote=q, settlement_benchmark_matches_forecast=True)
    assert est.notes == []
    assert est.trustworthy is True


def test_compute_edge_basis_note_is_recorded_but_never_a_caveat():
    q = _quote(volume=1000.0, open_interest=500.0)
    note = "settlement-benchmark basis corrected using a real, measured recent-window estimate: mean -6.70 USD/bbl, stdev 0.15, n=10 real paired observations -- tight and well-supported enough to trust"
    est = compute_edge(
        our_probability=0.5,
        quote=q,
        settlement_benchmark_matches_forecast=True,
        basis_note=note,
    )
    assert est.notes == [note]
    assert not any("basis" in c for c in est.caveats)  # a note is never a caveat
    assert est.trustworthy is True


def test_compute_edge_basis_note_recorded_even_when_correction_not_trusted():
    # Caller decided NOT to flip settlement_benchmark_matches_forecast
    # (correction wasn't trustworthy), but still wants the real measured
    # numbers on the record for transparency.
    q = _quote(volume=1000.0, open_interest=500.0)
    note = "settlement-benchmark basis corrected using a real, measured recent-window estimate: mean -4.00 USD/bbl, stdev 2.50, n=10 real paired observations -- NOT trusted (too few points or too much spread)"
    est = compute_edge(our_probability=0.5, quote=q, basis_note=note)
    assert est.notes == [note]
    assert any("settlement-benchmark mismatch" in c for c in est.caveats)
    assert est.trustworthy is False


def test_days_to_close_from_last_observation_uses_data_origin_not_now():
    import datetime as dt

    # Real case this project hit 2026-10-19: last EIA observation 2026-10-14
    # (a stale, 5-real-day-old cache), a market closing 2026-10-20T18:30:00Z.
    days = days_to_close_from_last_observation("2026-10-20T18:30:00Z", "2026-10-14")
    assert days == pytest.approx(6.770833333333333, abs=1e-6)


def test_days_to_close_from_last_observation_accepts_datetime_like_object():
    import datetime as dt

    class _FakeTimestamp:
        def date(self):
            return dt.date(2026, 10, 14)

    days = days_to_close_from_last_observation("2026-10-20T18:30:00Z", _FakeTimestamp())
    assert days == pytest.approx(6.770833333333333, abs=1e-6)


def test_staleness_aware_horizon_is_wider_than_a_naive_now_based_one():
    # The real bug this fix addresses: a naive `close_time - now` (as if
    # the data were fresh) picks a much narrower, overconfident horizon
    # than the honest data-origin-based one.
    honest_days = days_to_close_from_last_observation("2026-10-20T18:30:00Z", "2026-10-14")
    honest_horizon = pick_validated_horizon(honest_days)
    naive_days = 1.05  # close_time - "now" (2026-10-19), ignoring staleness
    naive_horizon = pick_validated_horizon(naive_days)
    assert honest_horizon == 10
    assert naive_horizon == 2
    assert honest_horizon > naive_horizon
