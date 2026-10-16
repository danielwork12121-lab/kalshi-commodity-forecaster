import pytest

from commodity_forecast.edge import (
    EdgeEstimate,
    compute_edge,
    load_paper_bets,
    market_implied_probability,
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
