from __future__ import annotations

import datetime as _dt

import pytest

import commodity_forecast.markets.temperature as temp_mod
from commodity_forecast.benchmark import BasisMeasurement
from commodity_forecast.kalshi_client import KalshiMarketQuote
from commodity_forecast.markets.base import MarketNotScorable
from commodity_forecast.markets.temperature import (
    TemperatureBracket,
    TemperatureMarketFamily,
    backtest_climatology_coverage,
    climatological_samples,
    describe_temperature_basis_correction,
    empirical_exceedance_probability,
    fetch_real_settled_brackets,
    measure_temperature_basis,
    parse_ticker_settlement_date,
)


# ---------------------------------------------------------------------------
# Ticker date parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ticker,expected",
    [
        ("KXHIGHNY-26OCT23-T64", "2026-10-23"),
        ("KXHIGHNY-26OCT23-T57", "2026-10-23"),
        ("KXHIGHNY-26OCT23-B63.5", "2026-10-23"),
        ("KXHIGHCHI-26JAN05-T30", "2026-01-05"),
        ("KXHIGHNY-26DEC31-T40", "2026-12-31"),
    ],
)
def test_parse_ticker_settlement_date_handles_every_real_strike_shape(ticker, expected):
    assert parse_ticker_settlement_date(ticker) == expected


def test_parse_ticker_settlement_date_returns_none_for_an_unrecognized_format():
    assert parse_ticker_settlement_date("not-a-real-ticker") is None
    assert parse_ticker_settlement_date("KXHIGHNY-26XYZ23-T64") is None  # bad month code


# ---------------------------------------------------------------------------
# Real-settled-bracket extraction and basis measurement
# ---------------------------------------------------------------------------


def _quote(ticker, strike_type, floor_strike=None, cap_strike=None, result="", status="finalized"):
    return KalshiMarketQuote(
        ticker=ticker,
        event_ticker=ticker.rsplit("-", 1)[0],
        title="t",
        status=status,
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        close_time="2026-10-24T05:00:00Z",
        yes_bid=0.4,
        yes_ask=0.6,
        no_bid=0.4,
        no_ask=0.6,
        volume=100.0,
        open_interest=100.0,
        result=result,
    )


def test_fetch_real_settled_brackets_keeps_only_yes_settled_between_markets(monkeypatch):
    quotes = [
        _quote("KXHIGHNY-26OCT21-T66", "greater", floor_strike=66, result="no"),
        _quote("KXHIGHNY-26OCT21-B65.5", "between", floor_strike=65, cap_strike=66, result="no"),
        _quote("KXHIGHNY-26OCT21-B61.5", "between", floor_strike=61, cap_strike=62, result="yes"),
        _quote("KXHIGHNY-26OCT20-B67.5", "between", floor_strike=67, cap_strike=68, result="yes"),
        # A "between" market that hasn't settled yet should never be treated as a real bracket:
        _quote("KXHIGHNY-26OCT25-B70.5", "between", floor_strike=70, cap_strike=71, result="", status="active"),
    ]
    monkeypatch.setattr(temp_mod, "fetch_markets", lambda *a, **k: quotes)

    brackets = fetch_real_settled_brackets("KXHIGHNY")

    assert brackets == [
        TemperatureBracket(date="2026-10-20", floor_strike=67.0, cap_strike=68.0),
        TemperatureBracket(date="2026-10-21", floor_strike=61.0, cap_strike=62.0),
    ]


def test_measure_temperature_basis_pairs_real_brackets_with_matching_open_meteo_dates():
    brackets = [
        TemperatureBracket(date="2026-10-19", floor_strike=62.0, cap_strike=63.0),
        TemperatureBracket(date="2026-10-20", floor_strike=67.0, cap_strike=68.0),
        TemperatureBracket(date="2026-10-21", floor_strike=61.0, cap_strike=62.0),
    ]
    om_series = {"2026-10-19": 63.0, "2026-10-20": 66.7}  # 10-21 deliberately missing

    measurements = measure_temperature_basis(brackets, om_series)

    assert len(measurements) == 2  # the unmatched 10-21 bracket is silently skipped, never interpolated
    by_date = {m.date: m for m in measurements}
    assert by_date["2026-10-19"].basis == pytest.approx(62.5 - 63.0)
    assert by_date["2026-10-20"].basis == pytest.approx(67.5 - 66.7)


@pytest.mark.parametrize(
    "stdev,n,expected_trustworthy",
    [
        (0.5, 10, True),
        (1.0, 10, True),  # exactly at the bar
        (1.1, 10, False),  # the real measured value (2026-10-22 run) -- fails the bar, honestly
        (0.3, 4, False),  # tight but too few real points
    ],
)
def test_describe_temperature_basis_correction_applies_the_same_bar_as_wti(stdev, n, expected_trustworthy):
    trustworthy, note = describe_temperature_basis_correction(0.3, stdev, n)
    assert trustworthy is expected_trustworthy
    assert "degF" in note
    assert "Weather Company" in note


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------


def test_climatological_samples_collects_real_values_within_the_window():
    series = {
        "2024-10-20": 60.0,
        "2024-10-27": 70.0,  # 7 days away -- just outside a default 5-day window
        "2025-10-18": 61.0,
        "2025-10-22": 62.0,
    }
    samples = climatological_samples(series, "2026-10-20", window_days=5)
    assert sorted(samples) == [60.0, 61.0, 62.0]


def test_climatological_samples_handles_the_new_years_wraparound():
    series = {"2024-12-30": 40.0, "2025-01-02": 41.0, "2025-06-15": 90.0}
    samples = climatological_samples(series, "2026-01-01", window_days=3)
    assert sorted(samples) == [40.0, 41.0]


def test_climatological_samples_exclude_year_drops_only_that_real_year():
    series = {"2023-10-20": 50.0, "2024-10-20": 60.0, "2025-10-20": 70.0}
    samples = climatological_samples(series, "2026-10-20", window_days=1, exclude_year=2024)
    assert sorted(samples) == [50.0, 70.0]


def test_empirical_exceedance_probability_basic_cases():
    samples = [50.0, 55.0, 60.0, 65.0, 70.0]
    assert empirical_exceedance_probability(samples, 45.0) == pytest.approx(1.0)
    assert empirical_exceedance_probability(samples, 75.0) == pytest.approx(0.0)
    assert empirical_exceedance_probability(samples, 60.0) == pytest.approx(2 / 5)  # strictly >


def test_empirical_exceedance_probability_raises_on_no_samples():
    with pytest.raises(ValueError):
        empirical_exceedance_probability([], 50.0)


def test_backtest_climatology_coverage_on_a_real_wide_uniform_synthetic_series_is_reasonably_calibrated():
    """Not a real historical series -- a synthetic one with a known,
    stationary (non-seasonal) uniform distribution, specifically so the
    leave-one-year-out empirical-quantile machinery's own correctness can
    be checked against ground truth we actually know, independent of
    whatever real NYC temperature turns out to look like."""
    import random

    rng = random.Random(0)
    series = {}
    for year in range(2000, 2021):
        for doy_offset in range(-20, 21):
            d = (_dt.date(2026, 7, 1) + _dt.timedelta(days=doy_offset)).replace(year=year)
            series[d.isoformat()] = 50.0 + rng.uniform(-10.0, 10.0)

    results = backtest_climatology_coverage(series, window_days=20, alphas=(0.2,), eval_every=1)
    assert len(results) == 1
    r = results[0]
    assert r.n_evaluations > 50
    # A real stationary uniform series' own 80% empirical interval should
    # measure close to 80% real coverage -- allow real sampling slack.
    assert 0.65 < r.empirical_coverage < 0.95


# ---------------------------------------------------------------------------
# TemperatureMarketFamily end-to-end (offline: frozen series, monkeypatched
# bracket fetch -- no real network)
# ---------------------------------------------------------------------------


def _frozen_climatology_series() -> dict[str, float]:
    """A real-shaped (but synthetic) 10-year daily series centered on a
    known, deterministic late-October climatology so test assertions can
    reason about the exact expected probability."""
    series = {}
    for year in range(2016, 2026):
        for offset in range(-10, 11):
            d = _dt.date(year, 10, 23) + _dt.timedelta(days=offset)
            # A simple deterministic value: a seasonal baseline of 62, so
            # every real year contributes the same, known samples.
            series[d.isoformat()] = 62.0 + (offset % 3) - 1  # values in {61, 62, 63}
    return series


def _family_with_no_trustworthy_basis(monkeypatch, **kwargs) -> TemperatureMarketFamily:
    monkeypatch.setattr(temp_mod, "fetch_real_settled_brackets", lambda *a, **k: [])
    return TemperatureMarketFamily(series=_frozen_climatology_series(), window_days=10, **kwargs)


def test_family_raises_market_not_scorable_for_an_unparsable_ticker(monkeypatch):
    family = _family_with_no_trustworthy_basis(monkeypatch)
    bad = _quote("garbage-ticker", "greater", floor_strike=60.0, status="active")
    with pytest.raises(MarketNotScorable):
        family.estimate_probability(bad)


def test_family_raises_market_not_scorable_for_a_missing_floor_or_cap(monkeypatch):
    family = _family_with_no_trustworthy_basis(monkeypatch)
    broken = _quote("KXHIGHNY-26OCT23-T64", "greater", floor_strike=None, status="active")
    with pytest.raises(MarketNotScorable):
        family.estimate_probability(broken)


def test_family_scores_a_greater_market_with_the_continuity_correction(monkeypatch):
    family = _family_with_no_trustworthy_basis(monkeypatch)
    # floor=62 -> continuity-corrected threshold 62.5. Real samples are {61,62,63}
    # evenly split across 3 possible offsets; values > 62.5 are only the 63s.
    market = _quote("KXHIGHNY-26OCT23-T62", "greater", floor_strike=62.0, status="active")
    pe = family.estimate_probability(market)
    assert pe.adjusted_strike == pytest.approx(62.5)
    assert pe.our_probability == pytest.approx(1 / 3, abs=0.01)
    assert pe.settlement_benchmark_matches_forecast is False  # no real brackets -> not trustworthy


def test_family_scores_a_less_market_with_the_continuity_correction(monkeypatch):
    family = _family_with_no_trustworthy_basis(monkeypatch)
    # cap=62 -> continuity-corrected threshold 61.5. P(X < 61.5) is only the 61s.
    market = _quote("KXHIGHNY-26OCT23-T62", "less", cap_strike=62.0, status="active")
    pe = family.estimate_probability(market)
    assert pe.adjusted_strike == pytest.approx(61.5)
    assert pe.our_probability == pytest.approx(1 / 3, abs=0.01)


def test_family_scores_a_between_market_as_the_difference_of_two_exceedance_probabilities(monkeypatch):
    family = _family_with_no_trustworthy_basis(monkeypatch)
    # floor=61, cap=62 -> P(60.5 < X < 62.5), which is exactly the 61s and 62s (2/3).
    market = _quote("KXHIGHNY-26OCT23-B61.5", "between", floor_strike=61.0, cap_strike=62.0, status="active")
    pe = family.estimate_probability(market)
    assert pe.our_probability == pytest.approx(2 / 3, abs=0.01)


def test_family_raises_market_not_scorable_when_too_few_climatology_samples(monkeypatch):
    monkeypatch.setattr(temp_mod, "fetch_real_settled_brackets", lambda *a, **k: [])
    sparse_series = {"2026-10-20": 60.0, "2026-10-21": 61.0}  # only 2 real points anywhere
    family = TemperatureMarketFamily(series=sparse_series, window_days=10)
    market = _quote("KXHIGHNY-26OCT23-T60", "greater", floor_strike=60.0, status="active")
    with pytest.raises(MarketNotScorable):
        family.estimate_probability(market)


def test_family_applies_a_real_measured_trustworthy_basis_correction(monkeypatch):
    """When `fetch_real_settled_brackets` returns enough tight, consistent
    real-shaped measurements, the family should trust and apply the real
    basis correction -- shifting the effective threshold, not leaving it
    at the market's raw strike."""
    series = _frozen_climatology_series()
    # 6 real-shaped brackets whose midpoint sits a real, consistent +2.0 degF
    # above this series' own real October values (62ish) -- a tight, trustworthy signal.
    brackets = [
        TemperatureBracket(date=f"2025-10-1{i}", floor_strike=63.5, cap_strike=64.5) for i in range(1, 7)
    ]
    for b in brackets:
        series[b.date] = 62.0  # force the exact known value (these dates already exist in the frozen series)
    monkeypatch.setattr(temp_mod, "fetch_real_settled_brackets", lambda *a, **k: brackets)

    family = TemperatureMarketFamily(series=series, window_days=10)
    assert family._basis_trustworthy is True
    assert family._basis_mean == pytest.approx(2.0, abs=0.01)

    market = _quote("KXHIGHNY-26OCT23-T62", "greater", floor_strike=62.0, status="active")
    pe = family.estimate_probability(market)
    # adjusted_strike = basis_adjusted_strike(62.5, 2.0) = 62.5 - 2.0 = 60.5
    assert pe.adjusted_strike == pytest.approx(60.5)
    assert pe.settlement_benchmark_matches_forecast is True


def test_family_real_live_kxhighny_end_to_end():
    """Integration test: actually builds a real `TemperatureMarketFamily`
    against the real, live Kalshi `KXHIGHNY` series and the real, live
    Open-Meteo archive -- no mocking, same discipline as every other real
    live-API test in this suite (`test_kalshi_client.py`,
    `test_temperature_data.py`). Scores every real currently-open market;
    doesn't assert a specific probability (that's real, live, and will
    keep changing), just that the real pipeline runs end-to-end without
    error and produces sane, in-range real probabilities."""
    from commodity_forecast.markets.base import run_family_survey

    family = TemperatureMarketFamily()
    surveyed, skipped = run_family_survey(family)

    assert len(surveyed) + len(skipped) > 0
    for s in surveyed:
        assert 0.0 <= s.estimate.our_probability <= 1.0
