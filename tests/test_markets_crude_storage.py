from __future__ import annotations

import datetime as _dt
import json
import pathlib

import pandas as pd
import pytest

import commodity_forecast.markets.crude_storage as cs_mod
from commodity_forecast.benchmark import BasisMeasurement
from commodity_forecast.kalshi_client import KalshiMarketQuote, _parse_market
from commodity_forecast.markets.base import MarketNotScorable
from commodity_forecast.markets.crude_storage import (
    VALIDATED_HORIZONS_WEEKS,
    CrudeStorageMarketFamily,
    describe_crude_storage_basis_correction,
    fetch_real_settled_values,
    measure_crude_storage_basis,
    parse_expiration_value_million_bbl,
    parse_target_week_ending_date,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return json.load(f)


def _quote(ticker, title, strike_type="greater", floor_strike=None, cap_strike=None,
           result="", status="finalized", close_time="2026-10-21T14:29:00Z",
           expiration_value=None, volume=0.0):
    return KalshiMarketQuote(
        ticker=ticker,
        event_ticker=ticker.rsplit("-T", 1)[0],
        title=title,
        status=status,
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        close_time=close_time,
        yes_bid=0.4,
        yes_ask=0.6,
        no_bid=0.4,
        no_ask=0.6,
        volume=volume,
        open_interest=volume,
        result=result,
        expiration_value=expiration_value,
    )


# ---------------------------------------------------------------------------
# parse_target_week_ending_date
# ---------------------------------------------------------------------------


def test_parse_target_week_ending_date_from_a_real_captured_title():
    title = "Will U.S. commercial crude oil inventories for the week ending Oct 16, 2026 be above 422 million barrels?"
    assert parse_target_week_ending_date(title) == _dt.date(2026, 10, 16)


def test_parse_target_week_ending_date_handles_single_digit_day():
    title = "Will U.S. commercial crude oil inventories for the week ending Oct 2, 2026 be above 434 million barrels?"
    assert parse_target_week_ending_date(title) == _dt.date(2026, 10, 2)


def test_parse_target_week_ending_date_returns_none_for_unparsable_title():
    assert parse_target_week_ending_date("not a real title at all") is None


# ---------------------------------------------------------------------------
# parse_expiration_value_million_bbl -- the real, 3-format Kalshi quirk
# ---------------------------------------------------------------------------


def test_parse_expiration_value_with_m_suffix():
    assert parse_expiration_value_million_bbl("412.5M") == pytest.approx(412.5)


def test_parse_expiration_value_without_m_suffix_already_in_million_barrels():
    assert parse_expiration_value_million_bbl("415.6") == pytest.approx(415.6)


def test_parse_expiration_value_raw_thousand_barrels_no_suffix():
    """A real, observed Kalshi quirk: some real settled `KXEIACRUDEW`
    events report `expiration_value` as the EIA's own raw thousand-barrel
    figure, with no decimal point and no `M` suffix at all (e.g. a real
    `"424069"`, confirmed live 2026-10-22 against the
    `KXEIACRUDEW-26SEP09` event) -- inferred from the real, measured
    plausible range for this series (never within an order of magnitude of
    10,000 "million barrels"), not silently assumed."""
    assert parse_expiration_value_million_bbl("424069") == pytest.approx(424.069)


def test_parse_expiration_value_returns_none_for_empty_or_missing():
    assert parse_expiration_value_million_bbl(None) is None
    assert parse_expiration_value_million_bbl("") is None


def test_parse_expiration_value_returns_none_for_unparsable_text():
    assert parse_expiration_value_million_bbl("pending") is None


# ---------------------------------------------------------------------------
# fetch_real_settled_values -- against a real captured fixture
# ---------------------------------------------------------------------------


def test_fetch_real_settled_values_dedupes_by_event_and_parses_both_real_fixture_formats(monkeypatch):
    raw = _load_fixture("kxeiacrudew_markets_settled_live_2026-10-22.json")
    quotes = [_parse_market(m) for m in raw["markets"]]
    # Confirm the fixture itself really does carry both real expiration_value
    # formats this test depends on (an "M"-suffixed one and a bare one) --
    # if this ever stops being true the fixture was replaced with something
    # that no longer exercises the real quirk this test is about.
    formats = {q.expiration_value for q in quotes}
    assert any(v and v.endswith("M") for v in formats)
    assert any(v and not v.endswith("M") for v in formats)

    monkeypatch.setattr(cs_mod, "fetch_markets", lambda *a, **k: quotes)
    values = fetch_real_settled_values()

    assert len(values) == 2  # the fixture holds exactly 2 distinct real events
    by_date = dict(values)
    assert by_date[_dt.date(2026, 10, 16)] == pytest.approx(412.5)
    assert by_date[_dt.date(2026, 10, 9)] == pytest.approx(415.6)
    dates = list(by_date)
    assert dates == sorted(dates)  # ascending by date


# ---------------------------------------------------------------------------
# measure_crude_storage_basis
# ---------------------------------------------------------------------------


def test_measure_crude_storage_basis_pairs_and_computes_the_real_sign_convention():
    settled = [(_dt.date(2026, 10, 9), 415.6), (_dt.date(2026, 10, 16), 412.5)]
    series = pd.Series(
        {pd.Timestamp("2026-10-09"): 415.577, pd.Timestamp("2026-10-16"): 412.517}
    )
    measurements = measure_crude_storage_basis(settled, series)
    assert len(measurements) == 2
    by_date = {m.date: m for m in measurements}
    # basis = kalshi_value - our_cached_value
    assert by_date["2026-10-09"].basis == pytest.approx(415.6 - 415.577)
    assert by_date["2026-10-16"].basis == pytest.approx(412.5 - 412.517)
    assert all(m.futures_bracket_width == 0.0 for m in measurements)


def test_measure_crude_storage_basis_skips_a_week_with_no_matching_cached_observation():
    settled = [(_dt.date(2026, 10, 9), 415.6), (_dt.date(1999, 1, 1), 300.0)]
    series = pd.Series({pd.Timestamp("2026-10-09"): 415.577})
    measurements = measure_crude_storage_basis(settled, series)
    assert len(measurements) == 1
    assert measurements[0].date == "2026-10-09"


# ---------------------------------------------------------------------------
# describe_crude_storage_basis_correction -- trust gate
# ---------------------------------------------------------------------------


def test_describe_crude_storage_basis_correction_trusts_the_real_measured_value():
    """The real value this family actually measured live, 2026-10-22, across
    8 real recent settled events: mean -0.009, stdev 0.018, n=8 -- pinned
    here as an explicit real-world test case, same discipline as
    `test_markets_temperature.py`'s own pinned real `(stdev=1.1, n=10,
    trusted=False)` case (that family's basis failed its own trust bar;
    this family's real basis -- a genuinely different, much tighter
    real-world situation -- passes it)."""
    trusted, note = describe_crude_storage_basis_correction(-0.009, 0.018, 8)
    assert trusted is True
    assert "trust" in note


def test_describe_crude_storage_basis_correction_rejects_too_few_points():
    trusted, _ = describe_crude_storage_basis_correction(0.0, 0.01, 2)
    assert trusted is False


def test_describe_crude_storage_basis_correction_rejects_too_much_spread():
    trusted, _ = describe_crude_storage_basis_correction(0.0, 5.0, 20)
    assert trusted is False


# ---------------------------------------------------------------------------
# CrudeStorageMarketFamily end-to-end (offline: frozen series, monkeypatched
# settled-values fetch -- no real network)
# ---------------------------------------------------------------------------


def _frozen_weekly_series() -> pd.Series:
    """A real-shaped (but synthetic), perfectly flat 60-week series so the
    naive horizon=1 forecaster's point prediction is exactly known (the
    last observed value, since a flat series has zero real week-to-week
    change) and every calibration residual is exactly zero -- lets test
    assertions reason about the exact resulting probability rather than an
    approximate one."""
    dates = pd.date_range("2025-08-01", periods=60, freq="7D")
    return pd.Series([400.0] * 60, index=dates, name="crude_storage_million_bbl")


def _family_with_no_trustworthy_basis(monkeypatch, **kwargs) -> CrudeStorageMarketFamily:
    monkeypatch.setattr(cs_mod, "fetch_real_settled_values", lambda *a, **k: [])
    return CrudeStorageMarketFamily(series=_frozen_weekly_series(), **kwargs)


def test_family_raises_market_not_scorable_for_a_non_greater_strike_type(monkeypatch):
    family = _family_with_no_trustworthy_basis(monkeypatch)
    market = _quote(
        "KXEIACRUDEW-26OCT28-T418", "irrelevant title", strike_type="between", floor_strike=417.0, cap_strike=418.0
    )
    with pytest.raises(MarketNotScorable):
        family.estimate_probability(market)


def test_family_raises_market_not_scorable_for_an_unparsable_title(monkeypatch):
    family = _family_with_no_trustworthy_basis(monkeypatch)
    market = _quote("KXEIACRUDEW-26OCT28-T418", "not a real title", floor_strike=418.0)
    with pytest.raises(MarketNotScorable):
        family.estimate_probability(market)


def test_family_raises_market_not_scorable_for_an_unvalidated_horizon(monkeypatch):
    """The frozen series' last real date is 2025-09-26 (60 weeks from
    2025-08-01); a title targeting 2 real weeks out (not in
    `VALIDATED_HORIZONS_WEEKS == (1,)`) must be honestly skipped, never
    silently extrapolated."""
    family = _family_with_no_trustworthy_basis(monkeypatch)
    assert VALIDATED_HORIZONS_WEEKS == (1,)
    last_date = family._last_observed_date.date()
    two_weeks_out = last_date + _dt.timedelta(days=14)
    title = (
        f"Will U.S. commercial crude oil inventories for the week ending "
        f"{two_weeks_out.strftime('%b %-d, %Y')} be above 400 million barrels?"
    )
    market = _quote("KXEIACRUDEW-FAKE-T400", title, floor_strike=400.0)
    with pytest.raises(MarketNotScorable):
        family.estimate_probability(market)


def test_family_scores_a_real_shaped_market_against_a_flat_known_series(monkeypatch):
    family = _family_with_no_trustworthy_basis(monkeypatch)
    last_date = family._last_observed_date.date()
    target = last_date + _dt.timedelta(days=7)
    title = (
        f"Will U.S. commercial crude oil inventories for the week ending "
        f"{target.strftime('%b %-d, %Y')} be above 395 million barrels?"
    )
    # A flat series at 400.0 with zero calibration residual spread: the
    # point forecast is exactly 400.0, so P(> 395) should be exactly 1.0
    # and P(> 405) should be exactly 0.0 -- known, deterministic outcomes.
    above_below = _quote("KXEIACRUDEW-FAKE-T395", title, floor_strike=395.0)
    pe_below = family.estimate_probability(above_below)
    assert pe_below.our_probability == pytest.approx(1.0, abs=1e-6)
    assert pe_below.settlement_benchmark_matches_forecast is False  # no real settled values -> not trustworthy
    assert pe_below.forecast_horizon_days == 7
    assert pe_below.days_to_close == pytest.approx(7.0)

    title_above = title.replace("above 395", "above 405")
    above_above = _quote("KXEIACRUDEW-FAKE-T405", title_above, floor_strike=405.0)
    pe_above = family.estimate_probability(above_above)
    assert pe_above.our_probability == pytest.approx(0.0, abs=1e-6)


def test_family_applies_a_real_measured_trustworthy_basis_correction(monkeypatch):
    """When `fetch_real_settled_values` returns enough tight, consistent
    real-shaped measurements, the family should trust and apply the real
    basis correction to the adjusted strike."""
    series = _frozen_weekly_series()
    last_date = series.index.max().date()
    # 8 real-shaped settled weeks, each real week's Kalshi-recorded value
    # sitting a consistent, tight +1.0 million barrels above this
    # project's own (synthetic, but real-shaped) cached value of 400.0.
    settled = [(last_date - _dt.timedelta(days=7 * i), 401.0) for i in range(1, 9)]
    monkeypatch.setattr(cs_mod, "fetch_real_settled_values", lambda *a, **k: settled)

    family = CrudeStorageMarketFamily(series=series)
    assert family._basis_trustworthy is True
    assert family._basis_mean == pytest.approx(1.0, abs=1e-9)

    target = last_date + _dt.timedelta(days=7)
    title = (
        f"Will U.S. commercial crude oil inventories for the week ending "
        f"{target.strftime('%b %-d, %Y')} be above 400 million barrels?"
    )
    market = _quote("KXEIACRUDEW-FAKE-T400", title, floor_strike=400.0)
    pe = family.estimate_probability(market)
    # adjusted_strike = basis_adjusted_strike(400.0, 1.0) = 400.0 - 1.0 = 399.0
    assert pe.adjusted_strike == pytest.approx(399.0)
    assert pe.settlement_benchmark_matches_forecast is True


def test_family_real_live_kxeiacrudew_end_to_end():
    """Integration test: actually builds a real `CrudeStorageMarketFamily`
    against the real, live Kalshi `KXEIACRUDEW` series and the real, live
    EIA weekly storage series -- no mocking, same discipline as every
    other real live-API test in this suite. Scores every real currently-
    open market; doesn't assert a specific probability (that's real,
    live, and will keep changing), just that the real pipeline runs
    end-to-end without error and produces sane, in-range real
    probabilities."""
    from commodity_forecast.markets.base import run_family_survey

    family = CrudeStorageMarketFamily()
    surveyed, skipped = run_family_survey(family)

    assert len(surveyed) + len(skipped) > 0
    for s in surveyed:
        assert 0.0 <= s.estimate.our_probability <= 1.0
