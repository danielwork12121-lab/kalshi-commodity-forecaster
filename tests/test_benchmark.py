import json
import pathlib

import pytest

from commodity_forecast.benchmark import (
    BasisMeasurement,
    SettlementBracket,
    basis_adjusted_strike,
    bracket_settlements,
    describe_basis_correction,
    measure_basis,
    recent_basis_estimate,
)
from commodity_forecast.kalshi_client import _parse_market

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load_real_quotes():
    """Loads a real, captured slice of Kalshi's own `KXWTI` market history
    (5 real settlement events, 2026-08-12 through 2026-10-14, captured
    2026-10-19 -- not hand-written) and parses it with the same,
    already-tested `_parse_market` used in production."""
    raw = json.load(open(FIXTURES / "kxwti_settlement_history_2026-10-19.json"))
    return [_parse_market(m) for m in raw["markets"]]


def test_bracket_settlements_recovers_the_real_tight_bracket_for_one_event():
    """2026-08-12's real captured ladder resolved "yes" up through $82.99
    and "no" from $83.49 -- the real settlement value is known (from
    Kalshi's own resolution) to sit in that $0.50 window; this checks the
    bracketing logic recovers exactly that, not an approximation."""
    quotes = _load_real_quotes()
    brackets = {b.event_ticker: b for b in bracket_settlements(quotes)}

    b = brackets["KXWTI-26AUG1214"]
    assert b.date == "2026-08-12"
    assert b.max_yes_strike == pytest.approx(82.99)
    assert b.min_no_strike == pytest.approx(83.49)
    assert b.mid == pytest.approx(83.24)
    assert b.width == pytest.approx(0.50)


def test_bracket_settlements_skips_a_real_one_sided_ladder():
    """2026-09-02's real captured ladder resolved every single strike the
    same way (all "no") -- the true settlement value could be anywhere
    below the ladder's lowest strike, so there is nothing real to bracket.
    This must be skipped, not guessed at."""
    quotes = _load_real_quotes()
    dates = {b.date for b in bracket_settlements(quotes)}
    assert "2026-09-02" not in dates


def test_bracket_settlements_recovers_all_four_real_bracketable_events():
    quotes = _load_real_quotes()
    brackets = bracket_settlements(quotes)
    dates = sorted(b.date for b in brackets)
    assert dates == ["2026-08-12", "2026-10-07", "2026-10-13", "2026-10-14"]
    # Every real bracket recovered from this fixture happens to be exactly
    # one real $0.50 strike-spacing wide -- KXWTI's own observed real
    # strike spacing throughout the history this project has seen.
    assert all(b.width == pytest.approx(0.50) for b in brackets)


def test_measure_basis_pairs_real_brackets_with_real_spot_and_skips_unmatched_dates():
    quotes = _load_real_quotes()
    brackets = bracket_settlements(quotes)
    # Real EIA spot values for 3 of the 4 bracketable dates (as this
    # project's own cached series reports them) -- 2026-10-13 deliberately
    # left out of this spot map to confirm the unmatched-date skip works.
    spot_by_date = {"2026-08-12": 84.97, "2026-10-07": 95.02, "2026-10-14": 101.91}

    measurements = measure_basis(brackets, spot_by_date)
    assert {m.date for m in measurements} == {"2026-08-12", "2026-10-07", "2026-10-14"}

    by_date = {m.date: m for m in measurements}
    # Real, honestly measured finding: the basis is much larger in October
    # than in August -- not a stable constant across this history.
    assert by_date["2026-08-12"].basis == pytest.approx(83.24 - 84.97)
    assert by_date["2026-10-07"].basis == pytest.approx(88.24 - 95.02)
    assert by_date["2026-10-14"].basis == pytest.approx(95.24 - 101.91)
    assert by_date["2026-08-12"].basis > by_date["2026-10-14"].basis + 3


def test_recent_basis_estimate_uses_only_the_window_not_all_history():
    """A long-history-average basis would blend August's real ~-$1.7 basis
    with October's real ~-$6.7 basis into a misleading number -- this
    checks `recent_basis_estimate` with a small window actually ignores
    the older, stale August measurement rather than averaging it in."""
    measurements = [
        BasisMeasurement(date="2026-08-12", futures_mid=83.24, futures_bracket_width=0.5, spot=84.97, basis=-1.73),
        BasisMeasurement(date="2026-10-07", futures_mid=88.24, futures_bracket_width=0.5, spot=95.02, basis=-6.78),
        BasisMeasurement(date="2026-10-13", futures_mid=94.24, futures_bracket_width=0.5, spot=100.86, basis=-6.62),
        BasisMeasurement(date="2026-10-14", futures_mid=95.24, futures_bracket_width=0.5, spot=101.91, basis=-6.67),
    ]

    mean, stdev, n = recent_basis_estimate(measurements, window=3)
    assert n == 3
    # Only the 3 most recent (October) measurements, August excluded.
    assert mean == pytest.approx((-6.78 + -6.62 + -6.67) / 3)
    assert stdev < 0.1  # the 3 October values are real and tightly clustered
    assert mean < -1.73 + 3  # nowhere near August's stale, much-smaller-magnitude basis


def test_recent_basis_estimate_returns_none_for_no_measurements():
    assert recent_basis_estimate([], window=10) is None


def test_recent_basis_estimate_handles_a_single_measurement():
    measurements = [
        BasisMeasurement(date="2026-10-14", futures_mid=95.24, futures_bracket_width=0.5, spot=101.91, basis=-6.67)
    ]
    mean, stdev, n = recent_basis_estimate(measurements, window=10)
    assert n == 1
    assert mean == pytest.approx(-6.67)
    assert stdev == 0.0


def test_basis_adjusted_strike_converts_futures_strike_to_spot_terms():
    # real October basis ~-6.67: a market asking "futures > $95.49?" is
    # equivalent, in this project's own spot terms, to "spot > $102.16?"
    assert basis_adjusted_strike(95.49, basis_mean=-6.67) == pytest.approx(102.16)


def test_describe_basis_correction_trusts_a_tight_well_supported_estimate():
    trustworthy, note = describe_basis_correction(mean=-6.70, stdev=0.15, n=10)
    assert trustworthy is True
    assert "-6.70" in note
    assert "0.15" in note
    assert "n=10" in note


def test_describe_basis_correction_rejects_too_few_points():
    trustworthy, note = describe_basis_correction(mean=-6.70, stdev=0.15, n=2, min_n_for_trust=5)
    assert trustworthy is False
    assert "NOT trusted" in note


def test_describe_basis_correction_rejects_too_much_spread():
    trustworthy, note = describe_basis_correction(mean=-4.0, stdev=2.5, n=10, max_stdev_for_trust=1.0)
    assert trustworthy is False
    assert "NOT trusted" in note


def test_settlement_bracket_is_a_real_frozen_value_object():
    b = SettlementBracket(event_ticker="KXWTI-26OCT1414", date="2026-10-14", max_yes_strike=94.99, min_no_strike=95.49)
    assert b.mid == pytest.approx(95.24)
    assert b.width == pytest.approx(0.50)
    with pytest.raises(AttributeError):
        b.date = "2026-10-15"  # frozen -- a bracket is a historical fact, not mutable state
