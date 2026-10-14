import datetime as dt

import pytest

from commodity_forecast.data import _excel_serial_to_date, fetch_wti_daily, fetch_wti_daily_raw


def test_excel_serial_to_date_matches_known_reference():
    # Excel serial 31414 == 1986-01-02, the real EIA RWTC series' start date.
    assert _excel_serial_to_date(31414) == dt.date(1986, 1, 2)
    # Excel's own epoch bug: serial 1 == 1899-12-31, not 1900-01-01, because
    # Excel incorrectly treats 1900 as a leap year. Verified against this
    # quirk, not just against a "clean" reference date, so a subtle off-by-
    # one in the date math wouldn't slip through.
    assert _excel_serial_to_date(1) == dt.date(1899, 12, 31)


def test_fetch_wti_daily_raw_returns_real_recent_data():
    """Integration test: actually hits the real, live EIA endpoint -- no
    mocking. Consistent with this project's and conformal-forecast's own
    testing discipline (exercise the real thing, don't just assert against
    a mock); this means the test needs network access to eia.gov and will
    fail, not silently pass, if that's ever unreachable -- the correct,
    honest behavior for a project whose whole point is real public data.
    """
    rows = fetch_wti_daily_raw()
    assert len(rows) > 9000  # the real series has run daily since Jan 1986
    dates = [d for d, _ in rows]
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)  # no duplicate dates
    # Spot-check a known real value from this series, independently
    # cross-checked against EIA's own v2 JSON API while building this
    # module (same series, same number, different EIA endpoint).
    by_date = dict(rows)
    assert by_date[dt.date(2026, 10, 6)] == pytest.approx(96.24)


def test_fetch_wti_daily_uses_cache_without_hitting_network(tmp_path, monkeypatch):
    cache_path = tmp_path / "wti_spot_daily.csv"
    cache_path.write_text("date,price_usd_per_bbl\n2020-01-02,61.18\n2020-01-03,63.05\n")

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError(
            "fetch_wti_daily_raw should not be called when a cache file "
            "already exists and refresh=False"
        )

    monkeypatch.setattr("commodity_forecast.data.fetch_wti_daily_raw", _should_not_be_called)

    series = fetch_wti_daily(cache_path=cache_path, refresh=False)
    assert len(series) == 2
    assert float(series.iloc[0]) == pytest.approx(61.18)
    assert float(series.iloc[1]) == pytest.approx(63.05)
