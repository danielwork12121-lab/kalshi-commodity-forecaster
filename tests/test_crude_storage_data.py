from __future__ import annotations

import datetime as dt

import pytest

from commodity_forecast.crude_storage_data import (
    WEEKLY_CADENCE_START_DATE,
    DataFetchError,
    _excel_serial_to_date,
    fetch_crude_storage_weekly,
    fetch_crude_storage_weekly_raw,
)


def test_excel_serial_to_date_matches_known_reference():
    # Excel serial 30183 == 1982-08-20, the real EIA WCESTUS1 series' own
    # first cached row (confirmed directly against the raw downloaded file
    # while building this module).
    assert _excel_serial_to_date(30183) == dt.date(1982, 8, 20)


def test_fetch_crude_storage_weekly_raw_returns_real_recent_data():
    """Integration test: actually hits the real, live EIA endpoint -- no
    mocking, same discipline as `test_data.py`'s own live EIA test and
    `test_temperature_data.py`'s own live Open-Meteo test."""
    rows = fetch_crude_storage_weekly_raw()
    assert len(rows) > 2000  # the real consistently-weekly era alone is 40+ years
    dates = [d for d, _ in rows]
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)  # no duplicate dates
    assert all(d >= WEEKLY_CADENCE_START_DATE for d in dates)
    # Every real consecutive pair in the returned (restricted) series must be
    # exactly 7 real days apart -- the whole point of the cadence restriction.
    for a, b in zip(dates, dates[1:]):
        assert (b - a).days == 7
    # Spot-check a known real value (million barrels, converted from EIA's
    # own native thousand-barrel units) independently confirmed against a
    # direct raw-file parse while building this module.
    by_date = dict(rows)
    assert by_date[dt.date(2026, 10, 16)] == pytest.approx(412.517)


def test_fetch_crude_storage_weekly_uses_cache_without_hitting_network(tmp_path, monkeypatch):
    cache_path = tmp_path / "crude_storage_weekly.csv"
    cache_path.write_text("date,storage_million_bbl\n2020-01-03,440.0\n2020-01-10,438.5\n")

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError(
            "fetch_crude_storage_weekly_raw should not be called when a cache file already "
            "exists and refresh=False"
        )

    monkeypatch.setattr(
        "commodity_forecast.crude_storage_data.fetch_crude_storage_weekly_raw", _should_not_be_called
    )

    series = fetch_crude_storage_weekly(cache_path=cache_path, refresh=False)
    assert len(series) == 2
    assert float(series.iloc[0]) == pytest.approx(440.0)
    assert float(series.iloc[1]) == pytest.approx(438.5)


def test_fetch_crude_storage_weekly_refresh_true_refetches(tmp_path, monkeypatch):
    cache_path = tmp_path / "crude_storage_weekly.csv"
    cache_path.write_text("date,storage_million_bbl\n2020-01-03,440.0\n")
    calls = {"n": 0}

    def fake_raw():
        calls["n"] += 1
        return [(dt.date(2020, 1, 3), 999.0)]

    monkeypatch.setattr("commodity_forecast.crude_storage_data.fetch_crude_storage_weekly_raw", fake_raw)

    series = fetch_crude_storage_weekly(cache_path=cache_path, refresh=True)
    assert calls["n"] == 1
    assert float(series.iloc[0]) == pytest.approx(999.0)  # the real refreshed value, not the stale cached one


def test_fetch_crude_storage_weekly_raw_rejects_too_few_rows_after_cadence_restriction(monkeypatch):
    """Never silently proceeds with too little real data after restricting
    to the consistently-weekly era -- raises instead, same honesty
    discipline as `data.py`'s own row-count sanity check. Monkeypatches the
    parse step (not the network) so this stays a fast, offline unit test of
    the guard itself, distinct from the live integration test above."""
    import commodity_forecast.crude_storage_data as csd

    def fake_parse(raw_bytes):
        # Only 3 real rows, all after WEEKLY_CADENCE_START_DATE -- too few
        # to trust, same bar the real parser enforces.
        return [
            (dt.date(2026, 10, 2), 424.134),
            (dt.date(2026, 10, 9), 415.577),
            (dt.date(2026, 10, 16), 412.517),
        ]

    monkeypatch.setattr(csd, "_parse_wcestus1_xls", fake_parse)

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"irrelevant -- _parse_wcestus1_xls is monkeypatched above"

    monkeypatch.setattr(csd.urllib.request, "urlopen", lambda *a, **k: _FakeResp())

    with pytest.raises(DataFetchError):
        fetch_crude_storage_weekly_raw()


def test_data_fetch_error_is_a_runtime_error():
    assert issubclass(DataFetchError, RuntimeError)
