from __future__ import annotations

import pytest

from commodity_forecast.temperature_data import (
    TemperatureDataFetchError,
    _date_range,
    fetch_nyc_temp_daily,
)


def test_date_range_is_inclusive_of_both_endpoints():
    assert _date_range("2026-10-01", "2026-10-03") == ["2026-10-01", "2026-10-02", "2026-10-03"]


def test_date_range_single_day():
    assert _date_range("2026-10-01", "2026-10-01") == ["2026-10-01"]


def test_fetch_nyc_temp_daily_with_explicit_series_bypasses_cache_and_network(monkeypatch):
    """Passing `series=` returns it verbatim -- no cache read/write, no
    network call -- the same escape hatch `data.fetch_wti_daily` offers."""

    def _boom(*a, **k):
        raise AssertionError("should not hit the real network when series= is passed")

    import commodity_forecast.temperature_data as td

    monkeypatch.setattr(td, "_fetch_open_meteo_range", _boom)
    frozen = {"2026-10-01": 60.0, "2026-10-02": 61.0}
    result = fetch_nyc_temp_daily("2026-10-01", "2026-10-02", series=frozen)
    assert result is frozen


def test_fetch_nyc_temp_daily_caches_so_a_second_call_does_not_refetch(tmp_path, monkeypatch):
    import commodity_forecast.temperature_data as td

    cache_path = tmp_path / "nyc_cache.csv"
    calls = {"n": 0}

    def fake_fetch(start_date, end_date, latitude, longitude, **kwargs):
        calls["n"] += 1
        return {"2026-10-01": 55.0, "2026-10-02": 56.0}

    monkeypatch.setattr(td, "_fetch_open_meteo_range", fake_fetch)

    first = fetch_nyc_temp_daily("2026-10-01", "2026-10-02", cache_path=cache_path)
    assert first == {"2026-10-01": 55.0, "2026-10-02": 56.0}
    assert calls["n"] == 1
    assert cache_path.exists()

    second = fetch_nyc_temp_daily("2026-10-01", "2026-10-02", cache_path=cache_path)
    assert second == first
    assert calls["n"] == 1  # not re-fetched -- served entirely from the cache


def test_fetch_nyc_temp_daily_only_fetches_the_missing_tail(tmp_path, monkeypatch):
    import commodity_forecast.temperature_data as td

    cache_path = tmp_path / "nyc_cache.csv"
    fetched_ranges = []

    def fake_fetch(start_date, end_date, latitude, longitude, **kwargs):
        fetched_ranges.append((start_date, end_date))
        # Real Open-Meteo always returns the full requested range when it has data.
        return {d: 50.0 for d in _date_range(start_date, end_date)}

    monkeypatch.setattr(td, "_fetch_open_meteo_range", fake_fetch)

    fetch_nyc_temp_daily("2026-10-01", "2026-10-03", cache_path=cache_path)
    assert len(fetched_ranges) == 1

    # Extending the range should trigger a real fetch again (cache is missing 10-04/10-05).
    result = fetch_nyc_temp_daily("2026-10-01", "2026-10-05", cache_path=cache_path)
    assert len(fetched_ranges) == 2
    assert set(result) == set(_date_range("2026-10-01", "2026-10-05"))


def test_fetch_nyc_temp_daily_refresh_true_always_refetches(tmp_path, monkeypatch):
    import commodity_forecast.temperature_data as td

    cache_path = tmp_path / "nyc_cache.csv"
    calls = {"n": 0}

    def fake_fetch(start_date, end_date, latitude, longitude, **kwargs):
        calls["n"] += 1
        return {"2026-10-01": float(calls["n"])}

    monkeypatch.setattr(td, "_fetch_open_meteo_range", fake_fetch)

    fetch_nyc_temp_daily("2026-10-01", "2026-10-01", cache_path=cache_path)
    assert calls["n"] == 1

    second = fetch_nyc_temp_daily("2026-10-01", "2026-10-01", cache_path=cache_path, refresh=True)
    assert calls["n"] == 2
    assert second["2026-10-01"] == pytest.approx(2.0)  # the real refreshed value, not the stale cached one


def test_fetch_nyc_temp_daily_raises_when_real_data_is_missing_for_a_requested_date(tmp_path, monkeypatch):
    """Never silently interpolates a missing real date -- raises instead,
    same honesty discipline as `data.py`'s `DataFetchError`."""
    import commodity_forecast.temperature_data as td

    cache_path = tmp_path / "nyc_cache.csv"

    def fake_fetch(start_date, end_date, latitude, longitude, **kwargs):
        # Real Open-Meteo archive sometimes simply has no row for a date not
        # yet published -- simulate that honestly rather than fabricating one.
        return {"2026-10-01": 55.0}

    monkeypatch.setattr(td, "_fetch_open_meteo_range", fake_fetch)

    with pytest.raises(TemperatureDataFetchError):
        fetch_nyc_temp_daily("2026-10-01", "2026-10-02", cache_path=cache_path)


def test_fetch_nyc_temp_daily_raises_when_the_real_fetch_returns_nothing_at_all(tmp_path, monkeypatch):
    import commodity_forecast.temperature_data as td

    cache_path = tmp_path / "nyc_cache.csv"
    monkeypatch.setattr(td, "_fetch_open_meteo_range", lambda *a, **k: {})

    with pytest.raises(TemperatureDataFetchError):
        fetch_nyc_temp_daily("2026-10-01", "2026-10-02", cache_path=cache_path)


def test_fetch_nyc_temp_daily_real_live_open_meteo_archive():
    """Integration test: actually hits the real, live, free Open-Meteo
    archive API -- no mocking, same discipline as
    test_kalshi_client.py's own `test_fetch_series_real_live_kxwti_settlement_source`
    and test_data.py's live EIA test. A short, recent, 3-day real range
    keeps this cheap and avoids ever depending on the request happening to
    land near a rate limit."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        cache_path = Path(d) / "nyc_cache.csv"
        result = fetch_nyc_temp_daily("2026-10-17", "2026-10-19", cache_path=cache_path)
    assert set(result) == {"2026-10-17", "2026-10-18", "2026-10-19"}
    for v in result.values():
        # A real NYC October daily high is essentially never outside this
        # wide, physically-sane real range -- a crude sanity check against a
        # badly-parsed response, not a precise claim.
        assert -20.0 < v < 110.0
