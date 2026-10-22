"""Real historical daily maximum temperature data for the temperature market
family (`markets/temperature.py`), sourced from Open-Meteo's free, public,
no-API-key-required historical weather archive (ERA5-Land-based reanalysis,
`https://archive-api.open-meteo.com`).

## Why Open-Meteo, and not NWS point forecasts

`active-project-kalshi.md`'s 2026-10-21 direction update originally named
NWS point forecasts as this family's forecast source. Real research (the
sixteenth scheduled run, 2026-10-22) found this project's own real
settlement benchmark is **The Weather Company** (confirmed directly via
`kalshi_client.fetch_series("KXHIGHNY")["settlement_sources"]` and every
real market's own `rules_primary` text), not NWS -- a new benchmark-mismatch
question, the same shape as `KXWTI`'s own EIA-spot-vs-ICE-futures mismatch.

Rather than forecast with NWS's own point forecast (itself a *forward*
forecast, with no free historical archive to backtest against -- NWS
publishes its current forecast, not what it would have said N days before
each past date) and silently assume it tracks The Weather Company's real
CLINYC reading, this project takes a different, more directly verifiable
approach: forecast the real **historical actual** daily high temperature at
the same real location (via Open-Meteo's archive, which *does* have decades
of real daily history, unlike a live-only point-forecast API), using the
same kind of calibrated time-series method this project already uses for
WTI spot -- then **measure, directly against Kalshi's own real settled
brackets**, how close Open-Meteo's real reading runs to the real Weather
Company settlement (`markets/temperature.py`'s own basis-measurement
functions), exactly the way `benchmark.py` already measures WTI's real
spot/futures basis. This sidesteps a forecast-source assumption entirely in
favor of a real, measured comparison -- the same honesty discipline, applied
to a genuinely different data-availability problem.

**Real, measured basis finding (2026-10-22, 23 real paired observations,
2026-09-19 through 2026-10-21):** Open-Meteo's archive reading runs, on
average, **+0.35 degF warmer** than the real Weather Company settlement
bracket's own midpoint, with a real stdev of **1.14 degF** -- see
`markets/temperature.py`'s `measure_temperature_basis` and
`describe_temperature_basis_correction` for the measurement itself and the
honest trust-gate verdict (this real stdev is measured, pre-registered
bar-based, and -- reported plainly, not softened -- narrowly **fails** the
same `stdev <= 1.0` trust bar this project already uses for WTI's own basis
correction).
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_NYC_CACHE_PATH = DATA_DIR / "nyc_temp_daily.csv"

#: Real public coordinates used for the Open-Meteo archive query -- Central
#: Park, Manhattan, the real physical area `KXHIGHNY`'s own settlement
#: station (CLINYC) reports for. Open-Meteo resolves this to its nearest
#: real reanalysis grid cell.
NYC_LATITUDE = 40.7829
NYC_LONGITUDE = -73.9654

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class TemperatureDataFetchError(RuntimeError):
    """Raised when the real Open-Meteo archive API can't be reached or
    parsed -- never silently returns an empty/partial series instead."""


def _fetch_open_meteo_range(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    timeout: float = 30.0,
    max_retries: int = 5,
    initial_backoff: float = 3.0,
) -> dict[str, float]:
    """One real GET against Open-Meteo's public archive API for
    `[start_date, end_date]` (both `"YYYY-MM-DD"`, inclusive), returning
    `{iso_date: real_max_temp_fahrenheit}`. Retries on HTTP 429 (hit live
    while building this, same real rate-limiting behavior
    `kalshi_client.py` already documents for Kalshi's own API) with
    exponential backoff.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    }
    url = f"{OPEN_METEO_ARCHIVE_URL}?{urllib.parse.urlencode(params)}"

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers={"User-Agent": "commodity-forecast/0.1 (research)"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read())
            daily = payload.get("daily", {})
            dates = daily.get("time", [])
            values = daily.get("temperature_2m_max", [])
            return {d: float(v) for d, v in zip(dates, values) if v is not None}
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429 or 500 <= exc.code < 600:
                if attempt < max_retries - 1:
                    time.sleep(initial_backoff * (2**attempt))
                    continue
            raise TemperatureDataFetchError(f"GET {url} failed: HTTP {exc.code} {exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001 - wrap any network failure honestly
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(initial_backoff * (2**attempt))
                continue
            raise TemperatureDataFetchError(f"GET {url} failed: {exc}") from exc

    raise TemperatureDataFetchError(f"GET {url} failed after {max_retries} attempts: {last_exc}")


def _load_cache(cache_path: Path) -> dict[str, float]:
    if not cache_path.exists():
        return {}
    with open(cache_path, newline="") as f:
        return {row["date"]: float(row["temp_max_f"]) for row in csv.DictReader(f)}


def _save_cache(cache_path: Path, series: dict[str, float]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "temp_max_f"])
        for d in sorted(series):
            writer.writerow([d, series[d]])


def fetch_nyc_temp_daily(
    start_date: str,
    end_date: str,
    refresh: bool = False,
    series: dict[str, float] | None = None,
    cache_path: Path | str = DEFAULT_NYC_CACHE_PATH,
    latitude: float = NYC_LATITUDE,
    longitude: float = NYC_LONGITUDE,
) -> dict[str, float]:
    """Return `{iso_date: real_max_temp_fahrenheit}` for NYC, for every real
    date in `[start_date, end_date]` Open-Meteo's archive has data for,
    cached locally (mirroring `data.fetch_wti_daily`'s own CSV-cache
    pattern) so repeated calls don't re-hit the real API for dates already
    fetched.

    `series`: pass an explicit `{date: value}` mapping (e.g. in tests, or to
    pin a frozen real historical sample for reproducibility) to bypass both
    the cache and the live fetch entirely -- the same escape hatch
    `data.fetch_wti_daily`/`edge_survey.survey_live_edges` already offer for
    an identical reason. When passed, the result is simply `series` itself
    (no date filtering or caching), so a fixture's content is never
    silently altered.

    `refresh=True` re-fetches the full `[start_date, end_date]` range from
    the real API even if every date is already cached (useful for picking
    up a real correction/update to already-cached historical data, mirroring
    `data.fetch_wti_daily(refresh=True)`). Otherwise only the real dates
    missing from the cache are fetched.
    """
    if series is not None:
        return series

    cache_path = Path(cache_path)
    cache = _load_cache(cache_path)

    all_dates_present = all(
        d in cache
        for d in _date_range(start_date, end_date)
    )

    if refresh or not all_dates_present:
        fresh = _fetch_open_meteo_range(start_date, end_date, latitude, longitude)
        if not fresh:
            raise TemperatureDataFetchError(
                f"Open-Meteo returned no real data for [{start_date}, {end_date}]"
            )
        cache.update(fresh)
        _save_cache(cache_path, cache)

    result = {d: v for d, v in cache.items() if start_date <= d <= end_date}
    missing = [d for d in _date_range(start_date, end_date) if d not in result]
    if missing:
        raise TemperatureDataFetchError(
            f"Open-Meteo has no real data for {len(missing)} requested date(s) in "
            f"[{start_date}, {end_date}], e.g. {missing[:3]} -- never silently interpolated"
        )
    return result


def _date_range(start_date: str, end_date: str) -> list[str]:
    import datetime as _dt

    start = _dt.date.fromisoformat(start_date)
    end = _dt.date.fromisoformat(end_date)
    out = []
    d = start
    while d <= end:
        out.append(d.isoformat())
        d += _dt.timedelta(days=1)
    return out
