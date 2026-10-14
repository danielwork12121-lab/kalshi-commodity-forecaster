"""Real public commodity price data ingestion.

Data source: the U.S. Energy Information Administration's own published daily
spot-price series for WTI crude oil at Cushing, OK (EIA series id ``RWTC``),
served directly from eia.gov as a spreadsheet — no API key or registration
required, and no scraping of a rendered page: this is EIA's own "download the
data" link for the series, the same file a human would get from
https://www.eia.gov/dnav/pet/hist/RWTCD.htm.

(EIA also exposes the same series through its v2 JSON API at api.eia.gov, but
that requires a personal API key and the ungated shared "DEMO_KEY" hits a
rate limit almost immediately under any real use — confirmed directly while
building this module. The spreadsheet download needs no key and no account,
so it's the more robust ingestion path for an unattended, no-registration
pipeline. If Daniel ever registers his own EIA API key, the v2 API would add
other series (e.g. production, inventories) this spreadsheet alone doesn't
carry — worth revisiting then.)

Honest scope: this module ingests exactly one real series (WTI daily spot
price) to start. See the project README for why, and what's next.
"""

from __future__ import annotations

import datetime as _dt
import io
import pathlib
import urllib.request

RWTC_XLS_URL = "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"
_EXCEL_EPOCH = _dt.date(1899, 12, 30)  # Excel's (incorrect, leap-year-bug) day-0

DEFAULT_CACHE_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "wti_spot_daily.csv"


class DataFetchError(RuntimeError):
    """Raised when the real EIA data source can't be reached or parsed."""


def _excel_serial_to_date(serial: float) -> _dt.date:
    return _EXCEL_EPOCH + _dt.timedelta(days=int(serial))


def _parse_rwtc_xls(raw_bytes: bytes) -> list[tuple[_dt.date, float]]:
    """Parse the 'Data 1' sheet of EIA's RWTCd.xls into (date, price) pairs."""
    import xlrd  # local import: only needed for this one parsing step

    book = xlrd.open_workbook(file_contents=raw_bytes)
    sheet = book.sheet_by_name("Data 1")

    rows: list[tuple[_dt.date, float]] = []
    # Row 0 is a "Back to Contents" link, row 1 is the series title, row 2 is
    # the header ("Date", "Cushing, OK WTI Spot Price FOB ..."). Real data
    # starts at row 3.
    for r in range(3, sheet.nrows):
        serial, value = sheet.row_values(r)
        if serial == "" or value == "":
            continue
        rows.append((_excel_serial_to_date(serial), float(value)))

    if len(rows) < 100:
        raise DataFetchError(
            f"Parsed only {len(rows)} rows from EIA's RWTCd.xls — expected "
            "thousands of daily observations. The sheet layout may have "
            "changed; don't trust this result."
        )
    return rows


def fetch_wti_daily_raw(timeout: float = 30.0) -> list[tuple[_dt.date, float]]:
    """Download and parse the real, current WTI daily spot price series from EIA.

    Returns a list of (date, price_usd_per_barrel) pairs, sorted ascending by
    date. Raises DataFetchError if the download or parse fails — callers
    should not silently fall back to stale or synthetic data.
    """
    req = urllib.request.Request(RWTC_XLS_URL, headers={"User-Agent": "commodity-forecast/0.1 (research)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001 - we want to wrap *any* network failure
        raise DataFetchError(f"Could not download {RWTC_XLS_URL}: {exc}") from exc

    rows = _parse_rwtc_xls(raw)
    rows.sort(key=lambda pair: pair[0])
    return rows


def fetch_wti_daily(
    cache_path: pathlib.Path | str = DEFAULT_CACHE_PATH,
    refresh: bool = False,
):
    """Return the WTI daily spot price series as a pandas Series indexed by date.

    By default, reuses a local CSV cache (committed to the repo as a dated
    snapshot) instead of hitting the network every time. Pass refresh=True to
    force a fresh download — this is what should happen periodically (e.g.
    each time a run extends the forward track record) so the cache doesn't go
    stale; every refresh overwrites the cache with the real series as of that
    moment, it is never hand-edited.
    """
    import pandas as pd

    cache_path = pathlib.Path(cache_path)

    if refresh or not cache_path.exists():
        rows = fetch_wti_daily_raw()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("date,price_usd_per_bbl\n")
            for d, v in rows:
                f.write(f"{d.isoformat()},{v}\n")

    df = pd.read_csv(cache_path, parse_dates=["date"])
    series = df.set_index("date")["price_usd_per_bbl"].sort_index()
    series.name = "wti_spot_usd_per_bbl"
    return series


if __name__ == "__main__":
    s = fetch_wti_daily(refresh=True)
    print(f"Fetched {len(s)} observations, {s.index.min().date()} to {s.index.max().date()}")
    print(s.tail())
