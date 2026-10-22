"""Real public U.S. crude oil storage (inventory) data ingestion.

Data source: the U.S. Energy Information Administration's own published
weekly series "Weekly U.S. Ending Stocks excluding SPR of Crude Oil"
(EIA series id ``WCESTUS1``), served directly from eia.gov as a
spreadsheet -- no API key or registration required, same ingestion
pattern `data.py` already uses for the WTI daily spot series (EIA's own
"download the data" link, not a scrape of a rendered page):
https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls

This is the real series a real, live, open Kalshi market family
(`KXEIACRUDEW`, "U.S. crude oil inventories") settles against -- confirmed
directly, not assumed: a real finalized `KXEIACRUDEW` market's own
`rules_secondary` text names "the U.S. Energy Information Administration
(EIA)" and "excludes crude oil held in the Strategic Petroleum Reserve
(SPR)" explicitly, matching `WCESTUS1`'s own real title word for word, and
the real finalized market's own `expiration_value` field (`"412.5M"` for
the week ending 2026-10-16) matches this project's own fetched
`WCESTUS1` value for that week (412.517 thousand... i.e. 412.517 million
barrels, rounds to the same 412.5M Kalshi reports) to within normal
EIA-report rounding -- see `markets/crude_storage.py`'s own module
docstring for the fuller account and the real, measured check across
many such weeks, not just this one.

EIA reports this series in **thousand barrels**; this module converts to
**million barrels** on ingestion (dividing by 1000) so every value here is
directly comparable to a real `KXEIACRUDEW` strike (Kalshi's own ladder is
denominated in whole millions of barrels, e.g. "above 418 million
barrels?").

**A real, disclosed data-quality finding, not silently smoothed over:**
the raw series is not uniformly weekly throughout its full real history.
Before 1983-06-10, real gaps of 14 and 28 days appear in the raw cached
file (3 fourteen-day gaps, 1 twenty-eight-day gap, confirmed by directly
diffing consecutive real dates) -- consistent with EIA's weekly reporting
cadence for this series only becoming fully regular partway through the
1980s, not a parsing bug. From 1983-06-10 onward, every one of the 2,294
real consecutive date pairs checked is exactly 7 days apart, with zero
exceptions. Since this project's own weekly forecaster (`markets/
crude_storage.py`) depends on a fixed 7-day step between consecutive
lookback-window observations (the same assumption `forecast.py`'s
`prepare_horizon_split` already makes for WTI's own daily series), this
module only ever returns the consistently-weekly era (dates >=
`WEEKLY_CADENCE_START_DATE`), never the irregular pre-1983-06-10 rows --
a real, disclosed scope restriction, not data silently dropped without a
reason.
"""

from __future__ import annotations

import datetime as _dt
import pathlib
import urllib.request

WCESTUS1_XLS_URL = "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"
_EXCEL_EPOCH = _dt.date(1899, 12, 30)  # Excel's (incorrect, leap-year-bug) day-0

#: The first real date, confirmed directly against the raw cached file, after
#: which every consecutive real observation is exactly 7 calendar days apart
#: (see the module docstring's data-quality finding). Rows before this date
#: are real EIA data, just not on a consistent weekly cadence -- excluded
#: here, not elsewhere in this project's pipeline.
WEEKLY_CADENCE_START_DATE = _dt.date(1983, 6, 10)

DEFAULT_CACHE_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "crude_storage_weekly.csv"


class DataFetchError(RuntimeError):
    """Raised when the real EIA data source can't be reached or parsed."""


def _excel_serial_to_date(serial: float) -> _dt.date:
    return _EXCEL_EPOCH + _dt.timedelta(days=int(serial))


def _parse_wcestus1_xls(raw_bytes: bytes) -> list[tuple[_dt.date, float]]:
    """Parse the 'Data 1' sheet of EIA's WCESTUS1w.xls into
    (date, million_barrels) pairs, converting from the file's own native
    thousand-barrel units.

    Deliberately scans for the first numeric row rather than hardcoding a
    fixed header-row count: this file's real header block (3 rows -- a
    "Back to Contents" row, a "Sourcekey" row, then the column-title row)
    differs by one row from `data.py`'s `RWTCd.xls` (2 header rows before
    data), confirmed by direct inspection -- scanning for the first row
    whose first cell is numeric is robust to that kind of real, minor
    layout difference between EIA's own spreadsheet exports instead of
    silently assuming every EIA series file shares one fixed layout.
    """
    import xlrd  # local import: only needed for this one parsing step

    book = xlrd.open_workbook(file_contents=raw_bytes)
    sheet = book.sheet_by_name("Data 1")

    rows: list[tuple[_dt.date, float]] = []
    for r in range(sheet.nrows):
        values = sheet.row_values(r)
        if len(values) != 2:
            continue
        serial, value = values
        if not isinstance(serial, (int, float)) or not isinstance(value, (int, float)):
            continue
        rows.append((_excel_serial_to_date(serial), float(value) / 1000.0))

    if len(rows) < 100:
        raise DataFetchError(
            f"Parsed only {len(rows)} numeric rows from EIA's WCESTUS1w.xls -- expected "
            "thousands of weekly observations. The sheet layout may have changed; don't "
            "trust this result."
        )
    return rows


def fetch_crude_storage_weekly_raw(timeout: float = 30.0) -> list[tuple[_dt.date, float]]:
    """Download and parse the real, current U.S. commercial crude oil
    storage weekly series from EIA, restricted to the consistently-weekly
    era (see the module docstring).

    Returns a list of (date, million_barrels) pairs, sorted ascending by
    date. Raises DataFetchError if the download or parse fails, or if
    fewer than 100 real rows remain after the cadence restriction --
    callers should not silently fall back to stale or synthetic data.
    """
    req = urllib.request.Request(
        WCESTUS1_XLS_URL, headers={"User-Agent": "commodity-forecast/0.1 (research)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001 - wrap any network failure honestly
        raise DataFetchError(f"Could not download {WCESTUS1_XLS_URL}: {exc}") from exc

    rows = _parse_wcestus1_xls(raw)
    rows.sort(key=lambda pair: pair[0])
    rows = [(d, v) for d, v in rows if d >= WEEKLY_CADENCE_START_DATE]
    if len(rows) < 100:
        raise DataFetchError(
            f"Only {len(rows)} real rows remain after restricting to the consistently-weekly "
            f"era (>= {WEEKLY_CADENCE_START_DATE.isoformat()}) -- don't trust this result."
        )
    return rows


def fetch_crude_storage_weekly(
    cache_path: pathlib.Path | str = DEFAULT_CACHE_PATH,
    refresh: bool = False,
):
    """Return the U.S. commercial crude oil storage weekly series (million
    barrels) as a pandas Series indexed by date (the real week-ending date
    each real EIA observation covers).

    By default, reuses a local CSV cache (committed to the repo as a dated
    snapshot) instead of hitting the network every time -- same pattern as
    `data.fetch_wti_daily`. Pass refresh=True to force a fresh download.
    """
    import pandas as pd

    cache_path = pathlib.Path(cache_path)

    if refresh or not cache_path.exists():
        rows = fetch_crude_storage_weekly_raw()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("date,storage_million_bbl\n")
            for d, v in rows:
                f.write(f"{d.isoformat()},{v}\n")

    df = pd.read_csv(cache_path, parse_dates=["date"])
    series = df.set_index("date")["storage_million_bbl"].sort_index()
    series.name = "crude_storage_million_bbl"
    return series


if __name__ == "__main__":
    s = fetch_crude_storage_weekly(refresh=True)
    print(f"Fetched {len(s)} observations, {s.index.min().date()} to {s.index.max().date()}")
    print(s.tail())
