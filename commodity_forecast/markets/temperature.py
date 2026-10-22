"""The real temperature market family: `KXHIGH<STATION>` daily high-
temperature markets (starting with `KXHIGHNY`, NYC), scored via a real
**climatological empirical distribution** built from Open-Meteo's real
historical daily-max-temperature archive (`temperature_data.py`), basis-
corrected against a real, measured gap to this family's own real settlement
benchmark, The Weather Company (not NWS -- see `temperature_data.py`'s
module docstring for the full research account of why this project doesn't
use a live NWS point forecast here).

## Why climatology, not an autoregressive point forecast

`markets/wti.py` forecasts WTI spot as an autoregressive time series: the
next value is correlated with recent ones, so a lookback window plus a
conformal-calibrated point model is the right tool. Daily high temperature
in one city is a fundamentally different kind of series -- it is dominated
by a strong, real seasonal cycle (a September reading tells you almost
nothing about an October reading a month later), not by short-term
autocorrelation alone. A naive reuse of `forecast.fit_calibrated_forecaster_horizon`
(a short lookback window of the *immediately preceding* days) would be
answering a worse-specified question than the one this market actually
asks. Instead, this module uses the standard meteorological baseline for
exactly this situation -- **climatology**: the real, empirical distribution
of what daily high temperature has actually been on/near this same
calendar date in past real years. This is a real, honest, pre-registered
baseline method (not invented for this project to force a clean story),
and it is directly backtestable -- see `backtest_climatology_coverage`
below -- the same "measure the calibration, don't just assert it"
discipline as every other probability method in this project.

## Real strike shapes this family supports

Confirmed live against the real, open `KXHIGHNY` ladder (2026-10-22):
`strike_type` is `"greater"` (`floor_strike` only -- "is the high above
X?"), `"less"` (`cap_strike` only -- "is the high below X?"), or
`"between"` (`floor_strike` AND `cap_strike`, always one degree apart in
every real market observed -- "is the high X or X+1?"). All three are
handled here (unlike `WTIMarketFamily`, which only ever needed
`"greater"` for its own real ladder).

**A real, disclosed modeling choice: the +-0.5 degF continuity
correction.** Every real `rules_primary` text checked states a strict
integer comparison (e.g. *"...is greater than 64 degF..."*), and The
Weather Company's own real reported reading is itself a whole-degree
value (confirmed: real `"between"` brackets observed are always exactly
one degree wide, e.g. floor=63/cap=64 for "between 63-64 degF"). Treating
our own continuous climatological distribution as an unbiased estimate of
that real whole-degree reading, the standard way to convert an integer
threshold into a query against a continuous distribution is a +-0.5 degF
continuity correction: "greater than 64" (the real whole-degree reading is
>= 65) becomes `P(X > 64.5)`; "less than 57" (reading <= 56) becomes
`P(X < 56.5)`; "between 63-64" (reading in {63, 64}) becomes
`P(62.5 < X < 64.5)`. Stated plainly here rather than silently assumed.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path

from commodity_forecast.benchmark import BasisMeasurement, basis_adjusted_strike, recent_basis_estimate
from commodity_forecast.edge import days_to_close_from_last_observation
from commodity_forecast.kalshi_client import KalshiMarketQuote, fetch_markets
from commodity_forecast.markets.base import MarketFamily, MarketNotScorable, ProbabilityEstimate
from commodity_forecast.temperature_data import (
    EASTERN_TIMEZONE,
    NYC_LATITUDE,
    NYC_LONGITUDE,
    default_cache_path_for_station,
    fetch_nyc_temp_daily,
)

#: Re-exported here so a caller building a `TemperatureMarketFamily` for
#: the second station doesn't need to separately import from
#: `temperature_data` -- `MarketFamily` construction sites naturally
#: import from this module already.
from commodity_forecast.temperature_data import MIAMI_LATITUDE, MIAMI_LONGITUDE  # noqa: F401

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_TICKER_DATE_RE = re.compile(r"^[A-Z]+-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})-")

#: Real, human-readable station labels for `describe_temperature_basis_correction`'s
#: own note text -- keyed by series ticker so each station's own real
#: basis note says which real station it actually measured, never a
#: hardcoded "NYC" regardless of which station was built (the real bug
#: fixed alongside this run's Miami addition).
_STATION_LABELS = {
    "KXHIGHNY": "NYC",
    "KXHIGHMIA": "Miami",
}


def parse_ticker_settlement_date(ticker: str) -> str | None:
    """The real settlement date (`"YYYY-MM-DD"`) encoded in a real
    `KXHIGH<STATION>` market ticker (e.g. `"KXHIGHNY-26OCT23-T64"` ->
    `"2026-10-23"`), or `None` if the ticker doesn't match the expected
    real format. Works for every real strike-type suffix (`-T64`, `-T57`,
    `-B63.5`) since it only parses the shared date segment, unlike
    `benchmark._event_ticker` (which assumes a `"-T"` suffix specifically
    and would mis-split a `"-B..."` ticker).
    """
    m = _TICKER_DATE_RE.match(ticker)
    if not m:
        return None
    month = _MONTHS.get(m.group("mon"))
    if month is None:
        return None
    return f"20{m.group('yy')}-{month:02d}-{int(m.group('dd')):02d}"


@dataclass(frozen=True)
class TemperatureBracket:
    """One real settled `"between"` market's own real bracket: Kalshi's
    public API directly reports which single bracket of a real
    `KXHIGH<STATION>` ladder actually resolved "yes" -- no reconstruction
    from a ladder of "greater than" strikes needed (unlike
    `benchmark.bracket_settlements`'s method for `KXWTI`, which only ever
    has "greater" strikes to work with). This is a more direct, exact real
    measurement of the true settlement value than WTI's own bracket
    recovery, not an approximation of it.
    """

    date: str
    floor_strike: float
    cap_strike: float

    @property
    def mid(self) -> float:
        return (self.floor_strike + self.cap_strike) / 2.0


def fetch_real_settled_brackets(series_ticker: str, max_pages: int = 10, **kwargs) -> list[TemperatureBracket]:
    """Real, settled `"between"` markets in `series_ticker` that resolved
    `"yes"` -- each one directly IS a real measured settlement bracket,
    one per real past settlement date. `max_pages` bounds how far back to
    page (Kalshi's `status="settled"` listing is most-recent-first)."""
    quotes = fetch_markets(series_ticker, status="settled", max_pages=max_pages, **kwargs)
    brackets: list[TemperatureBracket] = []
    for q in quotes:
        if q.strike_type != "between" or q.result != "yes":
            continue
        if q.floor_strike is None or q.cap_strike is None:
            continue
        date = parse_ticker_settlement_date(q.ticker)
        if date is None:
            continue
        brackets.append(TemperatureBracket(date=date, floor_strike=q.floor_strike, cap_strike=q.cap_strike))
    return sorted(brackets, key=lambda b: b.date)


def measure_temperature_basis(
    brackets: list[TemperatureBracket], om_series: dict[str, float]
) -> list[BasisMeasurement]:
    """Pair each real settled bracket with this project's own cached
    Open-Meteo reading for the same real calendar date, and compute the
    real, measured basis for that date: `basis = bracket_mid -
    open_meteo_value` -- the same sign convention `benchmark.measure_basis`
    uses (`basis = real_settlement_benchmark - this_project's_own_data_source`),
    so `benchmark.basis_adjusted_strike`/`recent_basis_estimate` are
    directly reusable without modification (they operate generically on
    `BasisMeasurement.basis`/`.date`, with no WTI-specific logic in the
    function bodies themselves -- only their own docstrings mention WTI's
    units, which doesn't affect reuse here). A settlement date with no
    matching Open-Meteo observation is silently skipped (never
    interpolated), same discipline as `benchmark.measure_basis`.

    Returns `BasisMeasurement` records with `futures_mid`/`spot` fields
    repurposed to mean "real Weather Company bracket midpoint" /
    "real Open-Meteo reading" respectively -- the field *names* are
    WTI-flavored (inherited from the shared dataclass), but every value
    stored in them is real temperature data in degF, not WTI prices.
    """
    out: list[BasisMeasurement] = []
    for b in brackets:
        om_value = om_series.get(b.date)
        if om_value is None:
            continue
        out.append(
            BasisMeasurement(
                date=b.date,
                futures_mid=b.mid,
                futures_bracket_width=b.cap_strike - b.floor_strike,
                spot=om_value,
                basis=b.mid - om_value,
            )
        )
    return out


def describe_temperature_basis_correction(
    mean: float,
    stdev: float,
    n: int,
    max_stdev_for_trust: float = 1.0,
    min_n_for_trust: int = 5,
    station_label: str = "NYC",
) -> tuple[bool, str]:
    """The temperature-family equivalent of `benchmark.describe_basis_correction`,
    with real, correct units/wording (degF, Open-Meteo vs. The Weather
    Company) rather than reusing that function's own WTI-specific note
    text verbatim (which would misreport units -- "USD/bbl" -- and the
    wrong real data sources). Same trust-bar defaults as WTI's own
    (`stdev <= 1.0`, `n >= 5`) for a consistent, not-loosened-to-force-a-
    pass real bar across every family in this project.

    `station_label` defaults to `"NYC"` for backward compatibility (every
    caller before the 2026-10-22 Miami addition implicitly meant NYC and
    never passed one) -- the real bug this fixes: before `station_label`
    existed, this note's text always said "NYC" verbatim even when called
    for a different real station (Miami), which would have silently
    mislabeled Miami's own real measured basis note. `TemperatureMarketFamily.__init__`
    now passes the real station name it was actually built for.
    """
    trustworthy = n >= min_n_for_trust and stdev <= max_stdev_for_trust
    note = (
        f"temperature settlement-benchmark basis measured from {n} real paired observations "
        f"(real settled Weather Company bracket midpoint vs. this project's own Open-Meteo "
        f"archive reading, same real {station_label} location): mean {mean:+.2f} degF, stdev {stdev:.2f} degF "
        f"-- {'tight and well-supported enough to trust' if trustworthy else 'NOT trusted (too much spread for the same pre-registered bar WTI uses)'}"
    )
    return trustworthy, note


def recent_temperature_basis_estimate(
    measurements: list[BasisMeasurement], window: int = 10
) -> tuple[float, float, int] | None:
    """Thin, explicitly-named wrapper around `benchmark.recent_basis_estimate`
    (generic real mean/stdev-of-the-most-recent-`window` logic, reused
    unchanged) -- kept as its own function here so a reader of this module
    doesn't have to go look up what "recent_basis_estimate" means for a
    family it wasn't originally written for."""
    return recent_basis_estimate(measurements, window=window)


def _parse_iso_date(d: str) -> _dt.date:
    return _dt.date.fromisoformat(d)


def _day_of_year_distance(a: _dt.date, b: _dt.date) -> int:
    """Real circular distance in days between two dates' real calendar
    position in the year (month+day only, year-independent) -- e.g. Dec 30
    and Jan 2 are 3 real days apart, not ~363, so a climatology window
    correctly spans a New Year's boundary."""
    doy_a = a.timetuple().tm_yday
    doy_b = b.timetuple().tm_yday
    diff = abs(doy_a - doy_b)
    return min(diff, 365 - diff)


def climatological_samples(
    series: dict[str, float],
    target_date: str,
    window_days: int = 7,
    exclude_year: int | None = None,
) -> list[float]:
    """Real historical daily-max-temperature observations from `series`
    whose real calendar date falls within `window_days` of `target_date`'s
    own calendar position, across every real year `series` has data for
    (optionally excluding one real year -- used by
    `backtest_climatology_coverage`'s leave-one-year-out real validation).
    This is the real empirical climatological distribution -- no
    parametric (e.g. Gaussian) assumption imposed on it.
    """
    target = _parse_iso_date(target_date)
    out = []
    for d, v in series.items():
        dt = _parse_iso_date(d)
        if exclude_year is not None and dt.year == exclude_year:
            continue
        if _day_of_year_distance(dt, target) <= window_days:
            out.append(v)
    return out


def _empirical_quantile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("no samples to compute a quantile from")
    idx = q * (len(ordered) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def empirical_exceedance_probability(samples: list[float], threshold: float) -> float:
    """`P(X > threshold)` as the real fraction of `samples` exceeding it."""
    if not samples:
        raise ValueError("no samples to compute an exceedance probability from")
    return sum(1 for s in samples if s > threshold) / len(samples)


@dataclass(frozen=True)
class TemperatureClimatologyBacktestResult:
    """One real, honest leave-one-year-out calibration check: across every
    real (year, calendar-date) combination with enough other-year history,
    did the real actual value fall inside the empirical
    `[alpha/2, 1 - alpha/2]` quantile interval built from every *other*
    real year's data, at the rate a well-calibrated `1 - alpha` interval
    should?
    """

    alpha: float
    window_days: int
    n_evaluations: int
    empirical_coverage: float  # real measured fraction inside the interval
    target_coverage: float  # 1 - alpha


def backtest_climatology_coverage(
    series: dict[str, float],
    window_days: int = 7,
    alphas: tuple[float, ...] = (0.2, 0.1),
    min_samples_required: int = 20,
    eval_every: int = 7,
) -> list[TemperatureClimatologyBacktestResult]:
    """Real leave-one-year-out backtest of the climatological method's own
    calibration -- never asserted, measured. For a real subsample of real
    dates in `series` (every `eval_every`th real date, to keep the real
    computation cheap without evaluating only one part of the real
    calendar), rebuild the real empirical climatological distribution
    using every OTHER real year's data within `window_days`, and check
    whether that date's own real actual value falls inside the resulting
    `[alpha/2, 1-alpha/2]` empirical quantile interval. A real date whose
    other-year sample has fewer than `min_samples_required` points is
    skipped (too little real data to trust a quantile estimate from, not
    silently included anyway).
    """
    dates = sorted(series)
    results = []
    for alpha in alphas:
        n_eval = 0
        n_covered = 0
        for i, d in enumerate(dates):
            if i % eval_every != 0:
                continue
            dt = _parse_iso_date(d)
            samples = climatological_samples(series, d, window_days=window_days, exclude_year=dt.year)
            if len(samples) < min_samples_required:
                continue
            lower = _empirical_quantile(samples, alpha / 2)
            upper = _empirical_quantile(samples, 1 - alpha / 2)
            actual = series[d]
            n_eval += 1
            if lower <= actual <= upper:
                n_covered += 1
        if n_eval > 0:
            results.append(
                TemperatureClimatologyBacktestResult(
                    alpha=alpha,
                    window_days=window_days,
                    n_evaluations=n_eval,
                    empirical_coverage=n_covered / n_eval,
                    target_coverage=1 - alpha,
                )
            )
    return results


def format_climatology_backtest(results: list[TemperatureClimatologyBacktestResult]) -> str:
    lines = [f"{'target coverage':>16} {'measured coverage':>18} {'n':>6} {'window (days)':>14}"]
    for r in results:
        lines.append(
            f"{r.target_coverage:>16.0%} {r.empirical_coverage:>18.1%} {r.n_evaluations:>6} {r.window_days:>14}"
        )
    return "\n".join(lines)


class TemperatureMarketFamily(MarketFamily):
    """Scores the real, live `KXHIGH<STATION>` ladder (default: `KXHIGHNY`)
    via a real climatological empirical distribution (Open-Meteo archive),
    basis-corrected against this family's own real measured gap to its
    real settlement benchmark (The Weather Company). All construction-time
    work (fetching/caching the real historical series, measuring the real
    basis against real settled brackets) happens once in `__init__`, same
    pattern as `WTIMarketFamily`.

    ## Second station: Miami (`KXHIGHMIA`), added 2026-10-22

    v0.10.0's own "What's next" named the other confirmed-real, currently-
    open `KXHIGH<STATION>` series (`KXHIGHCHI`/`KXHIGHLAX`/`KXHIGHAUS`/
    `KXHIGHMIA`) as worth considering once NYC had more history. A real
    live check this run found `KXHIGHCHI`/`KXHIGHAUS` have no real open
    markets right now, and `KXHIGHLAX` has zero real settled history (no
    `"between"` brackets at all yet) -- not enough real data to measure a
    basis against. `KXHIGHMIA` does have real settled history (132+ real
    `"between"` brackets found on the first live page alone), so it's the
    real second station built here; its own real basis/climatology
    results are measured and reported independently of NYC's, not averaged
    together or assumed to behave the same way.

    **A real, disclosed bug found and fixed while adding this:**
    `__init__` previously called `fetch_nyc_temp_daily` without ever
    forwarding its own `cache_path` argument, so *any* station built with
    this class -- not just NYC -- would have silently read and written
    `data/nyc_temp_daily.csv`. This was never hit before because NYC was
    the only station ever instantiated, but it would have silently
    corrupted a second station's own cached series with NYC's temperature
    readings (or vice versa) the moment one was built. Fixed by defaulting
    `cache_path` to `temperature_data.default_cache_path_for_station(series_ticker)`
    (`"nyc_temp_daily.csv"` for `KXHIGHNY`, unchanged from before --
    `"miami_temp_daily.csv"` for `KXHIGHMIA`, new) rather than always
    reaching for the one hardcoded NYC path.
    """

    def __init__(
        self,
        series_ticker: str = "KXHIGHNY",
        latitude: float = NYC_LATITUDE,
        longitude: float = NYC_LONGITUDE,
        timezone: str = EASTERN_TIMEZONE,
        climatology_start_date: str = "2006-01-01",
        climatology_end_date: str | None = None,
        window_days: int = 7,
        refresh: bool = False,
        series: dict[str, float] | None = None,
        max_basis_events: int = 40,
        cache_path: "Path | str | None" = None,
    ) -> None:
        self.series_ticker = series_ticker
        self.latitude = latitude
        self.longitude = longitude
        self.window_days = window_days

        if climatology_end_date is None:
            climatology_end_date = (_dt.date.today() - _dt.timedelta(days=2)).isoformat()

        if cache_path is None:
            cache_path = default_cache_path_for_station(series_ticker)

        self._series = fetch_nyc_temp_daily(
            climatology_start_date,
            climatology_end_date,
            refresh=refresh,
            series=series,
            cache_path=cache_path,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
        )
        self._last_observed_date = max(self._series)
        self._last_observed_value = self._series[self._last_observed_date]

        brackets = fetch_real_settled_brackets(series_ticker)
        measurements = measure_temperature_basis(brackets, self._series)
        basis_est = recent_temperature_basis_estimate(measurements)
        station_label = _STATION_LABELS.get(series_ticker, series_ticker)
        if basis_est is not None:
            self._basis_mean, basis_stdev, basis_n = basis_est
            self._basis_trustworthy, self._basis_note = describe_temperature_basis_correction(
                self._basis_mean, basis_stdev, basis_n, station_label=station_label
            )
        else:
            self._basis_mean, self._basis_trustworthy = 0.0, False
            self._basis_note = "no real settled Weather Company brackets available to measure a basis this run"

    def list_open_markets(self) -> list[KalshiMarketQuote]:
        return fetch_markets(self.series_ticker, status="open")

    def estimate_probability(self, market: KalshiMarketQuote) -> ProbabilityEstimate:
        target_date = parse_ticker_settlement_date(market.ticker)
        if target_date is None:
            raise MarketNotScorable(f"could not parse a real settlement date from ticker {market.ticker!r}")

        # Continuity-corrected threshold(s) -- see module docstring.
        if market.strike_type == "greater" and market.floor_strike is not None:
            threshold = market.floor_strike + 0.5
        elif market.strike_type == "less" and market.cap_strike is not None:
            threshold = market.cap_strike - 0.5
        elif (
            market.strike_type == "between"
            and market.floor_strike is not None
            and market.cap_strike is not None
        ):
            lower_threshold = market.floor_strike - 0.5
            upper_threshold = market.cap_strike + 0.5
        else:
            raise MarketNotScorable(
                f"strike_type={market.strike_type!r} missing the floor_strike/cap_strike this "
                "survey needs, skipped"
            )

        days_to_close = days_to_close_from_last_observation(market.close_time, self._last_observed_date)
        if days_to_close < 0:
            raise MarketNotScorable(
                "close_time is before this project's own last observed Open-Meteo date "
                f"({self._last_observed_date}), skipped"
            )

        samples = climatological_samples(self._series, target_date, window_days=self.window_days)
        if len(samples) < 20:
            raise MarketNotScorable(
                f"only {len(samples)} real historical climatology samples available within "
                f"{self.window_days} days of {target_date} -- too few to trust an empirical quantile, skipped"
            )

        basis = self._basis_mean if self._basis_trustworthy else 0.0

        if market.strike_type == "between":
            adjusted_lower = basis_adjusted_strike(lower_threshold, basis)
            adjusted_upper = basis_adjusted_strike(upper_threshold, basis)
            p_above_lower = empirical_exceedance_probability(samples, adjusted_lower)
            p_above_upper = empirical_exceedance_probability(samples, adjusted_upper)
            our_probability = max(0.0, p_above_lower - p_above_upper)
            adjusted_strike = (adjusted_lower + adjusted_upper) / 2.0
        elif market.strike_type == "greater":
            adjusted_strike = basis_adjusted_strike(threshold, basis)
            our_probability = empirical_exceedance_probability(samples, adjusted_strike)
        else:  # "less"
            adjusted_strike = basis_adjusted_strike(threshold, basis)
            our_probability = 1.0 - empirical_exceedance_probability(samples, adjusted_strike)

        return ProbabilityEstimate(
            our_probability=float(our_probability),
            adjusted_strike=float(adjusted_strike),
            settlement_benchmark_matches_forecast=self._basis_trustworthy,
            forecast_horizon_days=None,
            # ^ Unlike WTI's autoregressive forecaster, this family's real climatological
            # method does not lose accuracy as `days_to_close` grows within any real
            # horizon a Kalshi daily temperature ladder actually lists (same calendar-day
            # empirical distribution regardless of how many days out "today" is) -- so the
            # horizon-mismatch caveat `edge.compute_edge` would otherwise attach doesn't
            # honestly apply here, and is skipped by passing `None` rather than forced to
            # fit WTI's own horizon-validated framing.
            days_to_close=days_to_close,
            reference_price=self._last_observed_value,
            basis_note=self._basis_note,
        )
