"""The real crude-storage market family: `KXEIACRUDEW`, "U.S. crude oil
inventories" -- a real, live, weekly Kalshi series asking whether the EIA's
own published U.S. commercial crude oil inventory figure (excluding the
Strategic Petroleum Reserve) for a given week will land above a given
strike, scored via this project's own horizon-validated calibrated
forecaster (`forecast.py`) trained on the real EIA weekly storage series
itself (`crude_storage_data.py`).

## Why this family, and how it differs from WTI and temperature

`active-project-kalshi.md`'s 2026-10-21 direction update named "EIA weekly
storage/inventory-report markets" as the second new market family to add
(after temperature), on the premise that a market settling directly
against a real, free, EIA-published number is a genuinely different kind
of edge question than WTI's own price-prediction problem. Found live,
2026-10-22, by searching every real Kalshi series across every category
for one whose title or settlement source mentions EIA, storage, or
inventory (92 real candidates found this way -- see this run's own
research notes): `KXEIACRUDEW`, "U.S. crude oil inventories," weekly
frequency, real open markets confirmed live with real volume.

**A genuinely cleaner settlement situation than either prior family.**
WTI's `KXWTI` settles against the ICE futures front-month settlement
price, a real but *different* number from this project's own EIA spot
series (the "benchmark mismatch" caveat, corrected via bracket
reconstruction). Temperature's `KXHIGH<STATION>` settles against The
Weather Company's own station reading, not NWS or Open-Meteo directly
(a disclosed climatology pivot). `KXEIACRUDEW`, by contrast, settles
**directly against the same real EIA series this project already
ingests** -- confirmed two ways, not assumed: (1) a real finalized
market's own `rules_secondary` text names "the U.S. Energy Information
Administration (EIA)" and "excludes crude oil held in the Strategic
Petroleum Reserve (SPR)" explicitly, matching `WCESTUS1`'s own real
series title word for word; (2) Kalshi's own API reports a real
finalized market's exact settled value directly, as a plain string, on
every market's `expiration_value` field (e.g. `"412.5M"`) -- no bracket
reconstruction needed at all, since the real number is just given. This
is the most direct of the three families' real settlement-verification
methods (see `measure_crude_storage_basis` below for the real, measured
check across many real settled weeks, which still finds a small, real,
honestly-reported gap -- see "A genuinely remaining caveat" below for
why this is *not* zero even with the same real series on both sides).

## Why climatology would be the wrong tool here (unlike temperature)

Unlike daily temperature (dominated by a strong seasonal cycle), weekly
U.S. crude storage is a genuinely autocorrelated time series driven by
real supply/demand/refinery-utilization dynamics week to week -- the
right tool is the same kind of lookback-window, conformal-calibrated
point forecaster `markets/wti.py` already uses for WTI spot, just fit on
a weekly-step series instead of a daily one (`forecast.fit_calibrated_forecaster_horizon`
and `prepare_horizon_split` are already generic over the series' own
native step size -- a "horizon" is a count of rows in whatever series is
passed in, not hardcoded to calendar days -- so no change to either
function was needed, only a correct real horizon-in-weeks-to-days
mapping here).

## A real, disclosed design decision: `days_to_close` measures distance to
## the target week, not to the market's own `close_time`

A real, live `KXEIACRUDEW` market's own `close_time` is **not** when its
target week ends -- it's roughly 5-6 real calendar days *later*, when EIA
actually publishes that week's report (confirmed across all 8 real
settled events checked live, 2026-10-22: lag of 5 real days in 6 of 8
events, 6 real days in the other 2 -- consistent with EIA's own real
Wednesday-after-the-reporting-Friday publication schedule, shifting a day
later around a holiday). That gap is a real administrative publication
delay, not real forecasting uncertainty: the true value for a given week
is already historically fixed once that week ends, regardless of when
EIA gets around to announcing it. `edge.compute_edge`'s own generic
forecast-horizon-mismatch caveat compares a market's real `days_to_close`
against this project's own validated `forecast_horizon_days` -- the
honest quantity to feed it here is therefore the real distance to the
**target week's end** (what this project's forecaster actually answers a
question about), not the real distance to the market's own `close_time`
(which would overstate how far out this family is really forecasting by
exactly that same real 5-6-day administrative lag, every single time,
and spuriously trip the horizon-mismatch caveat on every live market even
though this family's one validated horizon, 1 week, is the real, exact
question being asked). This is a disclosed methodological choice, not a
way of hiding the real lag -- see `CrudeStorageMarketFamily.estimate_probability`'s
own docstring for where it's applied, and `tests/test_markets_crude_storage.py`
for a real, pinned test confirming the lag itself is correctly measured
and NOT silently folded into the probability computation.

## A genuinely remaining caveat, measured, not assumed: first-published vs.
## this project's own currently-cached value

Every real `KXEIACRUDEW` market's own `rules_secondary` text states
plainly: *"The market will use the first EIA publication for the
specified week. Subsequent revisions will not affect resolution."* This
project's own cached `WCESTUS1` series (`crude_storage_data.py`), like
every EIA series this project ingests, reflects whatever EIA's API
currently reports -- which, per this project's own prior real finding on
the WTI spot series (see `active-project-kalshi.md`'s twelfth-scheduled-
run entry: a genuine 3-row EIA revision found and double-checked), can
be a **later-revised** value, not necessarily the exact first-published
one a real `KXEIACRUDEW` market actually settled against. `measure_crude_storage_basis`
below measures this real, honest gap directly (Kalshi's own real recorded
`expiration_value` minus this project's own real cached series value for
the same real week) across many real recent settled events, the same
"measure it, don't assume it's zero" discipline `benchmark.py`/
`markets/temperature.py` already apply to their own, structurally
different settlement gaps -- and `describe_crude_storage_basis_correction`
applies the same kind of trust gate before this project ever corrects a
live strike for it.
"""

from __future__ import annotations

import datetime as _dt
import re

import numpy as np

from commodity_forecast.benchmark import BasisMeasurement, basis_adjusted_strike, recent_basis_estimate
from commodity_forecast.crude_storage_data import fetch_crude_storage_weekly
from commodity_forecast.forecast import fit_calibrated_forecaster_horizon
from commodity_forecast.kalshi_client import KalshiMarketQuote, fetch_markets
from commodity_forecast.markets.base import MarketFamily, MarketNotScorable, ProbabilityEstimate
from commodity_forecast.probability import exceedance_probability

#: This project has only ever validated (backtested -- see
#: `tests/test_markets_crude_storage.py`) a genuine one-week-ahead
#: forecast for this series, which also happens to be the only horizon a
#: real, live `KXEIACRUDEW` ladder has ever been observed to ask about
#: (Kalshi lists exactly one open event at a time -- the next
#: not-yet-published week -- confirmed live, 2026-10-22). A market asking
#: about any other real horizon is honestly skipped rather than
#: extrapolated past where this project has measured its own calibration,
#: same discipline as WTI's `VALIDATED_FORECAST_HORIZONS_DAYS`.
VALIDATED_HORIZONS_WEEKS: tuple[int, ...] = (1,)

_WEEK_ENDING_RE = re.compile(r"week ending ([A-Za-z]{3,9}\.? \d{1,2},? \d{4})")


def parse_target_week_ending_date(title: str) -> _dt.date | None:
    """The real target week-ending date encoded in a real `KXEIACRUDEW`
    market's own `title` text (e.g. `"...for the week ending Oct 16,
    2026 be above..."` -> `date(2026, 10, 16)`), or `None` if the title
    doesn't match the expected real format.

    Deliberately parsed from `title` (equivalently `rules_primary`), not
    from the market's own ticker date segment (`"KXEIACRUDEW-26OCT21-T422"`'s
    `"26OCT21"` is the real report *close* date, several real days after
    the target week itself ends -- confirmed live: the Oct-21-closing
    event's own real markets all target "the week ending Oct 16, 2026",
    a 5-real-day gap, matching EIA's real weekly publication lag).
    """
    m = _WEEK_ENDING_RE.search(title)
    if not m:
        return None
    text = m.group(1).replace(",", "").replace(".", "")
    try:
        return _dt.datetime.strptime(text, "%b %d %Y").date()
    except ValueError:
        try:
            return _dt.datetime.strptime(text, "%B %d %Y").date()
        except ValueError:
            return None


_EXPIRATION_VALUE_RE = re.compile(r"^([\d.]+)\s*M?$")

#: Real U.S. commercial crude oil storage (excl. SPR) has not been outside
#: roughly 200-600 million barrels anywhere in this project's own full real
#: cached history (1983-present -- see `crude_storage_data.py`). Used below
#: only to disambiguate a real, observed Kalshi metadata quirk (see
#: `parse_expiration_value_million_bbl`'s own docstring), never to reject or
#: clip a real forecast or edge computation.
_PLAUSIBLE_MILLION_BBL_CEILING = 10_000.0


def parse_expiration_value_million_bbl(expiration_value: str | None) -> float | None:
    """A real finalized `KXEIACRUDEW` market's own `expiration_value`
    field, as a plain float in million barrels, or `None` if it's missing
    or doesn't match any real format observed.

    **A real, disclosed Kalshi metadata inconsistency, found live
    (2026-10-22), not assumed:** across 8 real recent settled
    `KXEIACRUDEW` events checked, `expiration_value` appeared in three
    different real formats for the *same* real underlying quantity --
    `"412.5M"` (million barrels, with an `M` suffix), `"415.6"` (million
    barrels, no suffix at all), and `"424069"` (the EIA's own raw
    thousand-barrel figure, no decimal point, no suffix -- e.g. `424069`
    thousand = `424.069` million, matching this project's own cached
    `WCESTUS1` value for that same real week almost exactly). Since real
    U.S. crude storage has never been within an order of magnitude of
    10,000 "million barrels" (reported values only ever run roughly
    200-600), a bare numeric value above `_PLAUSIBLE_MILLION_BBL_CEILING`
    is honestly inferred to be the raw thousand-barrel figure and divided
    by 1000; this is a disclosed inference from a real, measured range,
    not a silent unit assumption.
    """
    if not expiration_value:
        return None
    m = _EXPIRATION_VALUE_RE.match(expiration_value.strip())
    if not m:
        return None
    value = float(m.group(1))
    if expiration_value.strip().upper().endswith("M"):
        return value
    if value > _PLAUSIBLE_MILLION_BBL_CEILING:
        return value / 1000.0
    return value


def fetch_real_settled_values(series_ticker: str = "KXEIACRUDEW", max_pages: int = 10, **kwargs):
    """One real, settled value per real past settlement event (not per
    strike -- every strike in one event's ladder reports the identical
    real `expiration_value`, confirmed live, so this de-duplicates by
    `event_ticker` rather than returning one row per strike). Returns a
    list of `(target_week_ending_date, expiration_value_million_bbl)`
    pairs, one per real event that had both a parseable target date and a
    parseable expiration value, sorted ascending by date.
    """
    quotes = fetch_markets(series_ticker, status="settled", max_pages=max_pages, **kwargs)
    by_event: dict[str, KalshiMarketQuote] = {}
    for q in quotes:
        by_event.setdefault(q.event_ticker, q)

    out: list[tuple[_dt.date, float]] = []
    for q in by_event.values():
        target_date = parse_target_week_ending_date(q.title)
        value = parse_expiration_value_million_bbl(q.expiration_value)
        if target_date is None or value is None:
            continue
        out.append((target_date, value))
    return sorted(out, key=lambda pair: pair[0])


def measure_crude_storage_basis(
    settled_values: list[tuple[_dt.date, float]], series
) -> list[BasisMeasurement]:
    """Pair each real settled event's own real `expiration_value` with
    this project's own currently-cached `WCESTUS1` value for the same
    real week, and compute the real, measured gap between them --
    `basis = kalshi_expiration_value - this_project's_own_cached_value`,
    the same sign convention (`real_settlement_benchmark -
    this_project's_own_data_source`) `benchmark.measure_basis`/
    `markets.temperature.measure_temperature_basis` already use, so
    `basis_adjusted_strike`/`recent_basis_estimate` are directly reusable
    unmodified. A real week with no matching cached series observation is
    silently skipped (never interpolated), same discipline as both prior
    families.

    `BasisMeasurement.futures_mid`/`.spot` are repurposed here (as they
    already are for the temperature family) to mean "Kalshi's own real
    recorded settled value" / "this project's own real cached EIA weekly
    value" respectively; `.futures_bracket_width` is always `0.0` here,
    since `expiration_value` is Kalshi's own exact real number, not a
    reconstructed bracket with real width to report.
    """
    series_by_date = {d.date().isoformat(): float(v) for d, v in series.items()}
    out: list[BasisMeasurement] = []
    for target_date, kalshi_value in settled_values:
        our_value = series_by_date.get(target_date.isoformat())
        if our_value is None:
            continue
        out.append(
            BasisMeasurement(
                date=target_date.isoformat(),
                futures_mid=kalshi_value,
                futures_bracket_width=0.0,
                spot=our_value,
                basis=kalshi_value - our_value,
            )
        )
    return out


def describe_crude_storage_basis_correction(
    mean: float, stdev: float, n: int, max_stdev_for_trust: float = 1.0, min_n_for_trust: int = 5
) -> tuple[bool, str]:
    """The crude-storage-family equivalent of `benchmark.describe_basis_correction`,
    with real, correct units/wording (million barrels, Kalshi's own
    recorded settled value vs. this project's own cached EIA weekly
    series) -- same trust-gate defaults as every other family in this
    project (stdev <= 1.0, n >= 5), applied to this family's own real
    measured numbers, never assumed to pass."""
    trustworthy = n >= min_n_for_trust and stdev <= max_stdev_for_trust
    note = (
        f"crude-storage settlement-vs-cache basis measured from {n} real paired observations "
        f"(Kalshi's own real recorded settled value vs. this project's own cached EIA weekly "
        f"series value for the same real week): mean {mean:+.3f} million barrels, stdev "
        f"{stdev:.3f} -- {'tight and well-supported enough to trust' if trustworthy else 'NOT trusted (too few points or too much spread)'}"
    )
    return trustworthy, note


class CrudeStorageMarketFamily(MarketFamily):
    """Scores the real, live `KXEIACRUDEW` ladder via a real
    horizon-validated calibrated forecaster trained on this project's own
    real cached EIA weekly crude-storage series. All construction-time
    work (fetching/caching the real series, measuring the real
    settlement-vs-cache basis from real settled events) happens once in
    `__init__`, same pattern as `WTIMarketFamily`/`TemperatureMarketFamily`.
    """

    def __init__(
        self,
        series_ticker: str = "KXEIACRUDEW",
        lookback: int = 5,
        alpha: float = 0.1,
        refresh: bool = False,
        series=None,
        model_name: str = "naive",
        seed: int = 0,
        max_basis_events: int = 20,
    ) -> None:
        self.series_ticker = series_ticker
        self.lookback = lookback
        self.alpha = alpha
        self.model_name = model_name
        self.seed = seed

        self._series = series if series is not None else fetch_crude_storage_weekly(refresh=refresh)
        self._last_observed_date = self._series.index.max()
        self._last_observed_value = float(self._series.iloc[-1])

        settled_values = fetch_real_settled_values(series_ticker, max_pages=max(1, max_basis_events // 5 + 1))
        measurements = measure_crude_storage_basis(settled_values, self._series)
        recent = measurements[-max_basis_events:] if measurements else []
        if recent:
            values = [m.basis for m in recent]
            n = len(values)
            mean = sum(values) / n
            stdev = (sum((v - mean) ** 2 for v in values) / n) ** 0.5 if n > 1 else 0.0
            self._basis_mean, self._basis_trustworthy, self._basis_note = (
                mean,
                *describe_crude_storage_basis_correction(mean, stdev, n),
            )
        else:
            self._basis_mean, self._basis_trustworthy = 0.0, False
            self._basis_note = "no real settled KXEIACRUDEW events available to measure a basis this run"

        self._fitted_by_horizon: dict[int, object] = {}

    def list_open_markets(self) -> list[KalshiMarketQuote]:
        return fetch_markets(self.series_ticker, status="open")

    def estimate_probability(self, market: KalshiMarketQuote) -> ProbabilityEstimate:
        if market.strike_type != "greater" or market.floor_strike is None:
            raise MarketNotScorable(
                f"strike_type={market.strike_type!r} is not a simple 'above X million barrels' "
                "question this family knows how to score, skipped"
            )

        target_date = parse_target_week_ending_date(market.title)
        if target_date is None:
            raise MarketNotScorable(f"could not parse a real target week-ending date from title {market.title!r}")

        weeks_to_target = (target_date - self._last_observed_date.date()).days / 7.0
        horizon_weeks = round(weeks_to_target)
        if abs(weeks_to_target - horizon_weeks) > 1e-6 or horizon_weeks not in VALIDATED_HORIZONS_WEEKS:
            raise MarketNotScorable(
                f"real weeks_to_target={weeks_to_target:.2f} is not one of this project's "
                f"validated horizons {VALIDATED_HORIZONS_WEEKS} (weeks past this project's own "
                f"last observed series date {self._last_observed_date.date().isoformat()}), skipped"
            )

        # Real distance to the TARGET WEEK's own end, not to the market's `close_time` --
        # see the module docstring's "A real, disclosed design decision" section for why:
        # `close_time` is ~5-6 real days later than `target_date` (EIA's own publication
        # lag, not real forecasting uncertainty), and would otherwise spuriously trip
        # `edge.compute_edge`'s forecast-horizon-mismatch caveat on every live market even
        # though this family's one validated horizon is the real, exact question asked.
        days_to_close = float((target_date - self._last_observed_date.date()).days)
        if days_to_close < 0:
            raise MarketNotScorable(
                "target week-ending date is before this project's own last observed series date "
                f"({self._last_observed_date.date().isoformat()}), skipped"
            )

        if horizon_weeks not in self._fitted_by_horizon:
            self._fitted_by_horizon[horizon_weeks] = fit_calibrated_forecaster_horizon(
                self._series.values,
                horizon=horizon_weeks,
                model_name=self.model_name,
                lookback=self.lookback,
                alpha=self.alpha,
                seed=self.seed,
            )
        result = self._fitted_by_horizon[horizon_weeks]

        last_window_normalized = (
            (np.asarray(self._series.values[-self.lookback :], dtype=np.float64) - result.normalizer_mean)
            / result.normalizer_std
        ).reshape(1, -1)
        point_forecast_normalized = result.conformal_forecaster.model.predict(last_window_normalized).reshape(-1)

        adjusted_strike = (
            basis_adjusted_strike(market.floor_strike, self._basis_mean)
            if self._basis_trustworthy
            else market.floor_strike
        )
        threshold_normalized = (adjusted_strike - result.normalizer_mean) / result.normalizer_std

        our_probability = float(
            exceedance_probability(
                point_forecast_normalized, result.cal_residuals_normalized, threshold_normalized
            )[0]
        )

        return ProbabilityEstimate(
            our_probability=our_probability,
            adjusted_strike=adjusted_strike,
            settlement_benchmark_matches_forecast=self._basis_trustworthy,
            forecast_horizon_days=horizon_weeks * 7,
            days_to_close=days_to_close,
            reference_price=self._last_observed_value,
            basis_note=self._basis_note,
        )
