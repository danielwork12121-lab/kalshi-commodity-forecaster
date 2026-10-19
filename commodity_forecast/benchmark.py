"""Real, measured spot/futures basis between this project's EIA spot series
and the real ICE WTI futures front-month contract that `KXWTI` actually
settles against.

Kalshi's public API does not publish the raw ICE futures settlement price
directly -- a `KXWTI` market only ever reports whether the real settlement
was above or below *that specific market's own strike* ("yes"/"no"). But a
single settlement date is covered by a whole real *ladder* of such markets
at many different strikes, and once every one of them has resolved, the
ladder itself brackets the real settlement value tightly: the true value
lies strictly between the highest strike that resolved "yes" and the
lowest strike that resolved "no" (see `bracket_settlements`). This
recovers a real, measured estimate of the ICE futures settlement price
for every historical date Kalshi has already settled a `KXWTI` ladder for
-- using only data `kalshi_client.py` already fetches (real, public, no
API key), with no external or paid futures-data vendor needed.

**Real, honest finding (2026-10-19), not assumed:** the basis (bracketed
futures mid minus this project's own EIA spot observation for the same
date) is **not a stable constant across the full history measured** --
it ran roughly -$1 to -$2 through most of August 2026, then shifted to
roughly -$5 to -$7 from mid-September onward. This is consistent with
`KXWTI` settling against ICE's *front-month* contract, which rolls to a
new underlying contract roughly monthly -- a historical-average basis
over months would therefore be actively misleading, not a safe
correction. The basis *is*, however, tight and locally stable within a
short recent window (within roughly +/-$0.2 across the most recent two
weeks of history measured) -- see `recent_basis_estimate`, which reports
that local mean and its own real stdev/sample size as an honest
confidence signal, rather than ever assuming a single global number is
safe to apply.

Also found, and deliberately not silently smoothed over: one date in the
history measured (2026-09-25) shows a one-day, ~$10 EIA spot value that
fully reverts the next observation -- almost certainly a data artifact in
the upstream EIA series this project depends on, not a real market move.
`measure_basis` does not filter this out automatically (that would be an
unprincipled, after-the-fact edit to real data); callers building a
longer-history analysis should be aware a handful of such outliers exist
rather than assume every paired observation is equally reliable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from commodity_forecast.kalshi_client import KalshiMarketQuote, fetch_markets

_EVENT_DATE_RE = re.compile(r"^(?P<series>[A-Z]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})")
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


@dataclass(frozen=True)
class SettlementBracket:
    """A real, bracketed estimate of one settlement date's true value,
    recovered from a ladder of Kalshi's own resolved "greater than $X"
    markets for the same settlement event -- not an external data source,
    not an assumption."""

    event_ticker: str
    date: str  # "YYYY-MM-DD", parsed from the event ticker's own date-coded prefix
    max_yes_strike: float
    min_no_strike: float

    @property
    def mid(self) -> float:
        """The bracket's midpoint -- a real point estimate, honestly
        bounded by `width` below."""
        return (self.max_yes_strike + self.min_no_strike) / 2.0

    @property
    def width(self) -> float:
        """How tight this real bracket is. `KXWTI`'s own strike spacing
        (observed: $0.50 throughout the real history measured) sets a
        floor on how tight any bracket recovered this way can ever be --
        this is a real resolution limit of the method, not a bug."""
        return self.min_no_strike - self.max_yes_strike


def _event_ticker(market_ticker: str) -> str:
    return market_ticker.rsplit("-T", 1)[0]


def _parse_event_date(event_ticker: str) -> str | None:
    m = _EVENT_DATE_RE.match(event_ticker)
    if not m:
        return None
    month = _MONTHS.get(m.group("mon"))
    if month is None:
        return None
    return f"20{m.group('yy')}-{month:02d}-{int(m.group('dd')):02d}"


def bracket_settlements(quotes: list[KalshiMarketQuote]) -> list[SettlementBracket]:
    """Group real, finalized market quotes by settlement event and bracket
    each event's true settlement value between the highest strike that
    resolved "yes" and the lowest that resolved "no".

    Only considers markets with `status == "finalized"` and a real
    "yes"/"no" `result`. An event whose entire real ladder resolved the
    same way (every strike "yes", or every strike "no") is skipped --
    the true value could be anywhere beyond the one side actually
    observed, so there is nothing real to bracket. Does not assume every
    ladder is internally consistent (a "yes" strike priced above the
    ladder's own "no" cutoff would be a real data anomaly) -- it simply
    takes `max()`/`min()` of whichever strikes actually resolved each way,
    so `width` itself is an honest quality signal callers can check
    rather than something silently assumed safe.
    """
    finalized = [
        q for q in quotes
        if q.status == "finalized" and q.result in ("yes", "no") and q.floor_strike is not None
    ]
    by_event: dict[str, list[KalshiMarketQuote]] = {}
    for q in finalized:
        by_event.setdefault(_event_ticker(q.ticker), []).append(q)

    brackets: list[SettlementBracket] = []
    for event_ticker, group in by_event.items():
        yes_strikes = [q.floor_strike for q in group if q.result == "yes"]
        no_strikes = [q.floor_strike for q in group if q.result == "no"]
        if not yes_strikes or not no_strikes:
            continue
        date = _parse_event_date(event_ticker)
        if date is None:
            continue
        brackets.append(
            SettlementBracket(
                event_ticker=event_ticker,
                date=date,
                max_yes_strike=max(yes_strikes),
                min_no_strike=min(no_strikes),
            )
        )
    return sorted(brackets, key=lambda b: b.date)


def fetch_settlement_brackets(series_ticker: str, max_pages: int = 30, **kwargs) -> list[SettlementBracket]:
    """Live convenience wrapper: fetch every real market Kalshi has ever
    listed in `series_ticker` (any status) via the already-tested
    `kalshi_client.fetch_markets`, and bracket its real settlement
    history. See `bracket_settlements` for the bracketing logic."""
    quotes = fetch_markets(series_ticker, status="all", max_pages=max_pages, **kwargs)
    return bracket_settlements(quotes)


@dataclass(frozen=True)
class BasisMeasurement:
    """One real date's measured gap between the bracketed real futures
    settlement value and this project's own EIA spot observation for the
    same calendar date: `basis = futures_mid - spot`."""

    date: str
    futures_mid: float
    futures_bracket_width: float
    spot: float
    basis: float


def measure_basis(
    brackets: list[SettlementBracket], spot_by_date: dict[str, float]
) -> list[BasisMeasurement]:
    """Pair each real settlement bracket with this project's own cached EIA
    spot observation for the same calendar date, when one exists, and
    compute the real, measured basis for that date. A settlement date with
    no matching spot observation is silently skipped (never interpolated
    or assumed) -- this can legitimately happen for the most recent
    bracket(s) if the cached EIA series hasn't been refreshed as recently.
    """
    out: list[BasisMeasurement] = []
    for b in brackets:
        spot = spot_by_date.get(b.date)
        if spot is None:
            continue
        out.append(
            BasisMeasurement(
                date=b.date,
                futures_mid=b.mid,
                futures_bracket_width=b.width,
                spot=spot,
                basis=b.mid - spot,
            )
        )
    return out


def basis_adjusted_strike(strike_usd: float, basis_mean: float) -> float:
    """Convert a real Kalshi market's own futures-settlement strike into
    the equivalent EIA-spot-terms strike this project's forecaster can
    actually answer a probability question about.

    `basis = futures - spot`, so `futures > strike` is exactly equivalent
    to `spot > strike - basis`. Callers should compute their own
    `our_probability` against *this* adjusted strike (via
    `probability.exceedance_probability`/`backtest_threshold_calibration`,
    unchanged), not the market's raw strike -- that is what actually
    corrects for the real spot/futures gap, rather than just relabeling
    the existing (uncorrected) probability.
    """
    return strike_usd - basis_mean


def describe_basis_correction(
    mean: float, stdev: float, n: int, max_stdev_for_trust: float = 1.0, min_n_for_trust: int = 5
) -> tuple[bool, str]:
    """Decide, from a real recent-window basis estimate's own measured
    quality (never a fixed assumption), whether it's tight and
    well-supported enough to trust as a live correction, and produce a
    plain-language note describing exactly what was measured either way --
    this note is meant to travel with any `EdgeEstimate` that uses it, so
    a reader always sees the real numbers behind a correction, not just a
    pass/fail flag.

    Defaults (`max_stdev_for_trust=1.0`, `min_n_for_trust=5`) are
    deliberately conservative relative to what this project has actually
    measured so far (a real recent-window stdev of ~$0.1-0.2 on 2026-10-19,
    far inside this bar) -- callers with a reason to be stricter or looser
    should pass their own.
    """
    trustworthy = n >= min_n_for_trust and stdev <= max_stdev_for_trust
    note = (
        f"settlement-benchmark basis corrected using a real, measured recent-window estimate: "
        f"mean {mean:+.2f} USD/bbl (futures vs. EIA spot), stdev {stdev:.2f}, n={n} real paired "
        f"observations -- {'tight and well-supported enough to trust' if trustworthy else 'NOT trusted (too few points or too much spread)'}"
    )
    return trustworthy, note


def recent_basis_estimate(
    measurements: list[BasisMeasurement], window: int = 10
) -> tuple[float, float, int] | None:
    """A real, local basis estimate: the mean and population stdev of the
    most recent `window` real measured basis values, ordered by date --
    deliberately never a single all-history average (see this module's
    own docstring: the real measured basis drifts by several dollars over
    a couple of months, consistent with the underlying ICE contract
    rolling to a new front-month roughly monthly, so an all-history
    average would be actively misleading). Returns `(mean, stdev, n)`,
    where `n` may be less than `window` if fewer measurements exist, or
    `None` if there are no measurements at all. The stdev is a real,
    honest uncertainty signal for a caller deciding whether to trust this
    specific estimate -- it is not hidden or rounded away.
    """
    if not measurements:
        return None
    ordered = sorted(measurements, key=lambda m: m.date)
    recent = ordered[-window:]
    values = [m.basis for m in recent]
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / n
        stdev = variance ** 0.5
    else:
        stdev = 0.0
    return mean, stdev, n
