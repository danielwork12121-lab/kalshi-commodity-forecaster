"""Real "edge" vs. a real Kalshi market, and a real, paper-only, non-backdated
bet-sizing + track record -- the concrete build for Daniel's 2026-10-16
directive (see `active-project-kalshi.md`'s "Edge/profitability narrative"
section): *"I use this data to predict that data, and thus I've made a
calculated bet that beats the edge, and I'm profitable."*

What this module actually is, stated plainly up front: a real pipeline that
(1) fetches a real, live, public Kalshi market quote (`kalshi_client.py`),
(2) computes this project's own already-backtested calibrated probability
for the same threshold question via `probability.exceedance_probability`,
(3) reports the difference as an "edge" -- **with the real caveats attached
to every single estimate**, not buried in prose -- and (4) if, and only if,
the estimate looks trustworthy enough to act on *on paper*, sizes and logs a
**paper-only, never-executed, never-backdated** bet the same honest way
`track_record.py` already logs forward price predictions, resolved later
against the real, actual Kalshi market outcome (never against this
project's own forecast, which would be grading its own homework).

## The honest headline finding so far (2026-10-16)

Building this surfaced a real problem with the "profitable" story before any
bet is sized, and this module does not paper over it:

**Benchmark mismatch.** `KXWTI` markets settle against the *ICE WTI Crude
Futures front-month contract's own daily settlement price* (confirmed via
`kalshi_client.fetch_series("KXWTI")["settlement_sources"]` ->
`[{"name": "ICE", "url": ".../WTI-Crude-Futures"}]`, and via each market's
own `rules_primary` text, e.g. *"the daily settlement price for WTI crude
oil (December 2026 contract)..."*). This project's forecaster predicts the
EIA Cushing, OK **spot** price (`data.py`'s `RWTC` series) -- a real, genuine,
correlated-but-distinct number. Live on 2026-10-16: this project's own
cached spot series showed WTI at **$101.91/bbl** (2026-10-14's close) while
the real, live `KXWTI` Oct-19 ladder's own 50%-crossing strike sat around
**$95.50-96.00** -- i.e. the market was pricing the Dec-2026 ICE futures
settlement roughly $6/bbl *below* the spot price our model forecasts from.
Three trading days of forecast uncertainty cannot explain a gap that size;
it is almost certainly the real futures/spot basis (backwardation), not a
genuine mispricing our model has found. Computing "edge" as
`our_spot_based_probability - market_futures_based_probability` without
correcting for this would overstate the edge and would not be an honest
number.

**Liquidity.** The same live ladder showed `volume_fp` and `open_interest_fp`
of `0.00` on every single strike (a freshly-listed daily market, opened
2026-10-15). A zero-volume market's bid-ask is a maker-seeded placeholder,
not a real trading consensus -- a second, independent reason not to trust a
raw edge number from a market like that.

**Forecast-horizon mismatch -- found while building the real illustrative
example below.** `fit_calibrated_naive_forecaster` (see `forecast.py`) is
only ever validated as a true **single-trading-day-ahead** forecaster --
every coverage/calibration number in this project's README is measured one
step ahead, never further. Live on 2026-10-16, pulling the full real
`KXWTI` open-market ladder (135 real markets, many with real, non-zero
volume -- e.g. `KXWTI-...-T74.99` showed `volume=24393`, a genuinely liquid
market, not a placeholder) surfaced markets closing **more than two weeks**
out. Naively re-using the one-step-ahead calibration residual pool against
a question that far out produced probabilities like `0.999` for "still
above $95.49 in three weeks" -- i.e. near-total, overconfident certainty,
for a horizon this project's forecaster has never been validated at and
has no real reason to be well-calibrated for (price uncertainty compounds
with time; a residual pool measured for one day ahead understates it badly
multiple weeks out). This produced apparent "edges" of 60-95 percentage
points against the real market price on several real strikes -- numbers
that look spectacular and are, honestly, almost certainly **not** real
skill: they are the predictable result of applying a 1-day-validated
method to a 15-to-20-day-ahead question.

**What this means for "profitable" claims:** none are made here yet, and
none should be, honestly, until these are resolved -- see "What's next" in
the README's new section. `compute_edge` below reports all three issues
(benchmark mismatch, illiquidity, horizon mismatch) as explicit, structured
caveats on every single estimate (never silently), and `size_paper_bet`
refuses to size a bet at all when `EdgeEstimate.trustworthy` is `False`
(see its own docstring) specifically so this honesty constraint is enforced
in code, not just in a comment someone could skip past.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from dataclasses import asdict, dataclass, field

from commodity_forecast.kalshi_client import KalshiMarketQuote

PAPER_BETS_DIR = pathlib.Path(__file__).resolve().parent.parent / "predictions"
DEFAULT_PAPER_LOG_PATH = PAPER_BETS_DIR / "paper_bets.jsonl"

# A real, stated, conservative fractional-Kelly cap -- never full Kelly (full
# Kelly is famously overaggressive under any real model-probability error,
# and this project's own model is already honestly known to be imperfect --
# see the README's coverage-gap and N-BEATS results). This is a paper-only
# simulation parameter, not a live risk control, since no real money or real
# order is ever involved -- see the module docstring and README.
DEFAULT_KELLY_FRACTION_CAP = 0.05  # cap the *sizing fraction itself*, not literal quarter-Kelly
DEFAULT_KELLY_SHRINKAGE = 0.25  # apply quarter-Kelly to the raw Kelly fraction before capping


def _now_utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def market_implied_probability(quote: KalshiMarketQuote) -> float:
    """The market's own implied probability of the "YES" outcome: the
    mid-price of the real, current yes bid/ask -- the standard, simplest
    honest read of a binary prediction-market price as a probability (a
    $1-notional contract priced at $0.12 mid is pricing the event at ~12%).
    Uses the midpoint rather than either side alone so this isn't
    systematically biased toward whichever side happens to be posted
    tighter.
    """
    return (quote.yes_bid + quote.yes_ask) / 2.0


@dataclass
class EdgeEstimate:
    """One real, timestamped comparison of this project's own calibrated
    probability against one real, live Kalshi market's own implied
    probability -- with the actual caveats that bear on whether the
    resulting `edge` number should be trusted, computed and attached here
    rather than left to whoever reads the number later.
    """

    computed_at_utc: str
    kalshi_ticker: str
    kalshi_title: str
    strike_usd: float | None
    kalshi_close_time: str
    our_probability: float
    market_probability: float
    market_yes_bid: float
    market_yes_ask: float
    market_volume: float
    market_open_interest: float
    edge: float  # our_probability - market_probability; positive = we think YES is more likely than the market does
    caveats: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """False whenever a real, known reason exists not to trust this
        edge estimate at face value -- i.e. whenever `caveats` is non-empty.
        `size_paper_bet` refuses to size a bet when this is `False` (see its
        own docstring) -- the one place in this module where the honesty
        caveat above is actually enforced, not just documented.
        """
        return len(self.caveats) == 0


def compute_edge(
    our_probability: float,
    quote: KalshiMarketQuote,
    strike_usd: float | None = None,
    settlement_benchmark_matches_forecast: bool = False,
    min_volume_for_trust: float = 1.0,
    forecast_horizon_days: int | None = None,
    days_to_close: int | None = None,
) -> EdgeEstimate:
    """Compute one real `EdgeEstimate` for one real market quote.

    `settlement_benchmark_matches_forecast` defaults to `False` -- the
    honest default for this project's *current* state, where the forecaster
    predicts EIA spot and `KXWTI` settles against ICE futures (see the
    module docstring). Pass `True` only once this project has actually
    verified (not assumed) that the specific series being queried settles
    against the same real number the forecast predicts -- e.g. after the
    "What's next" work of sourcing a real, free, historical EIA-vs-ICE-WTI
    basis series, or finding a different Kalshi series that settles against
    EIA's own published spot index directly.

    `forecast_horizon_days`/`days_to_close`: pass both (e.g.
    `forecast_horizon_days=1` for `fit_calibrated_naive_forecaster`, and
    `days_to_close` computed from `quote.close_time` minus "now") to get the
    real horizon-mismatch caveat documented in the module docstring --
    flagged whenever `days_to_close` exceeds `forecast_horizon_days`, since
    this project's calibration residual pool has only ever been measured
    and validated at `forecast_horizon_days` ahead. Leaving either as
    `None` skips this specific check (it still defaults to the other two
    honest caveats) -- callers should pass real values whenever they're
    known rather than relying on that default.
    """
    caveats: list[str] = []
    if not settlement_benchmark_matches_forecast:
        caveats.append(
            "settlement-benchmark mismatch: this project's forecaster predicts the EIA Cushing "
            "spot price; this Kalshi market settles against a different real benchmark (see "
            "kalshi_client.fetch_series(...)['settlement_sources'] and this market's own "
            "rules_primary text) -- the computed edge may just be the real spot/futures basis, "
            "not genuine model skill"
        )
    if not quote.is_actively_traded or quote.volume < min_volume_for_trust:
        caveats.append(
            f"low/no liquidity: volume={quote.volume}, open_interest={quote.open_interest} -- this "
            "market's bid-ask may be a maker-seeded placeholder rather than a real trading consensus"
        )
    if (
        forecast_horizon_days is not None
        and days_to_close is not None
        and days_to_close > forecast_horizon_days
    ):
        caveats.append(
            f"forecast-horizon mismatch: this market closes in {days_to_close} real day(s), but the "
            f"forecaster's calibration residual pool is only validated {forecast_horizon_days} day(s) "
            "ahead -- probabilities at this horizon are an unvalidated extrapolation and are known to "
            "run overconfident the further out they're pushed (see the module docstring's real example)"
        )

    market_prob = market_implied_probability(quote)
    return EdgeEstimate(
        computed_at_utc=_now_utc_iso(),
        kalshi_ticker=quote.ticker,
        kalshi_title=quote.title,
        strike_usd=strike_usd if strike_usd is not None else quote.floor_strike,
        kalshi_close_time=quote.close_time,
        our_probability=float(our_probability),
        market_probability=float(market_prob),
        market_yes_bid=quote.yes_bid,
        market_yes_ask=quote.yes_ask,
        market_volume=quote.volume,
        market_open_interest=quote.open_interest,
        edge=float(our_probability - market_prob),
        caveats=caveats,
    )


def _kelly_fraction(true_probability: float, price_usd: float) -> float:
    """Kelly fraction for a binary contract: pay `price_usd` now, receive
    $1.00 if YES, $0.00 otherwise, when the real probability of YES is
    `true_probability`. Standard result: f* = (p - c) / (1 - c), clipped to
    >= 0 (never short an edge that doesn't exist by going net-negative
    here -- a negative edge on YES is handled by sizing a NO bet instead,
    see `size_paper_bet`).
    """
    if price_usd <= 0.0 or price_usd >= 1.0:
        return 0.0
    f = (true_probability - price_usd) / (1.0 - price_usd)
    return max(f, 0.0)


@dataclass
class PaperBet:
    """One real, timestamped, paper-only (never executed) simulated bet,
    sized from one `EdgeEstimate` -- mirrors `track_record.PredictionRecord`'s
    own forward/non-backdated discipline exactly: `opened_at_utc` is always
    read fresh from the real clock, `outcome` starts `"pending"`, and only
    `resolve_pending_paper_bets` (below) ever changes it, and only once the
    real Kalshi market itself has actually settled.
    """

    opened_at_utc: str
    kalshi_ticker: str
    side: str  # "yes" or "no"
    entry_price_usd: float  # the real price_usd this paper bet "buys" at (yes_ask for YES, no_ask for NO)
    our_probability: float  # this side's own probability, after the side flip for NO
    edge: float  # our_probability - entry_price_usd, this side's own edge over its own cost
    kelly_fraction_raw: float
    size_fraction: float  # the actual paper position size, as a fraction of a notional paper bankroll
    kalshi_close_time: str
    caveats: list[str]
    outcome: str = "pending"  # "pending" | "won" | "lost"
    resolved_at_utc: str | None = None
    resolved_kalshi_result: str | None = None  # Kalshi's own real "yes"/"no" result string

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json_line(line: str) -> "PaperBet":
        return PaperBet(**json.loads(line))


def size_paper_bet(
    estimate: EdgeEstimate,
    kelly_shrinkage: float = DEFAULT_KELLY_SHRINKAGE,
    fraction_cap: float = DEFAULT_KELLY_FRACTION_CAP,
) -> PaperBet | None:
    """Size one real, paper-only bet from `estimate`, picking whichever side
    (YES or NO) actually has a positive edge over its own real asking price,
    shrunk to `kelly_shrinkage` of full Kelly and capped at `fraction_cap` of
    a notional paper bankroll -- standard, conservative real-world practice
    for betting under an imperfect probability model, applied here with no
    real money ever at stake.

    Returns `None` (sizes nothing) whenever:
    - `estimate.trustworthy` is `False` (see `EdgeEstimate.trustworthy`) --
      this is the one place in this module that actually *enforces* the
      honesty caveats above, rather than just documenting them;
    - neither side has a positive edge over its own real asking price (a
      genuinely flat/no-edge market, or our probability close enough to the
      market's that the ask-side spread alone eats any paper edge).
    """
    if not estimate.trustworthy:
        return None

    yes_price = estimate.market_yes_ask
    no_price = round(1.0 - estimate.market_yes_bid, 4)  # buying NO costs 1 - yes_bid, by construction

    yes_kelly = _kelly_fraction(estimate.our_probability, yes_price)
    no_kelly = _kelly_fraction(1.0 - estimate.our_probability, no_price)

    if yes_kelly <= 0.0 and no_kelly <= 0.0:
        return None

    if yes_kelly >= no_kelly:
        side, price, prob, raw_kelly = "yes", yes_price, estimate.our_probability, yes_kelly
    else:
        side, price, prob, raw_kelly = "no", no_price, 1.0 - estimate.our_probability, no_kelly

    size_fraction = min(raw_kelly * kelly_shrinkage, fraction_cap)

    return PaperBet(
        opened_at_utc=_now_utc_iso(),
        kalshi_ticker=estimate.kalshi_ticker,
        side=side,
        entry_price_usd=float(price),
        our_probability=float(prob),
        edge=float(prob - price),
        kelly_fraction_raw=float(raw_kelly),
        size_fraction=float(size_fraction),
        kalshi_close_time=estimate.kalshi_close_time,
        caveats=list(estimate.caveats),
    )


def append_paper_bet(bet: PaperBet, path: pathlib.Path | str = DEFAULT_PAPER_LOG_PATH) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(bet.to_json_line() + "\n")


def load_paper_bets(path: pathlib.Path | str = DEFAULT_PAPER_LOG_PATH) -> list[PaperBet]:
    path = pathlib.Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [PaperBet.from_json_line(line) for line in f if line.strip()]


def resolve_pending_paper_bets(
    fetch_market_fn,
    path: pathlib.Path | str = DEFAULT_PAPER_LOG_PATH,
) -> list[PaperBet]:
    """Resolve every `"pending"` paper bet in `path` against the real Kalshi
    market's own real, actual settlement result -- never against this
    project's own forecast or spot series, which would be grading this
    project's own homework rather than checking against the real,
    independent outcome the bet was actually about.

    `fetch_market_fn` is a one-argument callable `(ticker) -> KalshiMarketQuote`
    (pass `kalshi_client.fetch_market` for real use; a stub in tests, same
    pattern `track_record.resolve_pending_predictions` already uses for
    `series`). A bet is only resolved once that market's own `status` is
    `"finalized"` (or any status with a non-empty, real `result` field) --
    left pending, unchanged, otherwise. An already-resolved bet is never
    re-touched. Rewrites the file only if at least one bet actually changed.
    """
    bets = load_paper_bets(path)
    changed = False
    for bet in bets:
        if bet.outcome != "pending":
            continue
        quote = fetch_market_fn(bet.kalshi_ticker)
        if not quote.result:
            continue  # not yet settled -- leave pending
        yes_won = quote.result.lower() == "yes"
        side_won = yes_won if bet.side == "yes" else not yes_won
        bet.outcome = "won" if side_won else "lost"
        bet.resolved_kalshi_result = quote.result
        bet.resolved_at_utc = _now_utc_iso()
        changed = True

    if changed:
        path = pathlib.Path(path)
        with open(path, "w") as f:
            for bet in bets:
                f.write(bet.to_json_line() + "\n")

    return bets
