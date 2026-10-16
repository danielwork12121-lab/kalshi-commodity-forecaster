"""Real, public, unauthenticated Kalshi market-data client.

Kalshi's production REST API exposes genuine read-only market data --
current yes/no bid-ask prices (which, for a Kalshi binary contract, are
themselves market-implied probabilities: a $1-notional "YES" contract priced
at $0.12 means the market is pricing the event at ~12%) -- for every public
market, with **no API key, login, or trading account required**. This was
confirmed two ways while building this module:

1. Kalshi's own docs (https://docs.kalshi.com/getting_started/quick_start_market_data)
   state plainly: "No authentication headers are required for the endpoints
   in this guide" -- covering `/series/{ticker}`, `/markets`, `/events/{ticker}`,
   and `/markets/{ticker}/orderbook`.
2. Independently verified live, directly against the real production API
   (`https://api.elections.kalshi.com/trade-api/v2`) on 2026-10-16: a plain
   unauthenticated GET against `/series/KXWTI` and `/markets?series_ticker=KXWTI`
   returned real, current market data (see `tests/fixtures/kxwti_markets_live_2026-10-16.json`,
   a byte-for-byte capture of one such real response, used to keep the unit
   tests below deterministic/offline without ever faking the shape of a real
   payload).

This module is deliberately generic (any Kalshi series ticker, not just
WTI) -- `edge.py` is what specializes it to this project's own WTI
forecaster.

Honest, load-bearing limitation discovered while building this (see
`edge.py`'s own module docstring and the README's new "Edge vs. real market
price" section for the full discussion, and `active-project-kalshi.md`'s
"Edge/profitability narrative" section for how this changes the plan):
Kalshi's `KXWTI` series settles against the **ICE WTI Crude Futures
front-month contract's own daily settlement price** (see a series' own
`settlement_sources` field), not the EIA Cushing, OK *spot* price this
project's `data.py`/`forecast.py` forecast. These two real series are
correlated but **not the same number** -- confirmed directly on 2026-10-16,
when this project's own cached EIA spot series showed WTI at $101.91/bbl
(2026-10-14) while the live `KXWTI` Oct-19 ladder's own 50%-crossing strike
sat around $95.50-96.00, a gap far too large to be explained by three days
of forecast uncertainty alone. Comparing this project's spot-based
probability directly against this market's probability without correcting
for that basis would overstate any computed "edge" and would not be honest.
`edge.py` surfaces this as an explicit caveat on every `EdgeEstimate` rather
than silently assuming the two prices are interchangeable.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiAPIError(RuntimeError):
    """Raised when the real Kalshi public API can't be reached or parsed."""


@dataclass
class KalshiMarketQuote:
    """One real market's current quote, as reported by Kalshi's own public API.

    Prices are already in dollars (Kalshi's `_dollars` fields), i.e.
    probabilities in [0, 1] for a $1-notional binary contract -- `$0.12` on
    `yes_ask_dollars` means "pay $0.12 now, receive $1.00 if YES, $0.00 if
    NO", which is exactly a market-implied probability of ~12% once you
    also account for the bid-ask spread (see `edge.py`'s
    `market_implied_probability`).
    """

    ticker: str
    event_ticker: str
    title: str
    status: str
    strike_type: str  # e.g. "greater" for "above $X"
    floor_strike: float | None
    close_time: str  # ISO-8601, as reported by Kalshi -- this market's own real deadline
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume: float  # volume_fp -- real lifetime contract volume traded, 0.0 if never traded
    open_interest: float  # open_interest_fp -- real open contracts outstanding, 0.0 if none
    result: str  # "" while open; the real settled outcome once status == "finalized"

    @property
    def is_actively_traded(self) -> bool:
        """True only if this market has real recorded trading activity.

        A freshly-listed market (this project found real examples of this on
        2026-10-16: a just-opened `KXWTI` daily ladder with `volume_fp` and
        `open_interest_fp` both `0.00` on every strike) shows a maker-seeded
        bid-ask spread, not a real trading consensus -- treating that as a
        trustworthy "market-implied probability" would be its own honesty
        problem, separate from the spot-vs-futures basis issue above. Callers
        computing an edge should check this (or inspect `volume`/
        `open_interest` directly) before trusting the resulting number.
        """
        return self.volume > 0 or self.open_interest > 0


def _parse_market(raw: dict) -> KalshiMarketQuote:
    return KalshiMarketQuote(
        ticker=raw["ticker"],
        event_ticker=raw.get("event_ticker", ""),
        title=raw.get("title", ""),
        status=raw.get("status", ""),
        strike_type=raw.get("strike_type", ""),
        floor_strike=(float(raw["floor_strike"]) if raw.get("floor_strike") is not None else None),
        close_time=raw.get("close_time", ""),
        yes_bid=float(raw.get("yes_bid_dollars", 0.0)),
        yes_ask=float(raw.get("yes_ask_dollars", 0.0)),
        no_bid=float(raw.get("no_bid_dollars", 0.0)),
        no_ask=float(raw.get("no_ask_dollars", 0.0)),
        volume=float(raw.get("volume_fp", 0.0)),
        open_interest=float(raw.get("open_interest_fp", 0.0)),
        result=raw.get("result", ""),
    )


def _get_json(
    path: str,
    params: dict | None = None,
    base_url: str = KALSHI_BASE_URL,
    timeout: float = 15.0,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
) -> dict:
    """GET one real, public, unauthenticated Kalshi API endpoint.

    Retries on HTTP 429 ("too many requests" -- confirmed hit live while
    building this module, this is a real behavior of the real API, not a
    hypothetical) and on 5xx, with exponential backoff -- a courteous real
    client backs off rather than hammering a public API harder after being
    told to slow down. Raises `KalshiAPIError` (never silently returns stale
    or partial data) if every retry is exhausted or a non-retryable error
    occurs.
    """
    query = ""
    if params:
        parts = [f"{k}={urllib.request.quote(str(v))}" for k, v in params.items() if v is not None]
        if parts:
            query = "?" + "&".join(parts)
    url = f"{base_url}{path}{query}"

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, headers={"Accept": "application/json", "User-Agent": "commodity-forecast/0.1 (research)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429 or 500 <= exc.code < 600:
                if attempt < max_retries - 1:
                    time.sleep(initial_backoff * (2**attempt))
                    continue
            raise KalshiAPIError(f"GET {url} failed: HTTP {exc.code} {exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001 - wrap any network failure honestly
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(initial_backoff * (2**attempt))
                continue
            raise KalshiAPIError(f"GET {url} failed: {exc}") from exc

    raise KalshiAPIError(f"GET {url} failed after {max_retries} attempts: {last_exc}")


def fetch_series(series_ticker: str, **kwargs) -> dict:
    """Return the real series-level metadata for one Kalshi series (e.g.
    settlement source, category) -- used by `edge.py` to surface which real
    benchmark a series actually settles against."""
    return _get_json(f"/series/{series_ticker}", **kwargs)["series"]


def fetch_markets(
    series_ticker: str,
    status: str = "open",
    limit: int = 100,
    max_pages: int = 10,
    **kwargs,
) -> list[KalshiMarketQuote]:
    """Return every real market in one Kalshi series matching `status`
    ("open", "closed", "settled", or "all"), paginating through Kalshi's own
    cursor-based pagination until exhausted or `max_pages` is hit.
    """
    markets: list[KalshiMarketQuote] = []
    cursor = None
    for _ in range(max_pages):
        params = {"series_ticker": series_ticker, "limit": limit}
        if status != "all":
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        payload = _get_json("/markets", params=params, **kwargs)
        markets.extend(_parse_market(m) for m in payload.get("markets", []))
        cursor = payload.get("cursor") or None
        if not cursor:
            break
    return markets


def fetch_market(ticker: str, **kwargs) -> KalshiMarketQuote:
    """Return the real, current quote for one specific market ticker."""
    payload = _get_json(f"/markets/{ticker}", **kwargs)
    return _parse_market(payload["market"])
