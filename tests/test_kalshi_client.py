import json
import pathlib

import pytest

from commodity_forecast.kalshi_client import (
    KalshiAPIError,
    KalshiMarketQuote,
    _parse_market,
    fetch_market,
    fetch_markets,
    fetch_series,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return json.load(f)


def test_parse_market_matches_a_real_captured_kxwti_quote():
    """Parses one real market object exactly as Kalshi's live production API
    actually returned it on 2026-10-16 (captured verbatim into
    tests/fixtures/ -- not hand-written), so this test fails if the parsing
    logic ever silently drifts from the real response shape."""
    raw = _load_fixture("kxwti_markets_live_2026-10-16.json")
    first = raw["markets"][0]
    quote = _parse_market(first)

    assert quote.ticker == "KXWTI-26OCT1914-T99.99"
    assert quote.event_ticker == "KXWTI-26OCT1914"
    assert quote.strike_type == "greater"
    assert quote.floor_strike == pytest.approx(99.99)
    assert quote.close_time == "2026-10-19T18:30:00Z"
    assert quote.yes_bid == pytest.approx(0.05)
    assert quote.yes_ask == pytest.approx(0.09)
    assert quote.no_bid == pytest.approx(0.91)
    assert quote.no_ask == pytest.approx(0.95)
    # This specific real market had zero recorded trading activity when captured.
    assert quote.volume == pytest.approx(0.0)
    assert quote.open_interest == pytest.approx(0.0)
    assert quote.is_actively_traded is False
    assert quote.result == ""


def test_real_captured_kxwti_ladder_is_internally_consistent():
    """Sanity-checks the *whole* real captured ladder (30 real strikes), not
    just one market: yes_ask should be non-increasing as the strike rises
    (a higher bar to clear is never more likely), and yes+no should always
    price out close to $1 plus the real bid-ask spread (never wildly off,
    which would indicate a parsing bug rather than a real market quote)."""
    raw = _load_fixture("kxwti_markets_live_2026-10-16.json")
    quotes = [_parse_market(m) for m in raw["markets"]]
    quotes_by_strike = sorted((q for q in quotes if q.floor_strike is not None), key=lambda q: q.floor_strike)

    asks = [q.yes_ask for q in quotes_by_strike]
    # Real market noise aside, the ladder should be overwhelmingly monotone
    # non-increasing -- allow a small amount of slack for real bid-ask-driven
    # non-monotonicity between neighboring strikes rather than asserting a
    # mathematically perfect staircase.
    violations = sum(1 for a, b in zip(asks, asks[1:]) if b > a + 0.01)
    assert violations == 0, "yes_ask should not meaningfully increase as the strike rises"

    for q in quotes_by_strike:
        # yes_bid + no_ask should be ~1.00 and yes_ask + no_bid should be ~1.00
        # (the two real, independent sides of the same real spread).
        assert q.yes_bid + q.no_ask == pytest.approx(1.0, abs=0.01)
        assert q.yes_ask + q.no_bid == pytest.approx(1.0, abs=0.01)


def test_fetch_series_parses_real_settlement_source(monkeypatch):
    """Confirms, from a real captured response, that KXWTI's own series
    metadata states its real settlement source is ICE (futures) -- the
    actual evidence behind edge.py's documented spot-vs-futures caveat, not
    an assumption."""
    raw = _load_fixture("kxwti_series_live_2026-10-16.json")

    def fake_get_json(path, **kwargs):
        assert path == "/series/KXWTI"
        return raw

    monkeypatch.setattr("commodity_forecast.kalshi_client._get_json", fake_get_json)
    series = fetch_series("KXWTI")
    assert series["ticker"] == "KXWTI"
    assert series["settlement_sources"][0]["name"] == "ICE"


def test_fetch_markets_parses_a_full_real_page_and_stops_without_a_cursor(monkeypatch):
    raw = _load_fixture("kxwti_markets_live_2026-10-16.json")
    calls = []

    def fake_get_json(path, params=None, **kwargs):
        calls.append(params)
        # Real fixture happens to carry a cursor (more real pages existed) --
        # force this test's fixture-backed page to look like the last page,
        # so this test only exercises "stop when there's no more data",
        # covered honestly, not stitched into an artificial second fixture.
        payload = dict(raw)
        payload.pop("cursor", None)
        return payload

    monkeypatch.setattr("commodity_forecast.kalshi_client._get_json", fake_get_json)
    quotes = fetch_markets("KXWTI", status="open", limit=30)

    assert len(calls) == 1
    assert calls[0]["series_ticker"] == "KXWTI"
    assert calls[0]["status"] == "open"
    assert len(quotes) == len(raw["markets"])
    assert all(isinstance(q, KalshiMarketQuote) for q in quotes)


def test_fetch_market_single_ticker_parses_real_shape(monkeypatch):
    raw = _load_fixture("kxwti_markets_live_2026-10-16.json")
    first_raw = raw["markets"][0]

    def fake_get_json(path, **kwargs):
        assert path == f"/markets/{first_raw['ticker']}"
        return {"market": first_raw}

    monkeypatch.setattr("commodity_forecast.kalshi_client._get_json", fake_get_json)
    quote = fetch_market(first_raw["ticker"])
    assert quote.ticker == first_raw["ticker"]
    assert quote.floor_strike == pytest.approx(99.99)


def test_get_json_retries_on_429_then_succeeds(monkeypatch):
    """Confirms the retry/backoff logic actually fires on a real HTTP 429
    (confirmed live, while building this module, that Kalshi's real API
    does return 429 under real rate-limiting) rather than failing
    immediately -- with sleep patched out so the test itself stays fast."""
    import urllib.error

    from commodity_forecast import kalshi_client

    attempts = {"n": 0}
    real_urlopen = kalshi_client.urllib.request.urlopen

    class _FakeResp:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "too many requests", {}, None)
        return _FakeResp(json.dumps({"series": {"ticker": "KXWTI"}}).encode())

    monkeypatch.setattr(kalshi_client.time, "sleep", lambda *_: None)
    monkeypatch.setattr(kalshi_client.urllib.request, "urlopen", fake_urlopen)

    result = fetch_series("KXWTI", max_retries=5, initial_backoff=0.01)
    assert result["ticker"] == "KXWTI"
    assert attempts["n"] == 3


def test_get_json_raises_kalshi_api_error_after_exhausting_retries(monkeypatch):
    import urllib.error

    from commodity_forecast import kalshi_client

    def always_429(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "too many requests", {}, None)

    monkeypatch.setattr(kalshi_client.time, "sleep", lambda *_: None)
    monkeypatch.setattr(kalshi_client.urllib.request, "urlopen", always_429)

    with pytest.raises(KalshiAPIError):
        fetch_series("KXWTI", max_retries=3, initial_backoff=0.01)


def test_fetch_series_real_live_kxwti_settlement_source():
    """Integration test: actually hits the real, live Kalshi public API --
    no mocking, consistent with this project's and conformal-forecast's own
    testing discipline of exercising the real thing (see test_data.py's own
    live EIA test, `test_fetch_wti_daily_raw_returns_real_recent_data`).
    Confirms the real, current settlement source for the real KXWTI series
    -- the live basis for edge.py's own documented caveat. Uses the
    client's own real retry/backoff (this project independently confirmed
    Kalshi's real API does return a real, retryable HTTP 429 under load
    while building this module), so a transient rate-limit does not fail
    this test outright.
    """
    series = fetch_series("KXWTI", max_retries=6, initial_backoff=3.0)
    assert series["ticker"] == "KXWTI"
    assert series["settlement_sources"][0]["name"] == "ICE"


def test_parse_market_captures_cap_strike_for_less_and_between_markets():
    """v0.10.0: `cap_strike` (added for the real `KXHIGH<STATION>`
    temperature ladder, which -- unlike `KXWTI`'s own real ladder -- uses
    `"less"` and `"between"` strike shapes, not just `"greater"`) against a
    real captured `KXHIGHNY` response (tests/fixtures/
    kxhighny_markets_live_2026-10-22.json, a byte-for-byte real API
    capture, same discipline as the existing `KXWTI` fixture test above)."""
    raw = _load_fixture("kxhighny_markets_live_2026-10-22.json")
    quotes = {m["ticker"]: _parse_market(m) for m in raw["markets"]}

    greater = quotes["KXHIGHNY-26OCT23-T64"]
    assert greater.strike_type == "greater"
    assert greater.floor_strike == pytest.approx(64.0)
    assert greater.cap_strike is None

    less = quotes["KXHIGHNY-26OCT23-T57"]
    assert less.strike_type == "less"
    assert less.floor_strike is None
    assert less.cap_strike == pytest.approx(57.0)

    between = quotes["KXHIGHNY-26OCT23-B63.5"]
    assert between.strike_type == "between"
    assert between.floor_strike == pytest.approx(63.0)
    assert between.cap_strike == pytest.approx(64.0)


def test_kalshi_market_quote_defaults_cap_strike_to_none_for_existing_call_sites():
    """Every pre-v0.10.0 call site (every WTI test/fixture) constructs
    `KalshiMarketQuote` without `cap_strike` -- confirms the new field's
    default keeps those unaffected."""
    quote = KalshiMarketQuote(
        ticker="T",
        event_ticker="E",
        title="t",
        status="active",
        strike_type="greater",
        floor_strike=100.0,
        close_time="2026-11-01T00:00:00Z",
        yes_bid=0.4,
        yes_ask=0.6,
        no_bid=0.4,
        no_ask=0.6,
        volume=0.0,
        open_interest=0.0,
        result="",
    )
    assert quote.cap_strike is None


def test_parse_market_captures_expiration_value_for_a_real_settled_crude_storage_market():
    """v0.11.0: `expiration_value` (added for the real `KXEIACRUDEW` crude-
    storage family) against a real captured settled response (tests/
    fixtures/kxeiacrudew_markets_settled_live_2026-10-22.json, a byte-for-
    byte real API capture, same discipline as the existing `KXWTI`/
    `KXHIGHNY` fixture tests above)."""
    raw = _load_fixture("kxeiacrudew_markets_settled_live_2026-10-22.json")
    quotes = {m["ticker"]: _parse_market(m) for m in raw["markets"]}

    q = quotes["KXEIACRUDEW-26OCT21-T422"]
    assert q.expiration_value == "412.5M"
    assert q.result == "no"


def test_kalshi_market_quote_defaults_expiration_value_to_none_for_existing_call_sites():
    """Every pre-v0.11.0 call site (every WTI/temperature test or fixture)
    constructs `KalshiMarketQuote` without `expiration_value` -- confirms
    the new field's default keeps those unaffected."""
    quote = KalshiMarketQuote(
        ticker="T",
        event_ticker="E",
        title="t",
        status="active",
        strike_type="greater",
        floor_strike=100.0,
        close_time="2026-11-01T00:00:00Z",
        yes_bid=0.4,
        yes_ask=0.6,
        no_bid=0.4,
        no_ask=0.6,
        volume=0.0,
        open_interest=0.0,
        result="",
    )
    assert quote.expiration_value is None
