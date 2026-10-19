# commodity-forecast

Calibrated commodity-price forecasting, built as a real-world application on
top of [`conformal-forecast`](https://github.com/danielwork12121-lab/conformal-forecast)
— a from-scratch PyTorch + split-conformal-prediction library. This project
depends on `conformal-forecast` as an actual pip dependency (not copy-pasted
code) and reuses its calibrated forecasters — a point forecast plus a
prediction interval with a *verified* coverage guarantee — on real commodity
market data, with the long-run goal of turning that into probability
estimates for Kalshi-style threshold questions ("will WTI close above $X by
date Y?").

## Not a trading tool — read this first

This project **never touches any real Kalshi account and never places, or
stages, an order.** It is a public forecasting/analysis tool only: it
produces calibrated probability estimates for commodity-linked questions and
publishes the code and backtest results. It gives no financial advice. (A
separate, unrelated, currently-disabled automation under a different project
researches and stages real Kalshi trades on the author's own account — this
repository has no connection to that one.) This includes `edge.py` (added
2026-10-16, see "Edge vs. a real market price" below): it reads real, public
Kalshi market data and sizes/logs **paper-only, simulated** bets for
analysis purposes — no real order, position, or account is ever involved.

## No-fabrication commitment

Every result in this repository is real and honestly labeled:

- **Backtests are backtests.** Any historical accuracy/coverage number here
  is computed after the fact against historical data, and is always labeled
  as a backtest — never presented as a live prediction made at the time.
- **No fabricated or backdated history.** The commit history and the forward
  track record below (`predictions/`) build up for real, one dated entry at
  a time, starting 2026-10-14. Nothing here is backdated or invented to
  manufacture the appearance of a longer track record than actually exists.
- **Gaps are reported, not hidden.** See the coverage gap in the first real
  result below — it's real, it's measured, and it's discussed rather than
  smoothed over.

## Architecture

```
commodity_forecast/
  data.py       real EIA data ingestion (no API key) + local CSV cache
  forecast.py   wraps conformal-forecast's prepare_split_from_series + SplitConformalForecaster
                (static calibration, naive baseline or -- since 2026-10-15 --
                an LSTM via fit_calibrated_lstm_forecaster) and
                AdaptiveConformalForecaster / SlidingWindowAdaptiveConformalForecaster (ACI);
                plus (since 2026-10-17) prepare_horizon_split /
                fit_calibrated_naive_forecaster_horizon -- a genuine
                multi-step-ahead forecaster, composing conformal-forecast's
                own make_windows(horizon=...) a second way rather than
                modifying that shared library
  probability.py  converts a calibrated forecast into a Kalshi-style
                  threshold probability P(price > K), backtested against
                  real history with both a static and an online-adapting
                  residual pool (since 2026-10-15), plus (since 2026-10-15)
                  select_adaptive_pool_window: a rolling-origin CV
                  selection for the adaptive pool's own window, mirroring
                  conformal-forecast's auto-window/cv-window-select
  track_record.py makes + persists real, dated, forward (non-backtested)
                   predictions, and resolves them once real outcomes exist
                   (since 2026-10-15)
  kalshi_client.py (since 2026-10-16) real, public, unauthenticated Kalshi
                   market-data client (series/markets/orderbook), with real
                   retry/backoff on the real rate limiting this project
                   independently confirmed while building it
  edge.py          (since 2026-10-16) compares this project's own
                   calibrated probability against a real, live Kalshi
                   market's own implied probability, with explicit,
                   structured honesty caveats (settlement-benchmark
                   mismatch, illiquidity, forecast-horizon mismatch --
                   see "Edge vs. a real market price" below) attached to
                   every estimate, plus conservative fractional-Kelly
                   paper-bet sizing and a real, forward, non-backdated
                   paper track record -- never real money, never a real
                   order; plus (since 2026-10-17)
                   VALIDATED_FORECAST_HORIZONS_DAYS / pick_validated_horizon,
                   matching a real market's own days_to_close to one of
                   forecast.py's actually-validated horizons so the
                   forecast-horizon-mismatch caveat no longer fires by
                   default; plus (since 2026-10-19) a `notes` field on
                   `EdgeEstimate` (informational, never a caveat) and
                   `days_to_close_from_last_observation`, fixing a real
                   staleness bug in horizon-matching -- see "Fixing a
                   staleness bug in horizon-matching" below
  benchmark.py     (since 2026-10-19) recovers a real, measured estimate
                   of the ICE WTI futures settlement price Kalshi itself
                   never publishes directly, from the real resolved
                   "greater than $X" ladders in KXWTI's own settlement
                   history (no external/paid futures-data vendor needed),
                   and uses it to measure -- and, when trustworthy,
                   correct for -- the real spot/futures basis against
                   this project's own EIA spot series; see "Resolving
                   the settlement-benchmark mismatch" below
  edge_survey.py   (since 2026-10-19) survey_live_edges + summarize_edge_subsets
                   -- runs the real edge.py pipeline across every real
                   open market on a series in one call, instead of one
                   market at a time, and breaks the real result down by
                   real subsets (strike distance, horizon, liquidity) to
                   honestly check whether any slice of the ladder looks
                   different from the ladder-wide average; see "An
                   edge-subset survey" below
  edge_persistence.py (since 2026-10-19, continued) append_survey_snapshot
                   / load_survey_history / summarize_persistence -- records
                   each real edge_survey.py run to data/edge_subset_history.jsonl
                   and reports, once enough real snapshots exist, whether a
                   subset's edge sign has actually held up over time rather
                   than being single-snapshot noise; see "Tracking
                   persistence across real snapshots" below
tests/
  test_data.py         date-math unit test + live EIA integration test + cache-hit test
  test_forecast.py     synthetic-data + real-WTI-series end-to-end tests, static and ACI
  test_forecast_lstm.py synthetic + real-WTI tests for the LSTM point model (since 2026-10-15)
  test_probability.py  hand-computed sanity checks + a real-WTI-series backtest
  test_adaptive_pool_window_selection.py fold-boundary helpers + real-data CV window
                        selection checks (since 2026-10-15)
  test_track_record.py forward-prediction structure checks + synthetic-future resolution checks
  test_kalshi_client.py (since 2026-10-16) parses real captured Kalshi API
                        responses (tests/fixtures/), checks real 429
                        retry/backoff, plus one live integration test
                        against the real production API
  test_edge.py          (since 2026-10-16) hand-computed Kelly-sizing
                        checks, honesty-caveat checks, paper-bet
                        append/load/resolve round-trips, and (since
                        2026-10-17) pick_validated_horizon rounding checks
  test_forecast_horizon.py (since 2026-10-17) confirms horizon=1 reduces
                        exactly to the existing one-step pipeline, real-data
                        checks at every validated horizon, a real-data
                        monotonic-MAE check, and that probability.py's
                        existing backtest works unchanged on a horizon result
  test_benchmark.py     (since 2026-10-19) bracket-recovery checks against a
                        real captured slice of KXWTI's own settlement
                        history, real basis-measurement/recent-window-
                        estimate checks, and basis-correction trust checks
  test_edge_survey.py   (since 2026-10-19) pure-unit subset-bucketing checks
                        plus a real end-to-end run of survey_live_edges
                        against a real captured, currently-open KXWTI
                        ladder and a real settlement-history fixture --
                        confirms nothing is silently skipped/scored wrong
  test_edge_persistence.py (since 2026-10-19, continued) append/load
                        roundtrip checks, real-JSONL-file checks, and the
                        "not enough data yet" / sign-consistent /
                        sign-flipped persistence-verdict logic
data/
  wti_spot_daily.csv real cached WTI daily spot price series, 1986-01-02 to present
  edge_subset_history.jsonl (since 2026-10-19, continued) the real,
                     append-only, forward-only history of edge-subset
                     survey snapshots -- one real JSON line per real
                     survey run, never backdated or synthetic
predictions/
  README.md         explains the forward track record's format and resolution rule
  track_record.jsonl the real, dated predictions themselves (first batch made and
                      resolved 2026-10-15)
  paper_bets.jsonl  (since 2026-10-16) real, timestamped, paper-only
                     (never executed) simulated bets sized by edge.py,
                     resolved later against each market's own real
                     settlement result
```

### Data source

Real public data: the U.S. Energy Information Administration's own daily
spot-price series for WTI crude oil at Cushing, OK (EIA series `RWTC`),
fetched directly from EIA's own spreadsheet download
(`https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls`) — the same file a human
gets from EIA's own series page. No API key or registration required. (EIA's
v2 JSON API serves the same series but requires a registered key — its
shared, unregistered `DEMO_KEY` hits a rate limit almost immediately under
real use, confirmed directly while building this. If a registered key is
ever added, the v2 API would unlock other series, e.g. production and
inventories, this spreadsheet alone doesn't carry.)

### Forecasting

`forecast.py` depends on `conformal-forecast`'s public
`prepare_split_from_series` (its own chronological split / sliding-window /
z-score-normalize pipeline — the exact pipeline `conformal-forecast` itself
validates its coverage guarantees against), its `SplitConformalForecaster`
+ `NaiveForecaster` baseline (static calibration, `fit_calibrated_naive_forecaster`),
and — as of this run — its `AdaptiveConformalForecaster` /
`SlidingWindowAdaptiveConformalForecaster` (Adaptive Conformal Inference,
`fit_adaptive_naive_forecaster`). No splitting, windowing, normalization, or
calibration logic is reimplemented here — this project only ever calls into
`conformal-forecast`'s own, already-tested implementations.

## First real result (measured honestly — a backtest, not a live prediction)

| | |
|---|---|
| Series | WTI daily spot price, 9,517 points, 1986-01-02 to 2026-10-06 |
| Model | `NaiveForecaster` (today predicts tomorrow), lookback=5 |
| Train / calibration / test windows | 5,705 / 1,903 / 1,904 |
| MAE | $1.43/bbl |
| Nominal coverage | 90% |
| **Empirical coverage** | **80.9%** (a real, reported gap — not hidden) |
| Mean interval width | $4.18/bbl |

The coverage gap is real and expected: split conformal's guarantee assumes
calibration/test residual exchangeability, which a non-stationary real
commodity series (this one includes the real 2020 negative-price event,
-$36.98/bbl) only approximately satisfies — the same limitation
`conformal-forecast`'s own README documents for its own LSTM/airline case.

*Correction (2026-10-15): the mean interval width above was originally
reported as $0.16/bbl — that was actually the width in normalized
(z-score) units (`result.diagnostics["mean_interval_width"]`), never
multiplied through by `result.normalizer_std` to convert it to real
dollars the way the MAE row already was. The real, denormalized width is
$4.18/bbl (`0.1582 (normalized) * 26.428 (this series' std, $/bbl) =
$4.18`), confirmed directly against `result.normalizer_std` on the real
pushed code. Every mean-interval-width figure in the ACI comparison
table below is corrected the same way; nothing else in either table was
affected (coverage and MAE were already computed/reported correctly).

## Does Adaptive Conformal Inference close the gap? (measured honestly, this run)

Same series, same `NaiveForecaster` baseline, same lookback/alpha/split as
above — only the calibration method changes, so this is an apples-to-apples
test of whether `conformal-forecast`'s existing ACI implementations
(`fit_adaptive_naive_forecaster`, new this run) help on real commodity data
the way they already help on `conformal-forecast`'s own airline benchmark.

| Method | Empirical coverage | Gap vs. 90% nominal | Mean interval width |
|---|---|---|---|
| Static split conformal (baseline, above) | 80.9% | **-9.1pp** | $4.18/bbl |
| ACI, fixed calibration pool | 87.4% | -2.6pp | $5.49/bbl |
| ACI, sliding pool (unbounded/growing) | 88.4% | -1.6pp | $7.13/bbl |
| ACI, sliding pool, `window=75` | **89.8%** | **-0.2pp** | $6.42/bbl |

**Yes, substantially — a bounded sliding-window pool nearly closes it.**
Both ACI variants meaningfully reduce the gap over static conformal, and a
sliding pool bounded to a few dozen-to-few-hundred recent residuals
(swept at `window` in `{20, 30, 50, 75, 100, 150, 200, 300}`; every value in
that range lands at 89.5-89.8% empirical coverage, within real measurement
noise of each other — `window=75` is reported above as one concrete,
reproducible point, not asserted as the uniquely optimal value) gets within
roughly 0.2-0.5 percentage points of the 90% nominal target, down from
static conformal's 9.1-point miss. This matches the mechanism
`conformal-forecast`'s own README describes: a *fixed* calibration pool caps
how wide ACI's interval can ever grow at the pool's own largest observed
residual, and WTI's calibration window here is a tiny ~5-point lookback
slice of a 9,500+ point, highly non-stationary series — exactly the
situation where that ceiling binds hardest. Letting the pool itself update
online (sliding variant) removes the ceiling; bounding the window (rather
than an unbounded growing buffer) additionally lets the pool forget residuals
from early, less-representative stretches of a 40-year series, which is
likely why a bounded window outperforms the unbounded growing buffer here.
The interval does get wider to earn that coverage ($4.18 → $6.42/bbl mean
width, window=75) — an honest cost, not a free lunch: a correctly-calibrated
90% interval on this series is genuinely closer to $6.42/bbl wide than
$4.18, and static conformal's narrower interval was narrower because it was
systematically under-covering, not because it was more accurate.

Verified directly, not assumed: `fit_adaptive_naive_forecaster` (in
`forecast.py`) wraps `conformal-forecast`'s `AdaptiveConformalForecaster`
(`variant="fixed"`) and `SlidingWindowAdaptiveConformalForecaster`
(`variant="sliding"`, with an optional bounded `window`), reusing the exact
same `prepare_split_from_series` split as the static baseline so the
comparison is calibration-method-only. 7 new tests (5 → 12): an
unknown-variant rejection, both variants sanity-checked on synthetic data
and on the real full WTI series (with a `>0.85` empirical-coverage
regression check against the real, measured static-baseline gap above — a
loose threshold, not an exact pin, so a legitimate future refresh of the
committed CSV cache with new real data doesn't spuriously fail it), a
bounded-`window` sanity check, and a pinned check that `window=75`
specifically clears 89%. Full suite: **12/12 passed.**

## From a calibrated interval to a Kalshi-style probability (2026-10-15)

Everything above answers "is the true price in this interval, with ~90%
probability?" A Kalshi-style market asks a sharper question: P(price > K)
for a specific strike K, as one number — which the interval alone doesn't
give. `commodity_forecast/probability.py` converts one into the other by
treating the model's own held-out calibration-split residuals (real
forecast errors on data the model never trained or calibrated on, already
computed as part of split conformal's own machinery, previously thrown
away as nothing but an absolute-value quantile) as an empirical sample of
"the forecast error distribution": P(y > K) ≈ the fraction of those real
residuals that would have pushed today's point forecast past K. No
assumed shape (no "assume Gaussian errors"), no new hyperparameters, and —
the part that actually matters — it's backtestable the same honest way
coverage was: ask the same question over and over across real history and
check whether events predicted at ~p% actually happened about p% of the
time.

**Backtest, on the real WTI test split (same split as the tables above),
asking a forecast-relative question for every test day** (e.g. "will the
price be more than 5% above today's own point forecast?", at offsets
`{-10%, -5%, -2%, 0%, +2%, +5%, +10%}` — 1,904 test days × 7 offsets =
13,328 real, independent questions):

| Metric | Value | Context |
|---|---|---|
| Brier score | **0.0906** | 0.25 = no skill (always guessing 50%); 0 = perfect |
| Mean calibration error | **0.026** | count-weighted mean \|predicted − actual\| across probability-decile bins |

Reliability by predicted-probability decile (count, mean predicted vs. the
real, realized exceedance rate in that bucket):

| Predicted range | Count | Mean predicted | Actual rate |
|---|---|---|---|
| 0.0–0.1 | 4,541 | 1.8% | 4.6% |
| 0.1–0.2 | 965 | 13.1% | 17.0% |
| 0.2–0.3 | 164 | 23.1% | 29.9% |
| 0.3–0.4 | 37 | 34.6% | 40.5% |
| 0.4–0.5 | 5 | 42.3% | 80.0% |
| 0.5–0.6 | 1,905 | 51.4% | 53.1% |
| 0.6–0.7 | 42 | 65.7% | 59.5% |
| 0.7–0.8 | 195 | 76.0% | 81.0% |
| 0.8–0.9 | 1,244 | 86.5% | 85.4% |
| 0.9–1.0 | 4,230 | 98.4% | 95.9% |

Honestly read: the method is meaningfully better than guessing (a Brier
score of 0.09 vs. 0.25 for a coin flip) and most buckets track real
outcomes within a few points. It also has a real, disclosed weak spot: the
0.4–0.5 bucket (only 5 of 13,328 observations — a rare near-the-edge
offset/day combination) is badly off, 42% predicted vs. 80% actual, which
is a small-sample noise artifact rather than a sign the method
systematically fails near 50-50 calls (the neighboring 0.5–0.6 bucket, with
1,905 observations, is well-calibrated at 51.4% vs. 53.1%). This
limitation, and the underlying assumption it comes from (treating the
calibration-residual sample as the *entire* predictive distribution for
every test point, independent of forecast level or time), is stated
directly in `probability.py`'s own module docstring, not glossed over.

5 new tests (12 → 17): hand-computed sanity checks against a tiny known
residual sample (an exact P(y>K) calculation, monotonicity in the
threshold, correct broadcasting across multiple point forecasts), plus the
real-WTI backtest above re-run as a loose (`< 0.15` Brier, `< 0.08` mean
calibration error — real thresholds a future legitimate data refresh
shouldn't spuriously trip) regression check, and a custom-offsets test.
Full suite: **17/17 passed.**

## Does an online, adapting residual pool improve the probability estimate? (2026-10-15)

The backtest above treats the probability layer's residual sample as
*frozen* at the original calibration window — the same simplification
`probability.py`'s own module docstring names as the module's one real
honesty caveat, and the exact same one the interval layer had before ACI
(see above) let the residual pool itself adapt online. `commodity_forecast
/probability.py`'s new `backtest_threshold_calibration_adaptive` answers
the obvious follow-up: does the same idea — folding each day's own signed
residual into the pool, in strict step order, after it's scored — improve
the *probability* estimate the same way it measurably improved raw interval
coverage?

**Measured, not assumed, on the real WTI series, same 1,904-day test split
and the same 7 relative offsets as the static backtest (13,328 real
questions either way):**

| Method | Brier score | Mean calibration error |
|---|---|---|
| Static (frozen calibration pool) | 0.0906 | 0.0258 |
| Adaptive, unbounded growing pool | 0.0901 | 0.0197 |
| Adaptive, bounded window=75 | 0.0901 | 0.0138 |
| **Adaptive, bounded window=200** | **0.0894** | **0.0100** |
| Adaptive, bounded window=1000 | 0.0900 | 0.0146 |

**The honest result is narrower than "adaptive wins": the Brier score barely
moves** (0.0906 → ~0.089-0.090 regardless of window — essentially flat,
since Brier is dominated by the large, easy-to-call tail questions every
version already gets right) **but the mean calibration error — specifically
how well predicted probabilities track real outcomes bucket-by-bucket —
improves substantially**, roughly 2.5x better at the best bounded window
than the static version, and the improvement is *not* monotonic in window
size: an unbounded growing pool (0.0197) is clearly worse than any bounded
window tried in the 75-500 range, and windows much larger than that
(750, 1000) drift back up toward the unbounded pool's own number. This
mirrors the interval-ACI case's own finding (a bounded sliding window beats
an unbounded growing buffer) rather than just assuming "more adaptive is
always better" — measured with an actual sweep, not picked to look good:

| window | 50 | 75 | 100 | 150 | 200 | 300 | 500 | 750 | 1000 | unbounded |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean calibration error | 0.0145 | 0.0138 | 0.0133 | 0.0130 | **0.0100** | 0.0112 | 0.0113 | 0.0134 | 0.0146 | 0.0197 |

Reliability by predicted-probability decile at the best window (200),
compare against the static table above:

| Predicted range | Count | Mean predicted | Actual rate |
|---|---|---|---|
| 0.0–0.1 | 3,808 | 1.5% | 2.0% |
| 0.1–0.2 | 1,322 | 13.9% | 16.0% |
| 0.2–0.3 | 509 | 23.9% | 24.2% |
| 0.3–0.4 | 73 | 31.8% | 37.0% |
| 0.4–0.5 | 330 | 48.4% | 53.6% |
| 0.5–0.6 | 1,577 | 54.0% | 52.9% |
| 0.6–0.7 | 52 | 64.0% | 65.4% |
| 0.7–0.8 | 463 | 75.6% | 79.7% |
| 0.8–0.9 | 1,224 | 85.1% | 84.9% |
| 0.9–1.0 | 3,970 | 97.7% | 97.1% |

Every bucket here tracks its own real outcome rate more tightly than the
static version's table did (the static version's worst disclosed bucket,
0.4-0.5, was 42.3% predicted vs. 80.0% actual on only 5 observations — here
the same bucket has 330 observations and is 48.4% vs. 53.6%, a far smaller
and far better-sampled gap).

Implementation note: this does *not* reuse `forecasting.adaptive
.SlidingWindowAdaptiveConformalForecaster` from the conformal-forecast
dependency — that class only ever tracks *absolute* residuals internally
(enough for a symmetric interval), not the signed residuals an exceedance
probability needs to recover the empirical CDF's direction. The online pool
is reimplemented directly in `probability.py`, replaying the same strict
step-ordering discipline (score against the pool as it stood *before* this
step, fold this step's own residual in only afterward) that class itself
uses, so the "never peek at your own answer" guarantee carries over exactly
even though the code path is separate.

4 new tests (25 → 29): a fully hand-computed 2-step scenario pinning down
the exact step-ordering contract (and proving the adaptive and static
versions genuinely diverge, not just run through different code that
happens to agree), a bounded-window seed-truncation check (mirroring the
real bug `SlidingWindowAdaptiveConformalForecaster` itself once had, see
that class's own docstring), a mismatched-length error-handling check, and
the real-WTI backtest above re-run as a loose regression check. Full suite:
**29/29 passed.**

## Picking the adaptive pool's window honestly (2026-10-15)

The section above picked `window=200` by sweeping a handful of values and
reading off the best real-test-set number — exactly the "eyeballed, not
principled" selection `active-project.md` (conformal-forecast's own docs)
already flags as a pattern worth automating for its own ACI case, and this
project's own "What's next" item 3. `select_adaptive_pool_window`
(`commodity_forecast/probability.py`) is the fix: a rolling-origin
cross-validation over the calibration set (five expanding-window folds,
each candidate window scored by the mean calibration error of its own
online-pool backtest on that fold's validation chunk — the exact same
`_run_adaptive_pool_backtest` loop the real backtest above uses, just run
against a calibration-set fold instead of the real test set), never
touching the real test set to select — which is used only afterward, via
the already-tested `backtest_threshold_calibration_adaptive`, as an
honest post-hoc check.

**Measured, not assumed, on the real WTI series (5 folds, candidates
20/50/75/100/150/200/300/unbounded):**

| Window | Mean fold calibration error (used to select) | Mean fold Brier | Real test calibration error (post-hoc) | Real test Brier (post-hoc) |
|---|---|---|---|---|
| 20 | 0.0196 | 0.0799 | 0.0156 | 0.0906 |
| 50 | 0.0166 | 0.0777 | 0.0150 | 0.0898 |
| 75 | 0.0153 | 0.0776 | 0.0141 | 0.0901 |
| 100 | 0.0175 | 0.0774 | 0.0133 | 0.0898 |
| **150 (selected)** | **0.0146** | 0.0773 | **0.0136** | 0.0894 |
| 200 (best in hindsight) | 0.0165 | 0.0773 | 0.0101 | 0.0894 |
| 300 | 0.0205 | 0.0772 | 0.0126 | 0.0897 |
| unbounded | 0.0226 | 0.0776 | 0.0198 | 0.0901 |

*Reference, real test set, static (frozen) pool: calibration error 0.0261,
Brier 0.0905.*

**A genuine, principled win, not a tie or a loss this time:** the
fold-averaged metric correctly picks window=150 out of eight real
candidates — not the oracle's own window=200, but close to it (real
test-set calibration error 0.0136 vs. the oracle's 0.0101), and a
substantial, real improvement over both the static baseline (0.0261,
~1.9x worse) and every smaller/larger candidate the fold metric correctly
ranks below it. Unlike the EnbPI window-CV feature shipped the same day on
the conformal-forecast side (see that repo's own README — a real loss on
its smallest-calibration-set case), this project's own calibration set is
large enough (1,903 windows) that five real folds of ~270-370 points each
give the selection genuine signal rather than falling through to a tie-break.

16 new tests (33 → 49, `tests/test_adaptive_pool_window_selection.py`):
the new `_make_rolling_folds`/`_select_best_window` fold-boundary helpers
(pure, no model dependency — reimplemented here rather than importing
conformal-forecast's own private experiment-module versions, since this
project only depends on that repo's public API), real-data integration
checks (every candidate reported, folds partition the calibration set
with no gap or overlap, determinism for a fixed seed), a direct check that
shrinking the test set passed in changes nothing about which window gets
selected (proving the selection criterion never actually reads the test
set, not just trusting the docstring's claim), the too-small-calibration-set
and invalid-argument error paths, and a check that the honest post-hoc
result matches calling `backtest_threshold_calibration_adaptive` directly.
Full suite: **49/49 passed.**

## Does a stronger point model help? (2026-10-15)

Every result above uses `NaiveForecaster` ("tomorrow = today, in normalized
units") as the point model — deliberately, for the first slice, to prove
the real dependency + real data + real calibration pipeline held together
end to end before chasing accuracy (see "What's next" below, item 2, as it
read before this run). conformal-forecast's own from-scratch LSTM is
already a real dependency (it's what every ACI/EnbPI result in *that*
repo's own README is measured against), so trying it here costs no new
infrastructure — just a new `fit_calibrated_lstm_forecaster` function
(`commodity_forecast/forecast.py`) that wraps the LSTM in the same
`DeltaWrapper` (predict the *change*, not the level) convention
conformal-forecast's own experiment code uses for every neural model on
every dataset, and otherwise runs through the exact same split, lookback,
and alpha as the naive baseline — so any difference is attributable to the
model, not a different split.

The honest answer, measured on the real WTI series, same test split both
ways (1,904 points):

| Model | MAE | Coverage (static) | Coverage gap | Mean interval width | Brier score | Mean calibration error |
|---|---|---|---|---|---|---|
| Naive | $1.43/bbl | 80.9% | -9.1pp | $4.18/bbl | 0.0906 | 0.0258 |
| LSTM (delta) | $1.44/bbl | 80.6% | -9.4pp | $4.18/bbl | 0.0907 | 0.0262 |

**A stronger model does not measurably help here — and that's a real,
useful negative result, not a bug.** WTI daily spot price is close enough
to a random walk that `NaiveForecaster`'s implicit "predict zero change"
is already close to the best achievable one-day-ahead point forecast on
this series; a `DeltaWrapper`-wrapped LSTM trained to predict the same
delta converges to essentially the same thing, so every downstream number
(interval width, coverage, Brier score, calibration error) comes out
statistically indistinguishable from the naive baseline's. This mirrors a
well-known property of liquid commodity/financial daily price series
(near-random-walk behavior makes naive point forecasts hard to beat) rather
than indicating anything wrong with the LSTM, the calibration pipeline, or
the probability layer — all three visibly *do* work (the coverage gap and
Brier score both land in the same real, reasonable range as the naive
baseline's, not broken or degenerate).

**Confirms the probability layer is genuinely model-agnostic, not
coincidentally naive-shaped:** `backtest_threshold_calibration`/
`backtest_threshold_calibration_adaptive` (`commodity_forecast/probability.py`)
needed zero code changes to run against the LSTM-based
`CalibratedForecastResult` — both functions only ever read
`result.conformal_forecaster.{model,predict}`, `result.cal_residuals_normalized`,
and `result.normalizer_{mean,std}`, never anything `NaiveForecaster`-specific.
This was checked directly (`tests/test_forecast_lstm.py`'s
`test_lstm_calibrated_result_is_a_drop_in_for_the_probability_layer`), not
just inferred from reading the code.

4 new tests (29 → 33, `tests/test_forecast_lstm.py`): the LSTM forecaster
sanity-checked on synthetic data and the real WTI series (same bounds as
the naive baseline's own equivalent tests), confirmation that both
functions produce the identical split sizes (making the comparison above
apples-to-apples), and the probability-layer drop-in check above. Full
suite: **33/33 passed.**

## Forward track record (2026-10-15 — started)

Everything above (including the backtest just above) is retrospective:
every question it answers was already resolved by the time the
measurement looked at it. This section is the opposite kind of real — a
prediction made *before* the outcome is known, timestamped to prove it,
graded later only once the real outcome actually exists. See
`predictions/README.md` for the full format and the no-self-grading rule
(a future run, never this one, resolves these).

First batch, `commodity_forecast/track_record.py`'s `make_forward_prediction`,
using the real cached WTI series (last real observation: 2026-10-06,
$96.24/bbl) and the exact same calibration-residual machinery the backtest
above already validated:

| Made at (UTC) | Last observed | Question | Threshold | P(exceed) | Resolved against | Outcome |
|---|---|---|---|---|---|---|
| 2026-10-15 03:42:33 | 2026-10-06, $96.24 | Will the next real observed price be *above* $96.24 (flat)? | $96.24 | 51.3% | 2026-10-07, $95.02 | **not exceeded** |
| 2026-10-15 03:42:33 | 2026-10-06, $96.24 | Will the next real observed price be *above* $98.16 (+2%)? | $98.16 | 5.8% | 2026-10-07, $95.02 | **not exceeded** |
| 2026-10-15 03:42:33 | 2026-10-06, $96.24 | Will the next real observed price be *above* $94.32 (-2%)? | $94.32 | 93.5% | 2026-10-07, $95.02 | **exceeded** |

**First batch resolved 2026-10-15 (a later run than the one that made
these predictions — never the same run, per the no-self-grading rule
above).** EIA published new observations (2026-10-07 through 2026-10-14)
after four consecutive earlier checks that day had found nothing newer
than 2026-10-06; `resolve_pending_predictions()` resolved all three
against the first of those, 2026-10-07 ($95.02/bbl), strictly after the
prediction's own `last_observed_date` — never a later, more convenient
point, even though five more real observations existed by the time this
resolution ran. All three of this tiny first batch happen to land on the
side the stated probability favored (the two "not exceeded" calls were
predicted at 51.3%/5.8% — both correctly favored "no" — and the "exceeded"
call was predicted at 93.5%, correctly favored "yes"), which is a good
early sign but, with n=3, not itself a claim of calibration — the real
calibration evidence is the 13,328-question backtest above, not this
tiny forward sample. No new predictions were added in the same run that
resolved these, per the project's own standing discipline (see
`predictions/README.md`).

Nothing here is a guess dressed up after the fact — `made_at_utc` in
`predictions/track_record.jsonl` is the real system clock at the moment
each line was generated, and `outcome` starts, and stays, `"pending"`
until a *later* run calls `resolve_pending_predictions()` against data
that didn't exist yet when these were made. 8 new tests
(`tests/test_track_record.py`, 17 → 25): the forward-prediction function's
output is checked against the real cached series it was run on (right
last-observed date/price, threshold tracks the requested offset,
probability monotonic in the offset, timestamp genuinely recent);
append/load round-trips through a real file; and `resolve_pending_predictions`
is checked against synthetic future data for the three things that matter
for honesty — it resolves exceeded/not-exceeded correctly, it never
resolves using data at or before the question's own date, it never
re-touches an already-resolved record, and it resolves against the
*first* future observation, not whichever one is most convenient. Full
suite: **25/25 passed.**

**Second batch, added 2026-10-15** (a later, separate run from the one
that resolved the first batch — never the same run that just graded a
prior batch, per the no-self-grading rule), against the real cached
series refreshed the same run (last real observation: 2026-10-14,
$101.91/bbl):

| Made at (UTC) | Last observed | Question | Threshold | P(exceed) | Outcome |
|---|---|---|---|---|---|
| 2026-10-15 22:46:22 | 2026-10-14, $101.91 | Will the next real observed price be *above* $101.91 (flat)? | $101.91 | 51.4% | pending |
| 2026-10-15 22:46:31 | 2026-10-14, $101.91 | Will the next real observed price be *above* $103.95 (+2%)? | $103.95 | 5.0% | pending |
| 2026-10-15 22:46:41 | 2026-10-14, $101.91 | Will the next real observed price be *above* $99.87 (-2%)? | $99.87 | 94.5% | pending |

All three are genuinely pending — EIA had not published anything newer
than 2026-10-14 at the time these were made (confirmed via a fresh
`fetch_wti_daily(refresh=True)` the same run), so there is no future data
yet for any of them to resolve against. A later run should resolve these
once newer data exists, never this one.

## Edge vs. a real market price (2026-10-16 — real data, honest negative finding)

Daniel asked, live in chat on 2026-10-16, for a project whose story is "I
use this data to predict that data, and thus I've made a calculated bet
that beats the edge, and I'm profitable." This section is the real,
honest first step toward that — and an honest report that it isn't there
yet, with the specific, concrete reasons why, found by actually building
and running the real pipeline rather than assumed in advance.

**Kalshi really does expose genuine, public, live market data with no
login or trading account required** — confirmed against Kalshi's own docs
(`docs.kalshi.com/getting_started/quick_start_market_data`: *"No
authentication headers are required..."*) and independently, live, against
the real production API on 2026-10-16 (`commodity_forecast/kalshi_client.py`,
`tests/fixtures/kxwti_*_live_2026-10-16.json` — byte-for-byte real captures,
not hand-written). `KXWTI` (daily WTI threshold markets, e.g. "Will WTI
settle above $95.49 on Oct 16, 2026?") is a real series with real, live
strikes and, for many of them, real trading volume — this is not a toy or
a sandbox/demo environment.

**Building the actual comparison surfaced two honest reasons not to trust
a naive edge number yet** (a third, illiquidity, is also checked but
wasn't the binding issue in the example below, which used a genuinely
liquid market):

1. **Settlement-benchmark mismatch.** `KXWTI`'s own series metadata
   (`kalshi_client.fetch_series("KXWTI")["settlement_sources"]`) states it
   settles against the **ICE WTI Crude Futures front-month contract's**
   daily settlement price. This project's forecaster predicts the **EIA
   Cushing spot price** — a real, correlated, but genuinely different
   number.
2. **Forecast-horizon mismatch.** `fit_calibrated_naive_forecaster` is only
   ever validated one trading day ahead. Several real, liquid `KXWTI`
   markets close 2+ days out; naively reusing the one-day residual pool
   that far out is an unvalidated extrapolation.

**The real, live illustrative run (2026-10-16, refreshed EIA data, last
real observation 2026-10-14 at $101.91/bbl, compared against the real
`KXWTI-26OCT1614-*` ladder closing 2 real days later):**

| Strike | Our P(exceed) | Market P(exceed) (bid/ask mid) | Raw "edge" | Volume | Trustworthy? |
|---|---|---|---|---|---|
| $93.99 | 100.0% | 87.0% | +0.130 | 1,398.6 | No (horizon + benchmark) |
| $95.49 | 99.9% | 56.5% | +0.434 | 1,454.0 | No (horizon + benchmark) |
| $96.99 | 99.6% | 22.5% | +0.771 | 3,834.1 | No (horizon + benchmark) |
| $98.49 | 98.8% | 5.5% | +0.933 | 326.3 | No (horizon + benchmark) |
| $99.99 | 93.5% | 1.0% | +0.925 | 76.6 | No (horizon + benchmark) |

Every one of these markets had **real, non-zero trading volume** — the raw
"edge" column is not an artifact of a stale, untraded quote. Read naively,
this table looks spectacular (up to 93 percentage points of "edge"). It
almost certainly is not real skill: `edge.py`'s `compute_edge` flags every
single row above as untrustworthy, for the two structural reasons stated
above, not a one-off disclaimer in prose — `EdgeEstimate.trustworthy` is
`False` for each of them, and `size_paper_bet` refuses to size a paper bet
on any of them as a direct, code-level consequence (see `edge.py`'s own
module docstring and tests for the enforcement, not just the claim).

**This is the honest finding, not a dead end**, consistent with this
project's standing discipline of reporting what's real rather than what
looks good (see the LSTM "no measurable improvement" and the coverage-gap
results above). No "profitable, beats-the-edge" claim is made here, and
none should be until the two structural issues are actually resolved —
which is now a concrete engineering problem, not an open question, thanks
to today's real data:

- **Horizon**: extend `forecast.py` with a genuinely multi-step-ahead
  forecaster (or a separately-validated calibration residual pool *per*
  horizon, e.g. 1-day/2-day/3-day), and backtest its own coverage/
  calibration the same honest way every existing model here already is —
  not assumed, measured.
- **Benchmark**: either source a real, free historical series for ICE WTI
  front-month futures settlement prices (to recalibrate against the actual
  thing `KXWTI` settles on, or at minimum to measure the real spot/futures
  basis so it can be corrected for honestly), or find a different real
  Kalshi series that settles against EIA's own published index directly
  (not yet searched for).

`edge.py`'s paper-bet machinery (`size_paper_bet`, `PaperBet`,
`resolve_pending_paper_bets`) is real and tested end-to-end (conservative
quarter-Kelly, capped, resolved later against each market's own real
settlement result, never against this project's own forecast) — it is
simply not yet fed a trustworthy edge to act on. The first real paper bet
this project logs should be on a market where both issues above are
actually resolved, not merely ignored.

### Investigating alternative settlement benchmarks (2026-10-16, continued)

The "Benchmark" item above asks whether a different real Kalshi series
settles against EIA's own published index directly, removing the
settlement-benchmark mismatch outright. Checked for real, live, directly
against Kalshi's production API (not assumed from a series name):

- Pulled the full live series listing for `category=Commodities`
  (`GET /series?category=Commodities`) and checked every WTI-related
  ticker's own `settlement_sources` field directly — `KXWTI` (daily,
  what this project already compares against) and `KXWTIMONTHLY` both
  settle against **ICE**, same as documented above.
- Two series, `KXWTI15M` ("WTI 15-minute") and `KXWTIH` ("WTI Hourly"),
  settle against a *different* real source: **"Pyth - WTI"**, a
  continuous market-data index (`Commodities.Index.PYTHOIL/USD`), not
  ICE futures. `KXWTIH`'s own series metadata states this plainly: as of
  2026-07-30 it "transitions to a static 24/5 index feed... operating
  without intraday breaks," replacing rolling front-month futures
  contracts entirely.
- **Honest finding: this doesn't actually resolve the caveat, it changes
  its shape.** Pyth's WTI index is a continuous, real-time market price
  feed — arguably closer in spirit to "spot" than a monthly futures
  settlement — but it is still not the EIA Cushing, OK spot price this
  project's forecaster is calibrated on (a different venue and
  methodology: a market-data oracle vs. a government-published,
  delivery-point-specific survey price). No Kalshi series was found that
  settles directly against the EIA series itself. A real, live
  comparison (2026-10-16, ~16:45 UTC): the nearest-the-money open
  `KXWTI15M` market had `floor_strike` $98.04, against the cached EIA
  spot series' last value of $101.91/bbl — but that EIA value is dated
  2026-10-14, two days stale, so this ~$3.87 gap is **not** a clean,
  same-day basis measurement and isn't used as one here.
- **A second, compounding problem with `KXWTI15M`/`KXWTIH` specifically:**
  they resolve on a 15-minute/1-hour horizon — even further from this
  project's 1-day-ahead validated forecast than `KXWTI`'s own daily
  close, so neither is a cleaner drop-in replacement on the horizon axis
  either, even setting the benchmark question aside.
- **Conclusion:** the "find a different series" path is now checked and
  doesn't pan out — the real next step for the Benchmark issue remains
  sourcing a real historical ICE WTI futures series (to measure and
  correct the actual spot/futures basis against `KXWTI`, the series
  whose daily horizon already matches this project's own), not switching
  series. The Horizon issue (a genuine multi-step-ahead forecaster) is
  unaffected either way and remains open.

### Resolving the forecast-horizon mismatch (2026-10-17)

The "Horizon" item above asked for a genuine multi-step-ahead forecaster —
built this run as `forecast.fit_calibrated_naive_forecaster_horizon` +
`forecast.prepare_horizon_split`, instead of always reusing the one-day
residual pool past where it was ever validated.

**What it actually does:** the same `NaiveForecaster` + split-conformal
pipeline every other result in this README already uses, but windowed so
the *target* is genuinely `horizon` steps ahead (`series[i+lookback+horizon-1]`,
via conformal-forecast's own already-tested `make_windows(..., horizon=horizon)`
— unmodified, just threaded through where `prepare_split_from_series`
itself doesn't expose it). At `horizon=1` this is proven, not assumed, to
reduce *exactly* to the existing one-step pipeline (same coverage, same MAE,
same residual pool — see `tests/test_forecast_horizon.py`'s dedicated
check). This deliberately does not touch `conformal-forecast` itself —
only this project's own code composes that library's existing public
primitives a second way.

**Real, measured result on the full WTI series** (lookback=5, alpha=0.1;
MAE grows and the interval widens as horizon grows, exactly as it should
for a forecast held flat further and further out — a real, honest,
monotonic property, not a coincidence of one run's data):

| Horizon (trading days) | n_test | MAE ($/bbl) | Empirical coverage | Brier (threshold backtest) | Mean calibration error |
|---|---|---|---|---|---|
| 1  | 1,906 | $1.43 | 80.7% | 0.0905 | 0.0261 |
| 2  | 1,906 | $2.06 | 78.6% | 0.1163 | 0.0431 |
| 3  | 1,906 | $2.52 | 77.1% | 0.1328 | 0.0428 |
| 5  | 1,906 | $3.15 | 77.3% | 0.1521 | 0.0570 |
| 10 | 1,906 | $4.25 | 81.5% | 0.1750 | 0.0494 |
| 15 | 1,906 | $5.12 | 81.4% | 0.1918 | 0.0466 |
| 21 | 1,906 | $6.18 | 81.7% | 0.2064 | 0.0552 |

Honest takeaway: coverage stays in a fairly narrow 77–82% band across every
horizon tested (a persistent gap below the 90% nominal target — the same
known gap this README's "First real result" section already documents at
`horizon=1`, not a new problem), rather than collapsing toward 0%/100% the
way the *naive extrapolation* of the one-day pool did in the "Edge vs. a
real market price" example above. That stability is the actual point: a
calibration residual pool measured honestly *at* the horizon it's used for
degrades gracefully (wider, appropriately less certain) instead of staying
falsely confident the further out it's pushed.

`VALIDATED_FORECAST_HORIZONS_DAYS = (1, 2, 3, 5, 10, 15, 21)` in `edge.py`
is the set of horizons actually measured above; `edge.pick_validated_horizon(days_to_close)`
rounds a real market's own `days_to_close` up to the nearest one of these
(conservative — a slightly-too-wide pool, never an unvalidated one) and
returns `None` beyond 21 days, so `compute_edge` is never fed a horizon
this project hasn't actually measured.

**Real, live re-run of the "Edge vs. a real market price" example (2026-10-17,
~02:51 UTC), now passing the horizon-matched pool instead of always `horizon=1`:**

| Strike | Our P(exceed) | Market P(exceed) | Edge | Volume | days_to_close | Matched horizon | Caveats now |
|---|---|---|---|---|---|---|---|
| $99.99 | 82.1% | 39.0% | +0.431 | 886.1 | 2.65 | 3 | benchmark only |
| $99.49 | 87.3% | 45.0% | +0.423 | 258.7 | 2.65 | 3 | benchmark only |
| $98.99 | 91.6% | 52.0% | +0.396 | 734.8 | 2.65 | 3 | benchmark only |
| $98.49 | 94.1% | 59.0% | +0.351 | 1,178.6 | 2.65 | 3 | benchmark only |
| $97.99 | 96.1% | 66.0% | +0.301 | 315.0 | 2.65 | 3 | benchmark only |
| $97.49 | 97.1% | 73.0% | +0.241 | 386.0 | 2.65 | 3 | benchmark only |

**Real progress, honestly reported — the forecast-horizon-mismatch caveat
no longer fires for every row above** (`EdgeEstimate.trustworthy`'s
remaining `False` is now solely the settlement-benchmark mismatch, not
both issues at once as in the original example). This is **not** a
"profitable" result: the EIA spot cache is still 3 real days stale as of
this writing (last observation 2026-10-14, $101.91/bbl — EIA has not
published anything newer; see the forward-track-record section's own
standing note on this), and the whole reason these probabilities look so
lopsided is almost certainly the same real spot/futures basis gap
documented in the original example, not genuine model skill — exactly the
settlement-benchmark caveat that still, correctly, fires. One real caveat
down, one to go: the next real engineering step is unchanged from the
"What's next" item below — sourcing a historical ICE WTI futures series to
measure and correct that basis directly.

**A first, real attempt at that next step (2026-10-17) — a negative
finding, worth recording so a future run doesn't retread it:** EIA itself
publishes a real, free, no-API-key daily *NYMEX* WTI front-month futures
settlement series (`RCLC1`, "Cushing, OK Crude Oil Future Contract 1"),
served the same way as the spot series this project already uses
(`https://www.eia.gov/dnav/pet/hist_xls/RCLC1d.xls`). Confirmed real by
actually downloading and parsing it, not assumed from its name. **It
stopped being updated in April 2024** (10,183 real rows, the last dated
2024-04-05) — EIA has discontinued this series, so it cannot be used to
measure or correct today's real spot/futures basis against a live market.
It also measures the NYMEX WTI contract specifically, not the ICE WTI
contract `KXWTI` actually settles against — a related but genuinely
different futures contract, so even a current RCLC1 would only ever have
been an approximate proxy for the real ICE basis, not an exact one. The
real remaining path for the Benchmark issue is therefore either a
paid/API-key data vendor for live ICE WTI futures settlement prices (a
real cost/access decision for Daniel, not taken on unilaterally), or an
explicitly-caveated historical-average-basis estimate computed from
RCLC1's own stale overlap with the spot series (not a live correction,
and not yet attempted).

### Resolving the settlement-benchmark mismatch (2026-10-19)

The "Benchmark" item above asked for a real, free historical series for
ICE WTI futures settlement prices, or a measured real spot/futures basis.
RCLC1 (NYMEX, discontinued) was a dead end, and no paid vendor has been
used (that remains Daniel's own cost/access call, not taken on
unilaterally). **A different, real, free path was found instead: Kalshi's
own public API already publishes everything needed, just not as a single
number.**

`KXWTI` never reports the real ICE futures settlement price directly, but
every settled `KXWTI` market *is* a real "did it settle above $X?"
question, and Kalshi lists a whole real ladder of these (many strikes,
$0.50 apart) for every settlement date. Once every market in a ladder has
resolved, the real settlement value is bracketed exactly between the
highest strike that resolved "yes" and the lowest that resolved "no" — a
real, measured interval, never an external vendor or an assumption.
`commodity_forecast/benchmark.py`'s `bracket_settlements` recovers this
from `kalshi_client.fetch_markets(..., status="all")`'s own already-tested
real output.

**Pulled the full real `KXWTI` settlement history live (2026-10-19, 1,605
real market objects, 45 real bracketable settlement dates spanning
2026-08-12 through 2026-10-16 — one date, 2026-09-02, had a real ladder
that resolved entirely one-sided and was correctly skipped, nothing to
bracket) and paired each real bracket with this project's own cached EIA
spot observation for the same date (43 of 45 dates matched; the other 2
are more recent than the EIA cache's own last refresh).**

**Real, honest finding #1 — the basis is not a stable constant:**

| Period | Real measured basis (futures mid − EIA spot) |
|---|---|
| 2026-08-12 to 2026-08-31 | roughly −$1.0 to −$3.0 |
| 2026-09-01 to 2026-09-16 | roughly −$0.97 to −$1.45 |
| 2026-09-17 to 2026-09-22 | a real, sharp widening: −$4.7 to −$6.0 |
| 2026-09-23 to 2026-10-16 (most recent 2.5 weeks) | tight and stable: −$6.6 to −$7.0 |

This is consistent with `KXWTI` settling against ICE's *front-month*
contract, which rolls to a new underlying contract roughly monthly — a
single all-history average basis would blend two genuinely different
regimes and actively mislead, not safely correct. One real date
(2026-09-25) also shows a one-day ~$10 EIA spot value that fully reverts
the next observation — almost certainly a data artifact in the upstream
EIA series, reported here rather than silently dropped or smoothed over
(`measure_basis` does not filter it; see the module docstring).

**Real, honest finding #2 — the basis *is* tight and stable within a short
recent window, which is what a live correction actually needs:**
`recent_basis_estimate` over the most recent 10 real paired observations
(2026-10-02 through 2026-10-16) measured **mean −$6.86, stdev $0.16** —
tight enough that `describe_basis_correction`'s conservative trust bar
(stdev ≤ $1.00, n ≥ 5) clears easily. `basis_adjusted_strike` converts a
market's real futures-terms strike into the spot-terms strike this
project's forecaster can actually answer (`futures = spot + basis`, so
`futures > K` ⟺ `spot > K − basis`), and `EdgeEstimate.notes` now carries
the real correction's own numbers on every estimate it's applied to —
transparency about what was corrected, not just a flipped boolean.

**Real, live result of applying this (2026-10-19, see also the staleness
fix below, which this example already incorporates): the settlement-
benchmark caveat genuinely clears, and *no profitable edge survives it*.**
Once the basis correction is applied on top of an honestly-matched
forecast horizon, the previously "spectacular" apparent edge mostly
reverses sign — see the live table in the next section. This is reported
exactly as measured: resolving a real caveat is not the same as finding a
real edge, and this project does not pretend otherwise.

### Fixing a staleness bug in horizon-matching (2026-10-19)

Found while re-running the live edge example above with the new basis
correction: `pick_validated_horizon` was being fed `days_to_close`
computed from *today's wall-clock date* to a market's `close_time` — a
silent assumption that this project's cached EIA spot series is current.
**It frequently is not** (the EIA cache was 5 real days stale, last
observation 2026-10-14, as of this check). Using "now" instead of the
forecast's actual data origin understates how far out the forecast is
really being asked to extrapolate, which made `pick_validated_horizon`
select an inappropriately *narrow* (overconfident) validated horizon —
reintroducing the exact forecast-horizon-mismatch problem this machinery
exists to prevent, just through the staleness gap instead of the market's
own distance from today.

**Real, measured difference this made, for a real market closing
2026-10-20T18:30:00Z, checked 2026-10-19:**

| Computed from | days_to_close | Matched horizon |
|---|---|---|
| wall-clock "now" (the old, buggy way) | 1.05 | **2** |
| the EIA cache's own last observation (2026-10-14) — honest | 6.77 | **10** |

Fixed with a new `edge.days_to_close_from_last_observation(close_time,
last_observed_date)`, used in place of a bare `close_time - now` wherever
this project matches a real market to a validated horizon. This is a real
correctness fix (flagged here plainly, not silently folded into the
benchmark work above) — any real paper bet sized under stale data without
it would have been working from an overconfident, under-validated
residual pool.

**Real, live re-run of the edge example (2026-10-19, ~17:xx UTC) with both
fixes applied — the real `KXWTI` ladder closing 2026-10-20T18:30:00Z,
staleness-aware `days_to_close=6.77` → matched horizon 10, basis-corrected
at mean −$6.86 (trusted: stdev $0.16, n=10):**

| Strike | Raw P(exceed) | Corrected P(exceed) | Market P(exceed) | Raw edge | Corrected edge | Volume |
|---|---|---|---|---|---|---|
| $93.99 | 97.0% | 61.8% | 97.0% | −0.000 | −0.352 | 500.0 |
| $95.99 | 91.0% | 41.6% | 90.5% | +0.005 | −0.489 | 245.5 |
| $97.99 | 81.1% | 22.2% | 68.5% | +0.126 | −0.463 | 367.0 |
| $99.99 | 69.1% | 8.2% | 35.5% | +0.336 | −0.273 | 175.0 |
| $101.99 | 50.6% | 2.6% | 13.5% | +0.371 | −0.109 | 5,636.0 |
| $103.99 | 29.8% | 0.9% | 4.5% | +0.253 | −0.036 | 30.0 |

(Full 26-strike ladder available by re-running the snippet above against
live data; this table samples across the strike range.)

**The honest headline: with the staleness bug fixed, the *raw* (still
benchmark-mismatched) edge is already much smaller than the original
2026-10-16 example's 60-95-point apparent edges — most of that original
gap was the horizon bug compounding with stale data, not the benchmark
mismatch alone. Once the trustworthy basis correction is also applied,
`EdgeEstimate.trustworthy` is `True` for every row above (both real
caveats resolved) — and the resulting edge is consistently *negative*,
not positive.** This project's own corrected, honestly-calibrated
probability thinks these strikes are *less* likely to be exceeded than
the real market does, not more. **No profitable, beats-the-edge claim is
made here.** Both structural caveats named in "Edge vs. a real market
price" above are now resolved, and resolving them did not produce a real
edge — if anything, it removed one. This is the actual, measured
conclusion, reported the same honest way every other result in this
README is, not reframed to fit the "profitable" framing Daniel's
2026-10-16 direction asked this project to investigate.

### An edge-subset survey: does any real slice of the ladder show a different edge? (2026-10-19, continued)

The two sections above resolved both named structural caveats and found,
honestly, that the *ladder-wide average* corrected edge is not
trustworthy. The natural follow-up this project hadn't yet asked: does
any real **subset** of the live ladder look different from that average
— e.g. strikes near the current price vs. far out-of-the-money ones,
short-horizon vs. long-horizon questions, liquid vs. illiquid markets?

Built `commodity_forecast/edge_survey.py` (`survey_live_edges` +
`summarize_edge_subsets` + `format_edge_subset_summary`), 8 new tests
(111→119). `survey_live_edges` runs the full real pipeline (horizon pick,
basis correction, `compute_edge`) across *every* real open market on a
series in one call, instead of the one-market-at-a-time usage every prior
example above showed — fitting one horizon-validated forecaster per
distinct real horizon the live ladder actually needs, never once per
market. **Honest scope, stated plainly in the module docstring too: this
is one live snapshot, not a claim that any subset's edge is *persistent*
over time** — this project has no repeated-snapshot history yet to
support that stronger claim (see "What's next" below).

**Real, live result (2026-10-19, ~18:xx UTC, real `KXWTI` ladder, 105 real
open markets, 0 skipped):**

```
subset                    n   mean edge   mean |edge|
all                     103     -0.0173        0.1547
near-the-money           12     -0.2865        0.3028
far-out-of-the-money     91     +0.0182        0.1351
short-horizon             0         n/a           n/a
long-horizon            103     -0.0173        0.1547
liquid                  103     -0.0173        0.1547
illiquid                  0         n/a           n/a
```

(`short-horizon`/`illiquid` are empty on this particular snapshot — no
real open market right now closes within the cached data's own 5-day
horizon or trades zero volume; reported as `n/a`, not hidden.)

**The near-the-money subset's large negative number looked, at first
glance, like exactly the "some slice is different" finding this survey
was built to look for — investigated rather than taken at face value, and
it isn't one.** Reading the real `KXWTI-26OCT2014` ladder's own strikes in
order shows the market's own ~50%-probability crossing point sits around
a basis-adjusted $105/bbl, while this project's forecaster is still
anchored to its last real observed EIA spot price, $101.91 on
2026-10-14 — five real days stale (the same staleness this project's own
EIA ingestion has been honestly flagging since 2026-10-15, and confirmed
unchanged by a fresh `fetch_wti_daily(refresh=True)` call this run: EIA
still hasn't published anything newer). A market pricing in five more
real days of information our forecaster's own anchor point doesn't have
will disagree most sharply exactly *at* the strikes nearest that stale
anchor — which is exactly the near-the-money subset. Once a strike is far
enough from the stale anchor, both sides agree the probability is near 0%
or 100% regardless of a few dollars' difference in where "today's price"
really is, which is exactly why the far-OTM subset's mean edge sits close
to zero instead.

**Honest conclusion: no real subset of the live ladder shows a
trustworthy, persistent positive edge.** The one subset that looked
different (near-the-money) is explained by this project's own
already-known data-staleness limitation, not a real, tradable market
inefficiency — and that explanation itself is a useful, concrete result:
it says plainly that any future live use of this project's probabilities
would be *most* unreliable exactly where a trader would naturally want to
look first (strikes close to the current price), not a uniform small
error everywhere. No "profitable" or "beats-the-edge" claim is made here,
consistent with every other result in this README.

### Tracking persistence across real snapshots (2026-10-19, continued)

The edge-subset survey above is explicit that one snapshot can't support a
*persistent*-edge claim — a subset looking different on one day's ladder
could easily be noise. This run builds the machinery to actually check
that over time, rather than leaving it as a hypothetical "What's next"
item: `commodity_forecast/edge_persistence.py`
(`append_survey_snapshot` / `load_survey_history` / `summarize_persistence`
/ `format_persistence_summary`), 10 new tests (119→129).

`append_survey_snapshot` records one real, dated snapshot of a
`summarize_edge_subsets` result to `data/edge_subset_history.jsonl` —
append-only, one real JSON line per real survey run, never backdated or
synthetic. `summarize_persistence` reads back every snapshot recorded so
far and, per subset label, reports whether the sign of `mean_edge` has
been consistent across every real snapshot that had data for it — but
only once at least `MIN_SNAPSHOTS_FOR_A_PERSISTENCE_VERDICT` (3) real data
points exist; with fewer, it honestly reports "not enough data yet" rather
than calling a 1-for-1 or 2-for-2 sign match "persistent," which would be
indistinguishable from chance.

**This run recorded the first real entry in that history file** (a fresh
live survey, not a reuse of the snapshot earlier in this same document):

```
subset                    n   mean edge   mean |edge|
all                     103     -0.0455        0.1435
near-the-money           12     -0.3240        0.3240
far-out-of-the-money     91     -0.0088        0.1197
short-horizon             0         n/a           n/a
long-horizon            103     -0.0455        0.1435
liquid                  103     -0.0455        0.1435
illiquid                  0         n/a           n/a
```

Consistent with the snapshot earlier in this document, taken roughly two
hours earlier the same day: `near-the-money` is again the most negative
subset (-0.324 here vs. -0.2865 then) — in line with the same EIA-
staleness explanation (the spot anchor is still stuck at 2026-10-14, so
nothing about the explanation changed between the two snapshots).
`far-out-of-the-money` moved from a small positive (+0.0182) to a small
negative (-0.0088) — both close to zero, the kind of noise-level movement
expected if there's no real subset-level signal there, not a reversal
worth reading into. **With only one real entry recorded in the new
history file, `summarize_persistence` correctly reports "not enough real
snapshots yet" for every subset — this is not yet a persistence claim,
honestly stated as such.** A genuine verdict needs at least two more real
recorded snapshots from future runs.

**A second real entry was recorded roughly two hours later the same day**
(a fresh, live `survey_live_edges` + `summarize_edge_subsets` run, 105
real open markets, 0 skipped, at `2026-10-19T22:48:27Z`):

```
subset                    n   mean edge   mean |edge|
all                     103     -0.0414        0.1395
near-the-money           12     -0.3181        0.3181
far-out-of-the-money     91     -0.0049        0.1160
short-horizon             0         n/a           n/a
long-horizon            103     -0.0414        0.1395
liquid                  103     -0.0414        0.1395
illiquid                  0         n/a           n/a
```

Every subset's sign matches the first snapshot exactly (`near-the-money`
and `all`/`long-horizon`/`liquid` all negative both times; `far-OTM` now
negative both times too, both close to zero), and the magnitudes moved by
only a few thousandths — consistent with real intra-day market-price
movement on the live `KXWTI` ladder, not a different regime. **With 2 of
the required 3 real snapshots now recorded, `summarize_persistence`
still correctly reports "not enough real snapshots yet (2/3 needed)" for
every subset** — two agreeing signs is suggestive but not yet the
pre-registered threshold this project set for itself, and this project
does not lower that bar just because the next snapshot could clear it. A
third real recorded snapshot from a future run will produce this
project's first real sign-consistency verdict.

## What's next

1. ~~Try `conformal-forecast`'s existing Adaptive Conformal Inference
   (fixed-pool and sliding-window variants) on this series instead of
   static split conformal, to see whether it closes the coverage gap above
   the way it did for `conformal-forecast`'s own airline case.~~ **Done** —
   see above; a bounded sliding window gets within ~0.2-0.5pp of nominal.
2. ~~A stronger model than the naive baseline (conformal-forecast's LSTM is
   already available as a dependency) — does ACI's benefit above hold up
   with a model that isn't a trivial "today predicts tomorrow" baseline,
   or does a better point model change which calibration method wins?~~
   **Done (2026-10-15)** — see "Does a stronger point model help?" above.
   Honest answer: no, not measurably, on this series — WTI's near-random-
   walk behavior means the LSTM converges to essentially the same
   zero-delta forecast `NaiveForecaster` already makes, so every
   downstream number (coverage, interval width, Brier score, calibration
   error) comes out statistically indistinguishable from the naive
   baseline's. A real, useful negative result, not a dead end: it means
   the naive baseline wasn't quietly understating what's achievable here,
   so items 3 and the adaptive-pool work above are measuring real
   calibration-method effects, not an artifact of a weak point model.
   conformal-forecast's Transformer model family (shipped there 2026-10-15,
   not yet tried here) remains available if a future run wants a second
   data point on "does model choice matter at all on this series."
3. ~~A principled way to pick `window` instead of eyeballing a sweep (e.g.
   a held-out validation slice)~~ **Done (2026-10-15)** — see "Picking the
   adaptive pool's window honestly" above. `select_adaptive_pool_window`'s
   rolling-origin CV correctly selects window=150 out of 8 real
   candidates, landing close to the oracle's own window=200 (0.0136 vs.
   0.0101 real test-set calibration error) and substantially beating the
   static baseline (0.0261) — a genuine, measured win this time, unlike
   the mixed result the same idea got on conformal-forecast's own
   smallest-calibration-set EnbPI case the same day.
4. ~~Convert a calibrated interval into an actual Kalshi-style threshold
   probability (e.g. P(price > K) by a given date), and backtest that
   conversion against real history.~~ **Done** — see above.
5. ~~Wire the probability layer into ACI's online-adapting residual pool
   (currently it only ever uses `fit_calibrated_naive_forecaster`'s static
   calibration-split residuals) — plausibly improves calibration the same
   way ACI improved raw interval coverage, but unverified until it's
   actually built and backtested the same honest way.~~ **Done** — see
   "Does an online, adapting residual pool improve the probability
   estimate?" above. Measured result was more nuanced than "yes, strictly
   better": Brier score barely moved, but mean calibration error improved
   substantially (~2.5x) at a well-chosen bounded window, with the same
   "bounded beats unbounded" pattern the interval-ACI case already showed.
6. ~~Now that the probability layer is backtested, start the real forward
   track record in `predictions/` — one dated, non-backdated entry at a
   time: a specific real strike K and date, the probability this module
   assigned it *before* the outcome was known, and later, the real
   outcome.~~ **Done** — see "Forward track record" above. The first
   batch (3 predictions) was itself resolved by a later run the same day
   (2026-10-15) once newer EIA data existed — all three landed on the
   side their stated probability favored (not itself a calibration claim
   at n=3; see the backtest above for the real evidence). **A second batch
   (3 more predictions) was added 2026-10-15** by a later, separate run —
   see "Forward track record" above; all three are still pending as of
   this writing (EIA has not yet published anything newer than their own
   `last_observed_date`). The natural follow-up: a *future* run should
   resolve this second batch once newer EIA data exists (never the run
   that just added it), and once enough predictions accumulate across
   batches, start reporting a running hit rate / calibration check on the
   forward record itself, honestly caveated for its still-small sample
   size.
7. **(New, 2026-10-16) Make the real Kalshi edge comparison trustworthy.**
   See "Edge vs. a real market price" above for the two concrete, real
   issues found (settlement-benchmark mismatch: EIA spot vs. ICE futures;
   forecast-horizon mismatch: 1-day-validated model vs. 2+-day-out real
   markets) and the specific engineering path to resolving each. Only once
   at least one real market passes both checks should this project log its
   first real (paper-only) bet via `edge.py`'s already-built and tested
   `size_paper_bet` / `PaperBet` / `resolve_pending_paper_bets` machinery.
   ~~Check whether a different real Kalshi series settles against EIA's
   own index directly, removing the benchmark mismatch outright.~~
   **Checked (2026-10-16)** — see "Investigating alternative settlement
   benchmarks" above. Honest result: no such series exists; the two
   Pyth-settled alternatives found (`KXWTI15M`, `KXWTIH`) swap the
   benchmark problem for a different one rather than removing it, and
   make the horizon mismatch worse, not better. ~~Build a genuine
   multi-step-ahead forecaster (removing the horizon mismatch).~~ **Done
   (2026-10-17)** — see "Resolving the forecast-horizon mismatch" above.
   `fit_calibrated_naive_forecaster_horizon` is real, measured, and tested
   at 7 real horizons (1/2/3/5/10/15/21 trading days); `pick_validated_horizon`
   wires it into `compute_edge` so the horizon caveat clears for any real
   market closing within 21 days, which it now does in the re-run live
   example above. ~~Source a historical ICE WTI futures series (or a
   measured basis) to resolve the settlement-benchmark mismatch.~~ **Done
   (2026-10-19)** — see "Resolving the settlement-benchmark mismatch"
   above: RCLC1/NYMEX was a dead end, but Kalshi's own settlement history
   brackets the real ICE futures value directly, with no external vendor
   needed. **Both real structural caveats are now resolved** — and a real
   staleness bug in horizon-matching, found while re-running the live
   example with the new correction, was fixed the same run (see "Fixing a
   staleness bug in horizon-matching" above). **The honest conclusion:
   resolving both caveats did not produce a profitable edge — the
   corrected comparison's edge is consistently negative, not positive**
   (see the live re-run table above). No "profitable, beats-the-edge"
   claim is made, and the real next step is no longer "resolve the
   caveats" but "investigate why no edge survives honest correction" —
   one candidate is now checked: ~~does any real subset of the live
   ladder (near-the-money, long vs. short horizon, liquid vs. illiquid)
   show a persistently different edge than the ladder-wide average?~~
   **Checked (2026-10-19)** — see "An edge-subset survey" above.
   Honest result: no, not a real one — the one subset that looked
   different (near-the-money) is explained by this project's own
   already-known EIA staleness gap, not a genuine market inefficiency.
   Remaining candidates, not yet attempted: the real basis measured here
   is only ~2.5 weeks of history (worth re-checking as more settlement
   dates accumulate, since a contract roll could shift it again); the
   `NaiveForecaster` point forecast may simply be too weak once the
   genuinely-validated horizon is as wide as 10 trading days (worth
   re-trying conformal-forecast's own LSTM/Transformer/N-BEATS model
   families at this horizon specifically, not just at horizon=1 as
   already checked); and the EIA spot cache's own staleness (still 5 real
   days behind as of this writing) degrades the point forecast itself,
   independent of which horizon it's matched to — the edge-subset survey
   above is itself new evidence that this specific gap matters most
   right where it would be most tempting to look for a real edge.
8. ~~Repeat the edge-subset survey on successive runs and check whether any
   one subset's edge sign/magnitude actually holds up over multiple real
   snapshots, which is the only way a *persistent* (not just
   single-snapshot) edge claim could honestly be supported.~~ **Machinery
   built and the first real snapshot recorded (2026-10-19, continued)** —
   see "Tracking persistence across real snapshots" above:
   `edge_persistence.py`, `data/edge_subset_history.jsonl`. Honestly
   still in progress, not done: a real persistence verdict needs at least
   `MIN_SNAPSHOTS_FOR_A_PERSISTENCE_VERDICT` (3) real recorded snapshots,
   and only one exists so far. **A future run should call
   `survey_live_edges` + `summarize_edge_subsets` +
   `append_survey_snapshot` again (not re-use this run's numbers) and
   check `summarize_persistence`'s output once 2-3 real entries exist.**

## Running it

```bash
pip install -e ".[dev]"
python -m commodity_forecast.data   # refresh the real cached series from EIA
pytest
```
