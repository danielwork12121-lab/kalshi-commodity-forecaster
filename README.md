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
repository has no connection to that one.)

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
                (static calibration) and AdaptiveConformalForecaster /
                SlidingWindowAdaptiveConformalForecaster (ACI)
  probability.py  converts a calibrated forecast into a Kalshi-style
                  threshold probability P(price > K), backtested against
                  real history with both a static and an online-adapting
                  residual pool (since 2026-10-15)
  track_record.py makes + persists real, dated, forward (non-backtested)
                   predictions, and resolves them once real outcomes exist
                   (since 2026-10-15)
tests/
  test_data.py         date-math unit test + live EIA integration test + cache-hit test
  test_forecast.py     synthetic-data + real-WTI-series end-to-end tests, static and ACI
  test_probability.py  hand-computed sanity checks + a real-WTI-series backtest
  test_track_record.py forward-prediction structure checks + synthetic-future resolution checks
data/
  wti_spot_daily.csv real cached WTI daily spot price series, 1986-01-02 to present
predictions/
  README.md         explains the forward track record's format and resolution rule
  track_record.jsonl the real, dated predictions themselves (since 2026-10-15)
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
$96.24/bbl — re-verified fresh against EIA this run; no newer observation
was available from the source yet) and the exact same calibration-residual
machinery the backtest above already validated:

| Made at (UTC) | Last observed | Question | Threshold | P(exceed) | Outcome |
|---|---|---|---|---|---|
| 2026-10-15 03:42:33 | 2026-10-06, $96.24 | Will the next real observed price be *above* $96.24 (flat)? | $96.24 | **51.3%** | pending |
| 2026-10-15 03:42:33 | 2026-10-06, $96.24 | Will the next real observed price be *above* $98.16 (+2%)? | $98.16 | **5.8%** | pending |
| 2026-10-15 03:42:33 | 2026-10-06, $96.24 | Will the next real observed price be *above* $94.32 (-2%)? | $94.32 | **93.5%** | pending |

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

## What's next

1. ~~Try `conformal-forecast`'s existing Adaptive Conformal Inference
   (fixed-pool and sliding-window variants) on this series instead of
   static split conformal, to see whether it closes the coverage gap above
   the way it did for `conformal-forecast`'s own airline case.~~ **Done** —
   see above; a bounded sliding window gets within ~0.2-0.5pp of nominal.
2. A stronger model than the naive baseline (conformal-forecast's LSTM is
   already available as a dependency) — now with a real, interesting
   question to pair it with: does ACI's benefit above hold up with a model
   that isn't a trivial "today predicts tomorrow" baseline, or does a
   better point model change which calibration method wins?
3. A principled way to pick `window` instead of eyeballing a sweep (e.g. a
   held-out validation slice), since this run's own sweep shows the choice
   matters less than *having* a bounded window at all, but isn't flat.
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
   outcome.~~ **Done** — see "Forward track record" above; 3 real pending
   predictions as of 2026-10-15. The natural follow-up: a *future* run
   should call `resolve_pending_predictions()` against fresh data before
   adding new predictions, and once a few resolve, report the running
   hit rate honestly (small-sample caveats included) rather than only
   ever adding new pending entries.

## Running it

```bash
pip install -e ".[dev]"
python -m commodity_forecast.data   # refresh the real cached series from EIA
pytest
```
