# kalshi-commodity-forecaster

Calibrated commodity-price forecasting and Kalshi-style threshold probabilities, built on [`conformal-forecast`](https://github.com/danielwork12121-lab/conformal-forecast). Paper-only analysis: it never places an order, and it found **no trustworthy profitable signal**.

| WTI daily spot (EIA), 1986-2026 backtest | Result |
|---|---|
| 90% interval coverage, static split conformal (naive model) | 80.9% (MAE $1.43/bbl) |
| Same, sliding-pool ACI (`window=75`) | 89.8% (width $4.18 to $6.42/bbl) |
| Threshold-probability Brier score (0.25 = no skill) | 0.0906 static; calibration error 0.026 to 0.0136 adaptive |
| Edge vs. live Kalshi prices, after correction | none found; near-the-money mean edge -0.29 to -0.34 (5 snapshots) |

```bash
pip install -e ".[dev]"
python -m commodity_forecast.data   # refresh the cached EIA series
pytest                               # 220 tests
```

## Not a trading tool

This project never touches a Kalshi account and never places or stages an order. It reads public, unauthenticated Kalshi market data and produces probability estimates. `edge.py` sizes and logs only **paper, simulated** bets, and refuses to size one on any estimate flagged untrustworthy. Nothing here is financial advice.

## How results are labeled

- Backtests are labeled as backtests, never as live predictions.
- Forward predictions (`predictions/`) are append-only, timestamped before the outcome is known, and resolved later against the first real observation after the prediction's last data point. Nothing is backdated.
- Gaps and negative results are reported, not hidden.

## Architecture

```
commodity_forecast/
  data.py            EIA daily WTI spot (RWTC) download + local CSV cache; no API key
  forecast.py        wraps conformal-forecast: static/adaptive (ACI) calibration; naive, LSTM and
                     Transformer point models; multi-step-ahead (horizon) forecasters
  probability.py     calibrated forecast -> P(price > K); static and online-adapting residual pool;
                     rolling-origin CV selection of the adaptive pool's window
  track_record.py    forward predictions, persisted and resolved later
  kalshi_client.py   public Kalshi market-data client with retry/backoff
  edge.py            our probability vs. market-implied probability, with explicit caveat flags
  benchmark.py       recovers/measures the ICE-futures-vs-EIA-spot basis from Kalshi settlement history
  edge_survey.py     survey a whole live ladder; edge_persistence.py tracks it across snapshots
  horizon_model_comparison.py   naive vs LSTM vs Transformer at each validated horizon
  markets/           MarketFamily interface + WTI, temperature (NYC, Miami), crude-storage families
```

No splitting, windowing, normalization or calibration logic is reimplemented here; those come from `conformal-forecast`. The one exception is the signed-residual online pool in `probability.py`, since `conformal-forecast`'s ACI classes track only absolute residuals; it follows the same score-first-then-update ordering.

**Data.** EIA's daily WTI Cushing spot series (`RWTCd.xls`), fetched directly from EIA's spreadsheet (the v2 JSON API needs a registered key). EIA revises its most recent two to three days of preliminary data, so a refresh can change recent cached values. Forward predictions are unaffected because each threshold is written immutably when the prediction is made.

## Results

### Interval coverage

Naive model (lookback 5, alpha 0.1), 9,517 points; train / calibration / test = 5,705 / 1,903 / 1,904.

| Method | Coverage | Gap vs 90% | Mean width |
|---|---|---|---|
| Static split conformal | 80.9% | -9.1pp | $4.18/bbl |
| ACI, fixed pool | 87.4% | -2.6pp | $5.49/bbl |
| ACI, sliding pool (unbounded) | 88.4% | -1.6pp | $7.13/bbl |
| ACI, sliding pool, `window=75` | 89.8% | -0.2pp | $6.42/bbl |

Static conformal under-covers because the series is non-stationary (it includes the 2020 negative price, -$36.98/bbl), so calibration and test residuals are only approximately exchangeable. Bounded sliding windows from 20 to 300 all land at 89.5-89.8%, so `window=75` is one reproducible point, not a uniquely optimal value. The extra coverage costs width: a correctly calibrated 90% interval here is nearer $6.42 than $4.18, and the narrower static interval was narrower because it under-covered.

**Width normalization caveat.** `result.diagnostics["mean_interval_width"]` is in normalized (z-score) units. Multiply by `result.normalizer_std` ($26.428/bbl for this series) to get dollars: 0.1582 x 26.428 = $4.18. An early version of this table published the normalized 0.16 as if it were dollars; all widths here are denormalized. MAE and coverage were already correct.

### Threshold probabilities

Treating the model's held-out calibration residuals as an empirical error distribution gives P(price > K) with no assumed shape. Backtest: for each of 1,904 test days, 7 offsets relative to the point forecast (-10% to +10%), 13,328 questions.

| Method | Brier | Mean calibration error |
|---|---|---|
| Static (frozen pool) | 0.0906 | 0.0258 |
| Adaptive, unbounded pool | 0.0901 | 0.0197 |
| Adaptive, `window=75` | 0.0901 | 0.0138 |
| Adaptive, `window=200` (best in hindsight) | 0.0894 | 0.0100 |

The honest read: Brier barely moves (it is dominated by easy tail questions); calibration improves roughly 2.5x at the best bounded window, and an unbounded pool is worse than any window tried between 75 and 500. The static version's worst bucket (0.4-0.5: 42.3% predicted vs 80.0% actual) has only 5 observations, so it is small-sample noise. The method assumes the calibration residual sample is the whole predictive distribution at every level and time, which is the module's main stated limitation.

`select_adaptive_pool_window` picks the window by rolling-origin CV over the calibration set (5 folds), never reading the test set. It selected `window=150` out of 8 candidates: real-test calibration error 0.0136, against the oracle's `window=200` at 0.0101 and the static baseline's 0.0261.

### Does a stronger point model help?

At one day ahead, no. An LSTM matches naive (MAE $1.44 vs $1.43, coverage 80.6% vs 80.9%, Brier 0.0907 vs 0.0906). WTI daily spot is close to a random walk, so a zero-change forecast is already near the best available. The LSTM never wins on MAE at any horizon. A Transformer does show a small, growing advantage from horizon 3 onward:

| Horizon (trading days) | Naive MAE / coverage | Transformer MAE / coverage |
|---|---|---|
| 1 | $1.43 / 80.7% | $1.44 / 81.0% |
| 2 | $2.06 / 78.6% | $2.06 / 78.7% |
| 3 | $2.52 / 77.1% | $2.50 / 77.3% |
| 5 | $3.15 / 77.3% | $3.13 / 77.8% |
| 10 | $4.25 / 81.5% | $4.16 / 82.6% |
| 15 | $5.12 / 81.4% | $5.00 / 83.7% |
| 21 | $6.18 / 81.7% | $6.04 / 84.1% |

At horizon 21 that is about a 2.3% MAE reduction. Coverage stays in a 77-84% band at every horizon, below 90% nominal; measuring the residual pool at the horizon it is used for makes intervals degrade gracefully instead of staying falsely confident. The live survey uses the Transformer by default (`model_name="transformer"`); `"naive"` reproduces earlier behavior.

### Forward track record

`predictions/track_record.jsonl` holds forward predictions of "will the next observed price exceed flat / +2% / -2%?". At last count it has 15 records: 7 resolved exceeded, 5 not exceeded, 3 pending. That sample is far too small to say anything about calibration; the 13,328-question backtest is the real evidence.

### Edge versus real Kalshi prices

Kalshi's `KXWTI` daily threshold markets settle on **ICE front-month futures**, while the forecaster predicts the **EIA Cushing spot** price, and most open markets close several days out. Two structural issues therefore made a naive edge number untrustworthy:

1. **Settlement-benchmark mismatch.** No Kalshi series settles on EIA spot. The alternatives found (`KXWTI15M`, `KXWTIH`, Pyth-settled) swap the problem for a different one and have even shorter horizons. EIA's free NYMEX futures series (`RCLC1`) was discontinued in April 2024.
2. **Horizon mismatch.** Fixed by a multi-step forecaster validated at horizons 1, 2, 3, 5, 10, 15 and 21 days; `edge.pick_validated_horizon` matches each market to the nearest validated horizon at or above its distance (none beyond 21 days).

The settlement basis is recovered from Kalshi's own settled ladders: the highest strike resolving "yes" and the lowest resolving "no" bracket the true settlement value. The basis is **not** a stable constant. It was roughly -$1 to -$3 in early periods, widened sharply to -$4.7 to -$6.0, and was about -$6.6 to -$7.0 over the most recent weeks of that sample, consistent with front-month rolls. One EIA spot value in the sample was a one-day ~$10 outlier that reverted; it is reported, not filtered. The correction is applied only when the recent 10-observation basis has stdev <= $1.00 and n >= 5; that bar was never loosened.

A staleness bug was found along the way: `days_to_close` was computed from the wall-clock date rather than from the last observation date, which understated the forecast horizon and selected an overconfident residual pool (horizon 2 instead of 10 in the example checked). It was fixed with `days_to_close_from_last_observation`.

**The first uncorrected comparison showed apparent edges of 60 to 93 percentage points. Those were artifacts of the two mismatches plus stale data, not skill.** After both fixes, one live snapshot looked like this:

| Strike | Raw edge | Corrected edge |
|---|---|---|
| $93.99 | -0.000 | -0.352 |
| $97.99 | +0.126 | -0.463 |
| $99.99 | +0.336 | -0.273 |
| $103.99 | +0.253 | -0.036 |

With the caveats resolved, the corrected edge is consistently **negative**: our calibrated probability is below the market's. Resolving the caveats did not reveal an edge; if anything it removed one.

**Persistence across snapshots** (mean edge by subset, naive model, trustworthy estimates only; `edge_persistence.py` requires at least 3 snapshots before issuing a sign-consistency verdict):

| Snapshot | all | near-the-money | far-out-of-the-money |
|---|---|---|---|
| 1 | -0.0455 | -0.3240 | -0.0088 |
| 2 | -0.0414 | -0.3181 | -0.0049 |
| 3 | -0.0579 | -0.3385 | -0.0195 |
| 4 | -0.0284 | -0.2902 | +0.0061 |
| 5 | -0.0514 | -0.3387 | +0.0049 |

- `all` and `near-the-money` stay negative in every snapshot; near-the-money sits between -0.29 and -0.34. This is evidence the model's probability runs below the market's, not evidence of a tradeable opportunity.
- **`far-out-of-the-money` flipped sign at snapshot 4.** After snapshot 3 it briefly met the sign-consistency bar; the flip **retracted that "consistent" verdict**, and it is now classed as noise around zero.
- The large near-the-money number is partly explained by data staleness: the market prices in days of information the stale spot anchor lacks, so disagreement is largest right at the strikes a trader would look at first.
- A sixth snapshot had zero trustworthy estimates: EIA's revisions to its latest days distorted the 10-observation basis window (stdev 3.19, over the 1.0 bar), so every market was correctly excluded. It is kept as real data rather than discarded.
- The Transformer-model pipeline's trustworthy `all` subset has been `n=0` in every snapshot recorded, because the basis gate has not passed. Its raw uncorrected mean edge (about +0.27) matches the known futures-below-spot basis effect and is not a finding.
- `short-horizon` and `illiquid` subsets have never contained a market.

### Other market families

The `MarketFamily` interface (`list_open_markets`, `estimate_probability`, `settle`) and a shared `run_family_survey` loop let new market types reuse `edge.compute_edge`, which is market-agnostic.

- **Temperature (`KXHIGHNY`, `KXHIGHMIA`).** Settles on The Weather Company, not NWS. Forecasts use a climatological empirical distribution from Open-Meteo history with a +/-0.5 degF continuity correction. Leave-one-year-out coverage: NYC 80.9% at an 80% target and 89.6% at 90% (n=1,461); Miami 78.4% and 88.3% (n=1,086), slightly under-covering. The basis correction **fails** the trust bar for both (NYC stdev 1.14 degF, Miami mean +1.63 / stdev 1.56 degF), so trustworthy subsets are empty. A cache bug was found and fixed: every station silently shared the NYC cache file, which would have contaminated a second station.
- **Crude storage (`KXEIACRUDEW`).** Kalshi's `expiration_value` gives the EIA-reported storage figure directly (three formats are parsed). Basis: mean -0.0094 million bbl, stdev 0.018, n=8, the only family whose correction passes the trust gate out of the box. The first snapshot (n=3, mean edge +0.0068) is inconclusive. A later sign change in the `liquid` subset traced to which strikes happened to have volume at survey time, not a reversal of the per-strike pattern. Cached EIA values may include later revisions.
- **Natural-gas storage.** No Kalshi market matches this idea (`KXNATGASW` settles on a price index), so none was built.
- **Empty-response guard.** `run_family_survey` retries once if a 200 response carries an empty market list, so a transient blip cannot masquerade as "nothing trustworthy". Investigation showed earlier `n=0` entries were the trust filter working, not fetch failures.

## Limitations

- **No trustworthy profitable signal was found.** On WTI, where the basis correction passed, the edge was negative, and the one subset whose sign briefly looked consistent was retracted. Crude-storage results are inconclusive; temperature has no trustworthy subset.
- Everything is paper analysis. Forward track-record samples are tiny.
- Static coverage is below nominal at every horizon (77-82% for naive). Adaptive calibration (ACI) was measured only on the one-day-ahead backtest.
- The spot/futures basis is estimated from a short, regime-dependent window and is gated by a strict stdev rule; the gate fails often.
- Data freshness matters: a stale EIA anchor degrades forecasts most near the money, and EIA revises recent data.
- Probability estimates assume one residual sample describes the whole predictive distribution at every level and time.
- Climatology (temperature) and the storage series are single-source and lightly validated; results from one station do not transfer (Miami differs from NYC).

## Changelog

- Initial: EIA ingestion, static and ACI-calibrated WTI intervals.
- Probability layer, adaptive pool, CV window selection; LSTM comparison; forward track record.
- Kalshi client, edge computation, multi-horizon forecasting, ICE basis recovery, staleness fix.
- 0.7: horizon-level model comparison (naive / LSTM / Transformer); edge-persistence tracking.
- 0.8: Transformer becomes the live default.
- 0.9: `markets/` framework; WTI refactored into the first family.
- 0.10-0.12: temperature (NYC, Miami) and crude-storage families; per-station cache fix.
- 0.13: empty-response retry in `run_family_survey`.

## License

MIT, see [LICENSE](LICENSE).
