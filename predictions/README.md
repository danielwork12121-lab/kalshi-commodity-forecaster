# Forward track record

Real, dated predictions, made before the outcome is known — never
backdated. This is the step the whole project was building toward (see
`active-project-kalshi.md`'s "Why this exists" and the main README's
no-fabrication commitment): a probability stated on the record *before*
the fact it's about has happened, not fit or chosen afterward.

## How this differs from the backtest

The main README's "From a calibrated interval to a Kalshi-style
probability" section reports a *backtest*: 13,328 real historical
questions the method had never seen the answer to at the time it was
run, but all of them already resolved by the time the measurement was
made. Everything in this folder is the opposite kind of real: each entry
is a single, specific, forward-looking question, asked once, with a real
wall-clock timestamp proving when it was asked — resolved later, only
once the real outcome actually exists, never before.

## Format

One real prediction per line in `track_record.jsonl` (append-only; never
rewritten except by `resolve_pending_predictions()` filling in an
already-pending record's `outcome`/`resolved_*` fields once real data
makes that possible — see `commodity_forecast/track_record.py`). Fields:

| Field | Meaning |
|---|---|
| `made_at_utc` | Real wall-clock UTC timestamp this prediction was generated — always read fresh at call time, never backdated or supplied by a caller. |
| `last_observed_date` | The most recent real WTI spot-price date the model had when it made this prediction. |
| `last_observed_price_usd` | That date's real closing spot price. |
| `relative_offset` | The threshold, expressed relative to `last_observed_price_usd` (e.g. `0.02` = "2% above"). |
| `threshold_usd` | `last_observed_price_usd * (1 + relative_offset)`, in real dollars/barrel. |
| `predicted_probability` | P(the next real observed price > `threshold_usd`), from `commodity_forecast.probability.exceedance_probability`, using the exact same calibration-residual machinery the backtest above already validated. |
| `model` | Which method produced the probability. |
| `lookback`, `alpha` | The same hyperparameters used everywhere else in this project. |
| `outcome` | `"pending"` until resolved, then `"exceeded"` or `"not_exceeded"`. |
| `resolved_at_utc`, `resolved_date`, `resolved_price_usd` | Filled in only once resolved — the real timestamp of resolution, the real date of the first observation strictly after `last_observed_date`, and its real price. |

## Resolving predictions

A **future** run — never the same run that made the prediction — should
call `commodity_forecast.track_record.resolve_pending_predictions()` with
a freshly refreshed series (`refresh_data=True` upstream, or pass a fresh
`series` directly) before adding any new predictions of its own. This is
what keeps the record honest: a prediction is only ever graded against
data that didn't exist yet when it was made, and only once that data
actually exists — never sooner, and never by the same process that made
the guess.

## Status

See `track_record.jsonl` for the real entries. First batch (3 predictions,
made 2026-10-15T03:42:33 UTC against the 2026-10-06 observation) was
resolved by a later run the same day (2026-10-15T19:20:24 UTC), once EIA
published newer data — all three resolved against 2026-10-07 ($95.02/bbl),
the first real observation strictly after the batch's own
`last_observed_date`, never a more convenient later point even though
several more observations existed by resolution time. See the main
README's "Forward track record" section for the resolved outcomes. No new
predictions were added in the run that resolved these, per the
no-self-grading rule above — a future run should add the next batch
separately.

**Second batch (3 predictions, made 2026-10-15T22:46 UTC against the
2026-10-14 observation, $101.91/bbl)** was added by a later, separate run
— never the run that resolved the first batch. All three are still
`"pending"` as of this writing; EIA had not published anything newer than
2026-10-14 at the time they were made. A future run should resolve them
once newer data exists, and should not add a third batch in that same
run, per the no-self-grading rule above.
