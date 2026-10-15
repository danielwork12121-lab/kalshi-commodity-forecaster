"""The real, forward, non-backdated prediction track record.

Everything built before this module (`forecast.py`'s calibrated naive
forecaster, `probability.py`'s threshold-probability conversion) answers a
*backtested* question: "if this method had been used across the last
~1,900 real test-split days, how often would its stated probabilities have
matched what actually happened?" That's a real, honest measurement (Brier
0.0906, mean calibration error 0.026 -- see `probability.py` and the main
README), but it's still retrospective: every question it answers was
already resolved by the time the backtest looked at it.

This module answers the question this whole project was actually built
for (see `active-project-kalshi.md`'s "Why this exists" and Daniel's own
2026-10-14 request): a prediction made *before* the outcome is known, with
the timestamp to prove it, resolved later against the real outcome once
it's actually observed -- never the other way around.

Mechanics, deliberately simple and auditable:

- `make_forward_prediction()` reuses the exact same fitted model + the
  exact same calibration-residual sample `fit_calibrated_naive_forecaster`
  and `probability.exceedance_probability` already compute and already
  have a real, backtested calibration record. The only new step is
  applying that already-verified machinery to a point *beyond* the end of
  the historical series -- the real, not-yet-observed next trading day's
  closing spot price -- rather than to a historical test point whose
  answer the backtest already knew. `made_at_utc` is always read fresh
  from the real system clock at call time; nothing here ever accepts or
  defaults to a caller-supplied timestamp, so a record's own `made_at_utc`
  can't be set to look earlier than it really is.
- `append_prediction()` / `load_predictions()` persist records as one JSON
  object per line (`predictions/track_record.jsonl`) -- a plain, diffable,
  append-friendly format that makes "was anything inserted out of order or
  edited after the fact" a visible `git diff`, not something hidden inside
  a binary or a rewritten table.
- `resolve_pending_predictions()` is the *only* function that ever changes
  an existing record's `outcome` -- and only once real data strictly after
  the prediction's own `last_observed_date` actually exists. It is safe to
  call repeatedly (a record already resolved, or with no newer data yet
  available, is left untouched) and is meant to be called by a *future*
  run, once the real calendar has actually moved forward -- never by the
  same run that made the prediction, which is the discipline that keeps
  this an honest forward track record rather than a self-graded one.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from dataclasses import asdict, dataclass, field

import numpy as np

from commodity_forecast.data import fetch_wti_daily
from commodity_forecast.forecast import fit_calibrated_naive_forecaster
from commodity_forecast.probability import exceedance_probability

PREDICTIONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "predictions"
DEFAULT_LOG_PATH = PREDICTIONS_DIR / "track_record.jsonl"

MODEL_LABEL = "naive+split_conformal(calibration-residual-ECDF)"


@dataclass
class PredictionRecord:
    """One real, dated forward prediction.

    `outcome` starts `"pending"` and is only ever changed by
    `resolve_pending_predictions()`, never by `make_forward_prediction()`
    (which cannot know the future) and never by hand.
    """

    made_at_utc: str
    last_observed_date: str
    last_observed_price_usd: float
    relative_offset: float
    threshold_usd: float
    predicted_probability: float
    model: str
    lookback: int
    alpha: float
    outcome: str = "pending"  # "pending" | "exceeded" | "not_exceeded"
    resolved_at_utc: str | None = None
    resolved_date: str | None = None
    resolved_price_usd: float | None = None

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json_line(line: str) -> "PredictionRecord":
        return PredictionRecord(**json.loads(line))


def _now_utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def make_forward_prediction(
    relative_offset: float = 0.0,
    lookback: int = 5,
    alpha: float = 0.1,
    refresh_data: bool = False,
) -> PredictionRecord:
    """Make ONE real, forward prediction: "will the next real observed WTI
    spot price (the trading day immediately after the most recent one this
    data source has) close above `threshold_usd`?", where `threshold_usd`
    is `relative_offset` above the last *observed* price (e.g. 0.0 asks
    "will it go up at all", 0.02 asks "will it rise at least 2%").

    Fits via the exact same, already-tested `fit_calibrated_naive_forecaster`
    every other real result in this project is measured against -- no
    separate, unverified code path. The only genuinely new step is forming
    the model's input window from the *last* `lookback` real observations
    (there is no held-out `y` for this window; it hasn't happened yet) and
    reading the probability off the already-fit calibration-residual
    sample, via the same `exceedance_probability` the backtest already
    validated.
    """
    series = fetch_wti_daily(refresh=refresh_data)
    last_date = series.index.max()
    last_price = float(series.iloc[-1])

    result = fit_calibrated_naive_forecaster(series.values, lookback=lookback, alpha=alpha)

    last_window_normalized = (
        (np.asarray(series.values[-lookback:], dtype=np.float64) - result.normalizer_mean)
        / result.normalizer_std
    ).reshape(1, -1)
    point_forecast_normalized = result.conformal_forecaster.model.predict(last_window_normalized).reshape(-1)

    threshold_usd = last_price * (1.0 + relative_offset)
    threshold_normalized = (threshold_usd - result.normalizer_mean) / result.normalizer_std

    predicted_probability = float(
        exceedance_probability(
            point_forecast_normalized, result.cal_residuals_normalized, threshold_normalized
        )[0]
    )

    return PredictionRecord(
        made_at_utc=_now_utc_iso(),
        last_observed_date=last_date.date().isoformat(),
        last_observed_price_usd=last_price,
        relative_offset=relative_offset,
        threshold_usd=threshold_usd,
        predicted_probability=predicted_probability,
        model=MODEL_LABEL,
        lookback=lookback,
        alpha=alpha,
    )


def append_prediction(record: PredictionRecord, path: pathlib.Path | str = DEFAULT_LOG_PATH) -> None:
    """Append one record to the track-record log. Never overwrites or
    reorders existing lines -- this is the only function that should ever
    add a *new* prediction to the log (as opposed to `resolve_pending_predictions`,
    which only ever edits the `outcome` fields of lines already there).
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(record.to_json_line() + "\n")


def load_predictions(path: pathlib.Path | str = DEFAULT_LOG_PATH) -> list[PredictionRecord]:
    path = pathlib.Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [PredictionRecord.from_json_line(line) for line in f if line.strip()]


def resolve_pending_predictions(
    series=None,
    path: pathlib.Path | str = DEFAULT_LOG_PATH,
) -> list[PredictionRecord]:
    """Resolve every `"pending"` record in `path` against real data
    observed strictly after that record's own `last_observed_date`.

    Only a data point *after* `last_observed_date` can resolve a record --
    a point at or before it would be answering a question about data the
    prediction already had, not the forward question it actually asked.
    A pending record with no such data yet is left pending, unchanged. An
    already-resolved record (`outcome != "pending"`) is never re-touched,
    so re-running this after a record is resolved is always a safe no-op
    for that record.

    `series` defaults to a fresh (`refresh=True`) real fetch so this always
    checks against the real, current data source -- pass a specific
    `series` (e.g. in tests) to avoid hitting the network.

    Returns every record in the file, resolved and still-pending alike.
    Rewrites the whole file only if at least one record actually changed
    (the file is small -- one line per prediction ever made -- so a full
    rewrite is simpler than an in-place patch and still leaves every
    unresolved or already-resolved line byte-for-byte unless it changed).
    """
    if series is None:
        series = fetch_wti_daily(refresh=True)

    records = load_predictions(path)
    changed = False
    for rec in records:
        if rec.outcome != "pending":
            continue
        last_date = _dt.date.fromisoformat(rec.last_observed_date)
        future_mask = np.array([d > last_date for d in series.index.date])
        if not future_mask.any():
            continue
        future = series[future_mask]
        resolved_date = future.index[0]
        resolved_price = float(future.iloc[0])
        rec.outcome = "exceeded" if resolved_price > rec.threshold_usd else "not_exceeded"
        rec.resolved_date = resolved_date.date().isoformat() if hasattr(resolved_date, "date") else str(resolved_date)
        rec.resolved_price_usd = resolved_price
        rec.resolved_at_utc = _now_utc_iso()
        changed = True

    if changed:
        path = pathlib.Path(path)
        with open(path, "w") as f:
            for rec in records:
                f.write(rec.to_json_line() + "\n")

    return records
