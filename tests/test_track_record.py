import datetime as _dt

import numpy as np
import pandas as pd
import pytest

from commodity_forecast.data import fetch_wti_daily
from commodity_forecast.track_record import (
    PredictionRecord,
    append_prediction,
    load_predictions,
    make_forward_prediction,
    resolve_pending_predictions,
)


def test_make_forward_prediction_on_real_cached_series_is_well_formed():
    """Uses the real, currently-cached WTI series (no network) -- checks
    this is a genuinely *forward* prediction (dated today, about the next,
    not-yet-observed point), not a disguised backtest."""
    rec = make_forward_prediction(relative_offset=0.0, lookback=5, alpha=0.1, refresh_data=False)

    series = fetch_wti_daily(refresh=False)
    assert rec.last_observed_date == series.index.max().date().isoformat()
    assert rec.last_observed_price_usd == pytest.approx(float(series.iloc[-1]))
    assert rec.threshold_usd == pytest.approx(rec.last_observed_price_usd)
    assert 0.0 <= rec.predicted_probability <= 1.0
    assert rec.outcome == "pending"
    assert rec.resolved_at_utc is None
    assert rec.resolved_date is None
    assert rec.resolved_price_usd is None

    # made_at_utc must be the real wall-clock time this call happened --
    # within a couple of minutes of "now", never far in the past or future.
    made_at = _dt.datetime.fromisoformat(rec.made_at_utc)
    now = _dt.datetime.now(_dt.timezone.utc)
    assert abs((now - made_at).total_seconds()) < 120


def test_make_forward_prediction_threshold_tracks_relative_offset():
    rec_up = make_forward_prediction(relative_offset=0.05, refresh_data=False)
    rec_flat = make_forward_prediction(relative_offset=0.0, refresh_data=False)
    rec_down = make_forward_prediction(relative_offset=-0.05, refresh_data=False)

    assert rec_up.threshold_usd == pytest.approx(rec_flat.last_observed_price_usd * 1.05)
    assert rec_down.threshold_usd == pytest.approx(rec_flat.last_observed_price_usd * 0.95)
    # A higher bar is harder to clear -> predicted probability should not
    # increase as the threshold rises (same monotonicity probability.py's
    # own tests already establish for exceedance_probability directly).
    assert rec_up.predicted_probability <= rec_flat.predicted_probability <= rec_down.predicted_probability


def test_append_and_load_predictions_roundtrip(tmp_path):
    path = tmp_path / "track_record.jsonl"
    rec1 = make_forward_prediction(relative_offset=0.0, refresh_data=False)
    rec2 = make_forward_prediction(relative_offset=0.02, refresh_data=False)

    append_prediction(rec1, path=path)
    append_prediction(rec2, path=path)

    loaded = load_predictions(path=path)
    assert len(loaded) == 2
    assert loaded[0] == rec1
    assert loaded[1] == rec2


def test_load_predictions_on_missing_file_returns_empty_list(tmp_path):
    assert load_predictions(path=tmp_path / "does_not_exist.jsonl") == []


def _synthetic_series(dates, prices):
    return pd.Series(prices, index=pd.DatetimeIndex(dates), name="wti_spot_usd_per_bbl")


def test_resolve_pending_predictions_resolves_exceeded_and_not_exceeded(tmp_path):
    path = tmp_path / "track_record.jsonl"
    pending_exceed = PredictionRecord(
        made_at_utc="2026-10-14T12:00:00+00:00",
        last_observed_date="2026-10-14",
        last_observed_price_usd=100.0,
        relative_offset=0.0,
        threshold_usd=100.0,
        predicted_probability=0.5,
        model="test",
        lookback=5,
        alpha=0.1,
    )
    pending_not_exceed = PredictionRecord(
        made_at_utc="2026-10-14T12:00:00+00:00",
        last_observed_date="2026-10-14",
        last_observed_price_usd=100.0,
        relative_offset=0.05,
        threshold_usd=105.0,
        predicted_probability=0.2,
        model="test",
        lookback=5,
        alpha=0.1,
    )
    append_prediction(pending_exceed, path=path)
    append_prediction(pending_not_exceed, path=path)

    future_series = _synthetic_series(["2026-10-15"], [103.0])
    resolved = resolve_pending_predictions(series=future_series, path=path)

    assert resolved[0].outcome == "exceeded"
    assert resolved[0].resolved_date == "2026-10-15"
    assert resolved[0].resolved_price_usd == pytest.approx(103.0)
    assert resolved[0].resolved_at_utc is not None

    assert resolved[1].outcome == "not_exceeded"
    assert resolved[1].resolved_price_usd == pytest.approx(103.0)

    # Written back to disk, not just returned in memory.
    reloaded = load_predictions(path=path)
    assert reloaded[0].outcome == "exceeded"
    assert reloaded[1].outcome == "not_exceeded"


def test_resolve_pending_predictions_leaves_record_pending_with_no_newer_data(tmp_path):
    path = tmp_path / "track_record.jsonl"
    rec = PredictionRecord(
        made_at_utc="2026-10-14T12:00:00+00:00",
        last_observed_date="2026-10-14",
        last_observed_price_usd=100.0,
        relative_offset=0.0,
        threshold_usd=100.0,
        predicted_probability=0.5,
        model="test",
        lookback=5,
        alpha=0.1,
    )
    append_prediction(rec, path=path)

    # Only data at-or-before last_observed_date is "available" -- must NOT resolve.
    same_day_series = _synthetic_series(["2026-10-14"], [101.0])
    resolved = resolve_pending_predictions(series=same_day_series, path=path)
    assert resolved[0].outcome == "pending"
    assert resolved[0].resolved_price_usd is None

    reloaded = load_predictions(path=path)
    assert reloaded[0].outcome == "pending"


def test_resolve_pending_predictions_never_touches_already_resolved_records(tmp_path):
    path = tmp_path / "track_record.jsonl"
    already_resolved = PredictionRecord(
        made_at_utc="2026-10-01T12:00:00+00:00",
        last_observed_date="2026-10-01",
        last_observed_price_usd=90.0,
        relative_offset=0.0,
        threshold_usd=90.0,
        predicted_probability=0.5,
        model="test",
        lookback=5,
        alpha=0.1,
        outcome="exceeded",
        resolved_at_utc="2026-10-02T12:00:00+00:00",
        resolved_date="2026-10-02",
        resolved_price_usd=91.0,
    )
    append_prediction(already_resolved, path=path)

    future_series = _synthetic_series(["2026-10-03"], [999.0])
    resolved = resolve_pending_predictions(series=future_series, path=path)

    # Untouched: still the original resolution, not re-computed against the
    # unrelated later series passed into this call.
    assert resolved[0] == already_resolved


def test_resolve_pending_predictions_uses_first_future_observation_not_the_last(tmp_path):
    """Resolving against the *first* real observation after the question
    was asked -- not the latest available -- is what keeps this an honest
    answer to "what happens next", not "what eventually happened"."""
    path = tmp_path / "track_record.jsonl"
    rec = PredictionRecord(
        made_at_utc="2026-10-14T12:00:00+00:00",
        last_observed_date="2026-10-14",
        last_observed_price_usd=100.0,
        relative_offset=0.0,
        threshold_usd=100.0,
        predicted_probability=0.5,
        model="test",
        lookback=5,
        alpha=0.1,
    )
    append_prediction(rec, path=path)

    multi_day_series = _synthetic_series(
        ["2026-10-14", "2026-10-15", "2026-10-16"], [100.0, 95.0, 999.0]
    )
    resolved = resolve_pending_predictions(series=multi_day_series, path=path)
    assert resolved[0].resolved_date == "2026-10-15"
    assert resolved[0].resolved_price_usd == pytest.approx(95.0)
    assert resolved[0].outcome == "not_exceeded"
