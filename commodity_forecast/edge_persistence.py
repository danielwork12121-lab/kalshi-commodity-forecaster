"""Tracking whether an edge-subset survey's result actually *persists*
across repeated real snapshots over time, rather than stopping at one.

## Why this exists

`edge_survey.py`'s own module docstring says plainly that a single
snapshot cannot support a persistence claim: a subset that shows a larger
edge on one day's ladder could easily be noise. Its "What's next" item
(and `active-project-kalshi.md`'s "What's next" item 3, 2026-10-19) names
the honest next step as running the same survey on successive runs and
checking whether any one subset's edge sign/magnitude actually holds up
over multiple real snapshots.

This module is the machinery for that, not a persistence claim itself. A
single real run can only add one more real data point to the history file
and report what the history *so far* shows -- which, with only one or two
points, is honestly "not enough data yet," not a verdict either way. No
synthetic or backdated snapshots are ever added: `append_survey_snapshot`
only ever records a *real* `summarize_edge_subsets` result from the
moment it's called, timestamped at that real moment, same discipline as
`track_record.py`'s own forward-only prediction log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from commodity_forecast.edge_survey import EdgeSubsetSummary

#: Minimum number of real snapshots, each contributing a real (non-zero-n)
#: data point for a given subset, before this module will call a sign
#: "consistent" or "inconsistent" rather than "not enough data yet."
MIN_SNAPSHOTS_FOR_A_PERSISTENCE_VERDICT = 3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_survey_snapshot(
    summaries: list[EdgeSubsetSummary],
    history_path: str,
    run_at_utc: str | None = None,
) -> dict:
    """Append one real, dated snapshot of a `summarize_edge_subsets` result
    to a JSONL history file (creating it if it doesn't exist yet). Returns
    the exact record written.

    `run_at_utc` defaults to the real current UTC time -- pass an explicit
    value only to make the recorded timestamp exact when the survey itself
    was run moments before this call (e.g. from a script that measures
    `survey_live_edges` and this append in two separate steps), never to
    backdate a snapshot to a time before this function was actually called.
    """
    record = {
        "run_at_utc": run_at_utc if run_at_utc is not None else _utc_now_iso(),
        "subsets": [
            {
                "label": s.label,
                "n": s.n,
                "mean_edge": s.mean_edge,
                "mean_abs_edge": s.mean_abs_edge,
            }
            for s in summaries
        ],
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_survey_history(history_path: str) -> list[dict]:
    """Load every real snapshot recorded so far. Returns `[]` if the file
    doesn't exist yet (the honest state before this project's first real
    recorded snapshot) rather than raising."""
    try:
        with open(history_path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


@dataclass
class SubsetPersistence:
    """The real persistence picture for one subset label across every real
    snapshot recorded so far that had at least one scored market in that
    subset (`n > 0`) -- a snapshot where a subset was empty contributes no
    sign to agree or disagree with, so it's excluded from this count
    rather than silently counted as either.
    """

    label: str
    n_snapshots_with_data: int
    mean_edges: list[float] = field(default_factory=list)
    sign_consistent: bool | None = None  # None = not enough data yet for a verdict
    verdict: str = ""


def summarize_persistence(history: list[dict]) -> list[SubsetPersistence]:
    """For each subset label present in the history, report how many real
    snapshots had data for it, every real recorded mean edge, and whether
    the sign has been consistent across all of them -- or, honestly,
    that there isn't enough real history yet to say (fewer than
    `MIN_SNAPSHOTS_FOR_A_PERSISTENCE_VERDICT` real data points).

    Never reports a "persistent" verdict from noise: with only 1-2 real
    snapshots, a sign matching by chance is not meaningfully different
    from a sign that doesn't, so both are reported as "not enough data
    yet," not as a premature "persistent"/"not persistent" claim.
    """
    labels: list[str] = []
    for snap in history:
        for s in snap.get("subsets", []):
            if s["label"] not in labels:
                labels.append(s["label"])

    results = []
    for label in labels:
        mean_edges = [
            s["mean_edge"]
            for snap in history
            for s in snap.get("subsets", [])
            if s["label"] == label and s["n"] > 0
        ]
        n = len(mean_edges)
        if n < MIN_SNAPSHOTS_FOR_A_PERSISTENCE_VERDICT:
            results.append(
                SubsetPersistence(
                    label=label,
                    n_snapshots_with_data=n,
                    mean_edges=mean_edges,
                    sign_consistent=None,
                    verdict=f"not enough real snapshots yet ({n}/{MIN_SNAPSHOTS_FOR_A_PERSISTENCE_VERDICT} needed)",
                )
            )
            continue
        signs = {e > 0 for e in mean_edges}
        consistent = len(signs) == 1
        results.append(
            SubsetPersistence(
                label=label,
                n_snapshots_with_data=n,
                mean_edges=mean_edges,
                sign_consistent=consistent,
                verdict=(
                    f"sign consistent across all {n} real snapshots"
                    if consistent
                    else f"sign flipped across the {n} real snapshots recorded so far -- looks like noise, not a persistent edge"
                ),
            )
        )
    return results


def format_persistence_summary(results: list[SubsetPersistence]) -> str:
    """A plain-text table for the README -- real counts and real recorded
    mean-edge values only, never a significance claim beyond what
    `summarize_persistence` itself already computed."""
    lines = [f"{'subset':<22} {'snapshots':>9}  verdict"]
    for r in results:
        lines.append(f"{r.label:<22} {r.n_snapshots_with_data:>9}  {r.verdict}")
    return "\n".join(lines)
