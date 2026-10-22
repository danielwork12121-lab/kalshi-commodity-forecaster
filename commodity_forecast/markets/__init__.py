"""The multi-market edge-survey framework (v0.9.0).

One `MarketFamily` per real Kalshi market family (`base.py` defines the
shared interface; `wti.py` is the first real implementation, ported from
`edge_survey.py`'s own previously-inline WTI logic). `run_family_survey`
(in `base.py`) is the market-agnostic driving loop every family reuses:
fetch the real open ladder, score each market via that family's own
`estimate_probability`, compare against the real market price via
`edge.compute_edge`, and group the results into the same named
cross-sectional subsets every family shares (`edge_survey.summarize_edge_subsets`).

See `active-project-kalshi.md`'s 2026-10-21 direction-update item 2 for
why this exists: the WTI ladder has shown no trustworthy edge across 6
naive-model and 1 Transformer-model real snapshots, and re-measuring the
same market forever is not progress -- the next real family (temperature
vs. NWS point forecasts) is where a real edge is more plausible, and this
package is what makes adding it a matter of implementing one new
`MarketFamily`, not rewriting the survey/reporting machinery again.
"""

from commodity_forecast.markets.base import (
    MarketFamily,
    MarketNotScorable,
    ProbabilityEstimate,
    SurveyedEdge,
    run_family_survey,
)

__all__ = [
    "MarketFamily",
    "MarketNotScorable",
    "ProbabilityEstimate",
    "SurveyedEdge",
    "run_family_survey",
]
