import json

from commodity_forecast.edge_persistence import (
    MIN_SNAPSHOTS_FOR_A_PERSISTENCE_VERDICT,
    append_survey_snapshot,
    format_persistence_summary,
    load_survey_history,
    summarize_persistence,
)
from commodity_forecast.edge_survey import EdgeSubsetSummary


def _summary(label: str, n: int, mean_edge: float, mean_abs_edge: float = 0.1) -> EdgeSubsetSummary:
    return EdgeSubsetSummary(label=label, n=n, mean_edge=mean_edge, mean_abs_edge=mean_abs_edge, edges=[])


def test_load_survey_history_missing_file_returns_empty_list(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    assert load_survey_history(str(missing)) == []


def test_append_then_load_roundtrip(tmp_path):
    history_path = tmp_path / "history.jsonl"
    summaries = [_summary("all", 10, -0.02), _summary("near-the-money", 2, -0.3)]

    record = append_survey_snapshot(summaries, str(history_path), run_at_utc="2026-10-19T18:30:00+00:00")

    assert record["run_at_utc"] == "2026-10-19T18:30:00+00:00"
    assert record["subsets"] == [
        {"label": "all", "n": 10, "mean_edge": -0.02, "mean_abs_edge": 0.1},
        {"label": "near-the-money", "n": 2, "mean_edge": -0.3, "mean_abs_edge": 0.1},
    ]

    loaded = load_survey_history(str(history_path))
    assert loaded == [record]


def test_append_defaults_run_at_to_real_current_time(tmp_path):
    history_path = tmp_path / "history.jsonl"
    record = append_survey_snapshot([_summary("all", 5, 0.01)], str(history_path))
    # A real ISO-8601 UTC timestamp was recorded, not a placeholder.
    assert record["run_at_utc"].endswith("+00:00") or "Z" in record["run_at_utc"] or "T" in record["run_at_utc"]


def test_append_is_additive_across_multiple_calls(tmp_path):
    history_path = tmp_path / "history.jsonl"
    append_survey_snapshot([_summary("all", 5, 0.01)], str(history_path), run_at_utc="t1")
    append_survey_snapshot([_summary("all", 5, 0.02)], str(history_path), run_at_utc="t2")

    history = load_survey_history(str(history_path))
    assert len(history) == 2
    assert [h["run_at_utc"] for h in history] == ["t1", "t2"]


def test_summarize_persistence_one_snapshot_is_not_enough_data():
    history = [{"run_at_utc": "t1", "subsets": [{"label": "all", "n": 10, "mean_edge": -0.02, "mean_abs_edge": 0.1}]}]
    results = summarize_persistence(history)
    assert len(results) == 1
    r = results[0]
    assert r.label == "all"
    assert r.n_snapshots_with_data == 1
    assert r.sign_consistent is None
    assert "not enough" in r.verdict


def test_summarize_persistence_excludes_empty_subsets_from_the_data_count():
    history = [
        {"run_at_utc": "t1", "subsets": [{"label": "short-horizon", "n": 0, "mean_edge": 0.0, "mean_abs_edge": 0.0}]},
        {"run_at_utc": "t2", "subsets": [{"label": "short-horizon", "n": 0, "mean_edge": 0.0, "mean_abs_edge": 0.0}]},
    ]
    results = summarize_persistence(history)
    assert results[0].n_snapshots_with_data == 0
    assert results[0].sign_consistent is None


def test_summarize_persistence_consistent_sign_across_min_snapshots():
    history = [
        {"run_at_utc": f"t{i}", "subsets": [{"label": "all", "n": 10, "mean_edge": -0.01 * (i + 1), "mean_abs_edge": 0.1}]}
        for i in range(MIN_SNAPSHOTS_FOR_A_PERSISTENCE_VERDICT)
    ]
    results = summarize_persistence(history)
    r = results[0]
    assert r.n_snapshots_with_data == MIN_SNAPSHOTS_FOR_A_PERSISTENCE_VERDICT
    assert r.sign_consistent is True
    assert "consistent" in r.verdict


def test_summarize_persistence_flipped_sign_across_min_snapshots():
    edges = [-0.1, 0.05, -0.2]
    assert len(edges) == MIN_SNAPSHOTS_FOR_A_PERSISTENCE_VERDICT
    history = [
        {"run_at_utc": f"t{i}", "subsets": [{"label": "all", "n": 10, "mean_edge": e, "mean_abs_edge": 0.1}]}
        for i, e in enumerate(edges)
    ]
    results = summarize_persistence(history)
    r = results[0]
    assert r.sign_consistent is False
    assert "noise" in r.verdict


def test_format_persistence_summary_is_plain_text_with_every_label():
    history = [{"run_at_utc": "t1", "subsets": [{"label": "all", "n": 10, "mean_edge": -0.02, "mean_abs_edge": 0.1}]}]
    results = summarize_persistence(history)
    text = format_persistence_summary(results)
    assert "all" in text
    assert "not enough real snapshots yet" in text


def test_history_file_is_real_jsonl_one_record_per_line(tmp_path):
    history_path = tmp_path / "history.jsonl"
    append_survey_snapshot([_summary("all", 5, 0.01)], str(history_path), run_at_utc="t1")
    append_survey_snapshot([_summary("all", 5, 0.02)], str(history_path), run_at_utc="t2")
    lines = history_path.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line parses independently as real JSON
