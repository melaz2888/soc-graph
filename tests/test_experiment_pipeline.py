from datetime import timedelta
from pathlib import Path

from soc_graph.data.dataset import build_datasets
from soc_graph.data.pidsmaker import load_events
from soc_graph.model.pipeline import run_baseline_experiment


def test_run_baseline_experiment_on_sample_fixture() -> None:
    fixture = Path("tests/fixtures/sample_pidsmaker.csv")
    events = load_events(fixture)
    snapshots, artifacts = build_datasets(events, window=timedelta(minutes=15))

    result = run_baseline_experiment(
        snapshot_dataset=snapshots,
        artifact_dataset=artifacts,
        benign_ratio=0.5,
        threshold_k=1.0,
    )

    assert result.training_summary.num_windows == 1
    assert result.model_state.learned_threshold is not None
    assert result.graph_stats["num_windows"] == 2.0
    assert len(result.test_snapshots) == 1
    assert len(result.flagged_windows) == 1
    assert result.flagged_windows[0]

