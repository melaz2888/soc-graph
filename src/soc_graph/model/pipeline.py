from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from soc_graph.data.dataset import ArtifactGraphDataset, WindowedGraphDataset
from soc_graph.data.schemas import GraphSnapshot
from soc_graph.detection.threshold import flag_scores

from .detector import BehavioralAnomalyDetector
from .state import TemporalModelState
from .train import TrainingSummary, detect_anomalies, fit_detector


# ---------------------------------------------------------------------------
# Baseline (behavioral) experiment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineExperimentResult:
    training_summary: TrainingSummary
    model_state: TemporalModelState
    graph_stats: dict[str, float]
    test_snapshots: list[GraphSnapshot]
    flagged_windows: list[dict[str, float]]


def summarize_graph_windows(dataset: WindowedGraphDataset) -> dict[str, float]:
    if not dataset.snapshots:
        return {
            "num_windows": 0.0,
            "mean_nodes_per_window": 0.0,
            "mean_edges_per_window": 0.0,
            "mean_edge_observations_per_window": 0.0,
        }

    return {
        "num_windows": float(len(dataset)),
        "mean_nodes_per_window": mean(len(s.nodes) for s in dataset.snapshots),
        "mean_edges_per_window": mean(len(s.edges) for s in dataset.snapshots),
        "mean_edge_observations_per_window": mean(
            s.total_edge_observations for s in dataset.snapshots
        ),
    }


def run_baseline_experiment(
    snapshot_dataset: WindowedGraphDataset,
    artifact_dataset: ArtifactGraphDataset,
    benign_ratio: float = 0.7,
    threshold_k: float = 3.0,
) -> BaselineExperimentResult:
    train_snapshots, test_snapshots = snapshot_dataset.train_test_split(benign_ratio=benign_ratio)
    train_artifacts, test_artifacts = artifact_dataset.train_test_split(benign_ratio=benign_ratio)

    detector = BehavioralAnomalyDetector()
    model_state = TemporalModelState()
    training_summary = fit_detector(
        detector=detector,
        benign_artifacts=train_artifacts,
        threshold_k=threshold_k,
    )
    model_state.learned_threshold = training_summary.learned_threshold
    edge_profiles, signatures = detector.profile_snapshot()
    model_state.register_profiles(edge_profiles=edge_profiles, signatures=signatures)
    for artifact in train_artifacts:
        model_state.register_scores(list(detector.score_artifact(artifact).values()))

    flagged_windows = detect_anomalies(
        detector=detector,
        artifacts=test_artifacts,
        threshold=training_summary.learned_threshold,
    )
    return BaselineExperimentResult(
        training_summary=training_summary,
        model_state=model_state,
        graph_stats=summarize_graph_windows(snapshot_dataset),
        test_snapshots=test_snapshots,
        flagged_windows=flagged_windows,
    )


# ---------------------------------------------------------------------------
# GNN experiment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GNNExperimentResult:
    """Result of a full GNN training + inference run."""
    epochs: int
    final_loss: float
    learned_threshold: float
    checkpoint_path: str
    graph_stats: dict[str, float]
    test_snapshots: list[GraphSnapshot]
    flagged_windows: list[dict[str, float]]


def run_gnn_experiment(
    snapshot_dataset: WindowedGraphDataset,
    artifact_dataset: ArtifactGraphDataset,
    benign_ratio: float = 0.7,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    threshold_k: float = 3.0,
    checkpoint_path: str = "artifacts/models/gnn_detector.pt",
) -> GNNExperimentResult:
    """
    End-to-end GNN experiment:
      1. Split windows into benign (train) / test.
      2. Train the temporal GAT encoder-decoder on benign windows.
      3. Calibrate anomaly threshold on the training scores.
      4. Score all test windows with the trained GNN.
      5. Flag edges above the threshold.

    Requires torch and torch-geometric to be installed.
    On machines where they are unavailable, use run_baseline_experiment()
    as a drop-in replacement.

    Parameters
    ----------
    snapshot_dataset : WindowedGraphDataset — for test snapshot metadata.
    artifact_dataset : ArtifactGraphDataset — tensor features, used for GNN.
    benign_ratio     : fraction of earliest windows used for training.
    epochs           : training epochs.
    learning_rate    : Adam learning rate.
    threshold_k      : sigma multiplier for threshold (mean + k * stdev).
    checkpoint_path  : where to save the trained model weights.

    Returns
    -------
    GNNExperimentResult with flagged_windows matching the same structure as
    BaselineExperimentResult — downstream code (subgraph extraction, API,
    dashboard) works identically regardless of which experiment was run.
    """
    from .gnn import GNNModelConfig
    from .gnn_train import GNNTrainingConfig, train_gnn_detector
    from .gnn_inference import load_gnn_detector, score_windows

    _, test_snapshots = snapshot_dataset.train_test_split(benign_ratio=benign_ratio)
    train_artifacts, test_artifacts = artifact_dataset.train_test_split(benign_ratio=benign_ratio)

    # 1. Train
    training_summary = train_gnn_detector(
        artifacts=train_artifacts,
        model_config=GNNModelConfig(),
        training_config=GNNTrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            threshold_k=threshold_k,
            checkpoint_path=checkpoint_path,
        ),
    )

    # 2. Load checkpoint and score test windows
    detector = load_gnn_detector(checkpoint_path)
    test_scores = score_windows(detector, test_artifacts)

    # 3. Flag edges above learned threshold
    threshold = training_summary.learned_threshold
    flagged_windows = [flag_scores(scores, threshold) for scores in test_scores]

    return GNNExperimentResult(
        epochs=training_summary.epochs,
        final_loss=training_summary.final_loss,
        learned_threshold=threshold,
        checkpoint_path=training_summary.checkpoint_path,
        graph_stats=summarize_graph_windows(snapshot_dataset),
        test_snapshots=test_snapshots,
        flagged_windows=flagged_windows,
    )
