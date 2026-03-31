from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from soc_graph.data.schemas import GraphTensorArtifact
from soc_graph.detection.threshold import flag_scores, sigma_threshold

from .detector import BehavioralAnomalyDetector


@dataclass(frozen=True)
class TrainingSummary:
    num_windows: int
    mean_edges_per_window: float
    learned_threshold: float
    benign_score_count: int


def fit_detector(
    detector: BehavioralAnomalyDetector,
    benign_artifacts: list[GraphTensorArtifact],
    threshold_k: float = 3.0,
) -> TrainingSummary:
    if not benign_artifacts:
        raise ValueError("benign_artifacts must not be empty")

    detector.fit(benign_artifacts)
    benign_scores: list[float] = []
    for artifact in benign_artifacts:
        benign_scores.extend(detector.score_artifact(artifact).values())

    threshold = sigma_threshold(benign_scores, k=threshold_k)
    mean_edges_per_window = mean(artifact.num_edges for artifact in benign_artifacts)
    return TrainingSummary(
        num_windows=len(benign_artifacts),
        mean_edges_per_window=mean_edges_per_window,
        learned_threshold=threshold,
        benign_score_count=len(benign_scores),
    )


def detect_anomalies(
    detector: BehavioralAnomalyDetector,
    artifacts: list[GraphTensorArtifact],
    threshold: float,
) -> list[dict[str, float]]:
    return [flag_scores(detector.score_artifact(artifact), threshold) for artifact in artifacts]


def fit_baseline_detector(
    detector: BehavioralAnomalyDetector,
    benign_artifacts: list[GraphTensorArtifact],
    threshold_k: float = 3.0,
) -> TrainingSummary:
    return fit_detector(detector, benign_artifacts, threshold_k=threshold_k)
