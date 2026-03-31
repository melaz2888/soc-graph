from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean

from soc_graph.data.schemas import GraphSnapshot, GraphTensorArtifact


class BaselineAnomalyDetector:
    """
    Baseline structural scoring before the full GNN arrives.

    This keeps the pipeline runnable and testable while making it explicit
    that the real graph model is still pending.
    """

    def score_snapshot(self, snapshot: GraphSnapshot) -> dict[str, float]:
        if not snapshot.edges:
            return {}
        avg_count = mean(edge.count for edge in snapshot.edges)
        return {
            f"{edge.source_id}:{edge.edge_type.value}:{edge.target_id}": edge.count / avg_count
            for edge in snapshot.edges
        }

    def score_artifact(self, artifact: GraphTensorArtifact) -> dict[str, float]:
        if artifact.num_edges == 0:
            return {}
        avg_count = sum(artifact.edge_counts) / len(artifact.edge_counts)
        node_activity = [features[3] + features[4] for features in artifact.node_features]

        scores: dict[str, float] = {}
        for edge_index, edge_key in enumerate(artifact.edge_keys):
            src_idx = artifact.edge_index[0][edge_index]
            dst_idx = artifact.edge_index[1][edge_index]
            structural_bias = float(node_activity[src_idx] + node_activity[dst_idx]) / max(artifact.num_nodes, 1)
            count_score = artifact.edge_counts[edge_index] / max(avg_count, 1e-6)
            scores[edge_key] = count_score + structural_bias / 10.0
        return scores


@dataclass(frozen=True)
class EdgeProfile:
    mean_count: float
    seen_windows: int


class BehavioralAnomalyDetector:
    """
    Learned benign-behavior detector.

    This is not the final GNN, but it is a genuine trainable graph model:
    it learns which provenance edge patterns appear in benign windows and
    scores later windows by how surprising those edge signatures and counts are.
    """

    SIGNATURE_SPACE_SIZE = 3 * 7 * 3

    def __init__(
        self,
        smoothing: float = 1.0,
        novelty_weight: float = 2.0,
        deviation_weight: float = 1.0,
    ) -> None:
        self.smoothing = smoothing
        self.novelty_weight = novelty_weight
        self.deviation_weight = deviation_weight
        self.edge_profiles: dict[str, EdgeProfile] = {}
        self.signature_counts: dict[str, int] = {}
        self.total_signatures = 0

    def fit(self, artifacts: list[GraphTensorArtifact]) -> None:
        edge_history: dict[str, list[float]] = {}
        signature_counts: dict[str, int] = {}

        for artifact in artifacts:
            for edge_index, edge_key in enumerate(artifact.edge_keys):
                signature = self._signature_for_edge(artifact, edge_index)
                signature_counts[signature] = signature_counts.get(signature, 0) + 1
                edge_history.setdefault(edge_key, []).append(artifact.edge_counts[edge_index])

        self.signature_counts = signature_counts
        self.total_signatures = sum(signature_counts.values())
        self.edge_profiles = {
            edge_key: EdgeProfile(mean_count=sum(counts) / len(counts), seen_windows=len(counts))
            for edge_key, counts in edge_history.items()
        }

    def score_artifact(self, artifact: GraphTensorArtifact) -> dict[str, float]:
        scores: dict[str, float] = {}
        normalizer = self.total_signatures + self.smoothing * self.SIGNATURE_SPACE_SIZE

        for edge_index, edge_key in enumerate(artifact.edge_keys):
            signature = self._signature_for_edge(artifact, edge_index)
            signature_count = self.signature_counts.get(signature, 0)
            signature_probability = (signature_count + self.smoothing) / max(normalizer, self.smoothing)
            signature_surprise = -math.log(signature_probability)

            profile = self.edge_profiles.get(edge_key)
            observed_count = artifact.edge_counts[edge_index]
            if profile is None:
                count_deviation = observed_count
                novelty_bonus = self.novelty_weight
            else:
                count_deviation = abs(observed_count - profile.mean_count) / max(profile.mean_count, 1.0)
                novelty_bonus = 0.0

            scores[edge_key] = signature_surprise + self.deviation_weight * count_deviation + novelty_bonus

        return scores

    def profile_snapshot(self) -> tuple[dict[str, float], dict[str, int]]:
        return (
            {edge_key: profile.mean_count for edge_key, profile in self.edge_profiles.items()},
            dict(self.signature_counts),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "smoothing": self.smoothing,
            "novelty_weight": self.novelty_weight,
            "deviation_weight": self.deviation_weight,
            "signature_counts": dict(self.signature_counts),
            "total_signatures": self.total_signatures,
            "edge_profiles": {
                edge_key: {
                    "mean_count": profile.mean_count,
                    "seen_windows": profile.seen_windows,
                }
                for edge_key, profile in self.edge_profiles.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BehavioralAnomalyDetector":
        detector = cls(
            smoothing=float(payload.get("smoothing", 1.0)),
            novelty_weight=float(payload.get("novelty_weight", 2.0)),
            deviation_weight=float(payload.get("deviation_weight", 1.0)),
        )
        edge_profiles_payload = payload.get("edge_profiles", {})
        if isinstance(edge_profiles_payload, dict):
            detector.edge_profiles = {
                edge_key: EdgeProfile(
                    mean_count=float(profile["mean_count"]),
                    seen_windows=int(profile["seen_windows"]),
                )
                for edge_key, profile in edge_profiles_payload.items()
                if isinstance(profile, dict)
            }
        signature_counts = payload.get("signature_counts", {})
        if isinstance(signature_counts, dict):
            detector.signature_counts = {
                str(signature): int(count) for signature, count in signature_counts.items()
            }
        detector.total_signatures = int(payload.get("total_signatures", sum(detector.signature_counts.values())))
        return detector

    @staticmethod
    def _node_type_label(node_features: list[float]) -> str:
        if not node_features:
            return "UNKNOWN"
        labels = ("PROCESS", "FILE", "SOCKET")
        label_index = max(range(min(3, len(node_features))), key=lambda idx: node_features[idx])
        return labels[label_index]

    @staticmethod
    def _edge_type_label(edge_features: list[float]) -> str:
        if not edge_features:
            return "UNKNOWN"
        labels = ("READ", "WRITE", "EXECUTE", "CONNECT", "SEND", "RECV", "FORK")
        label_index = max(range(min(7, len(edge_features))), key=lambda idx: edge_features[idx])
        return labels[label_index]

    def _signature_for_edge(self, artifact: GraphTensorArtifact, edge_index: int) -> str:
        src_idx = artifact.edge_index[0][edge_index]
        dst_idx = artifact.edge_index[1][edge_index]
        src_type = self._node_type_label(artifact.node_features[src_idx])
        dst_type = self._node_type_label(artifact.node_features[dst_idx])
        edge_type = self._edge_type_label(artifact.edge_features[edge_index])
        return f"{src_type}:{edge_type}:{dst_type}"
