from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from .build_graph import build_graph_artifacts, build_snapshot, snapshot_to_graph_tensor
from .schemas import GraphSnapshot, GraphTensorArtifact, ProvenanceEvent


@dataclass
class WindowedGraphDataset:
    snapshots: list[GraphSnapshot]

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, index: int) -> GraphSnapshot:
        return self.snapshots[index]

    def train_test_split(self, benign_ratio: float = 0.7) -> tuple[list[GraphSnapshot], list[GraphSnapshot]]:
        if not 0 < benign_ratio < 1:
            raise ValueError("benign_ratio must be between 0 and 1")
        cutoff = max(1, int(len(self.snapshots) * benign_ratio))
        return self.snapshots[:cutoff], self.snapshots[cutoff:]

    def to_artifacts(self) -> list[GraphTensorArtifact]:
        return [snapshot_to_graph_tensor(snapshot) for snapshot in self.snapshots]


@dataclass
class ArtifactGraphDataset:
    artifacts: list[GraphTensorArtifact]

    def __len__(self) -> int:
        return len(self.artifacts)

    def __getitem__(self, index: int) -> GraphTensorArtifact:
        return self.artifacts[index]

    def train_test_split(
        self,
        benign_ratio: float = 0.7,
    ) -> tuple[list[GraphTensorArtifact], list[GraphTensorArtifact]]:
        if not 0 < benign_ratio < 1:
            raise ValueError("benign_ratio must be between 0 and 1")
        cutoff = max(1, int(len(self.artifacts) * benign_ratio))
        return self.artifacts[:cutoff], self.artifacts[cutoff:]


def build_datasets(
    events: list[ProvenanceEvent],
    window: timedelta,
) -> tuple[WindowedGraphDataset, ArtifactGraphDataset]:
    snapshots = build_snapshot(events, window=window)
    artifacts = build_graph_artifacts(events, window=window)
    return WindowedGraphDataset(snapshots=snapshots), ArtifactGraphDataset(artifacts=artifacts)
