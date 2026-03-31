from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TemporalModelState:
    """
    Minimal state container for the evolving graph model pipeline.

    The future GNN will keep learned node memories here. For now, we track
    calibration state and per-window summaries so training, scoring, and
    reporting are already separated cleanly.
    """

    learned_threshold: float | None = None
    seen_windows: int = 0
    edge_score_history: list[float] = field(default_factory=list)
    known_edge_profiles: dict[str, float] = field(default_factory=dict)
    known_signatures: dict[str, int] = field(default_factory=dict)

    def register_scores(self, scores: list[float]) -> None:
        self.seen_windows += 1
        self.edge_score_history.extend(scores)

    def register_profiles(self, edge_profiles: dict[str, float], signatures: dict[str, int]) -> None:
        self.known_edge_profiles.update(edge_profiles)
        self.known_signatures.update(signatures)
