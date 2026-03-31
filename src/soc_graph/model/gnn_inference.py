from __future__ import annotations

"""
GNN inference: load a trained checkpoint and produce per-edge anomaly scores.

The anomaly score for an edge e is:

    score(e) = -log P(e | model)
             = BCE(logit_e, label=1)

A high score means the model did not expect this edge given the learned
representation of normal system behaviour.

Usage
-----
    from soc_graph.model.gnn_inference import load_gnn_detector, score_windows

    detector = load_gnn_detector("artifacts/models/gnn_detector.pt")
    scored   = score_windows(detector, test_artifacts)
    # scored: list[dict[str, float]]  — one dict per window, edge_key -> score
"""

from dataclasses import dataclass
from pathlib import Path

from soc_graph.data.schemas import GraphTensorArtifact


@dataclass
class GNNDetector:
    """Thin wrapper around a loaded GNN checkpoint."""
    model: object          # GraphEdgeAutoencoder (torch.nn.Module)
    learned_threshold: float
    model_config: dict


def load_gnn_detector(checkpoint_path: str | Path) -> GNNDetector:
    """
    Load a GNN detector from a checkpoint saved by ``train_gnn_detector``.

    Parameters
    ----------
    checkpoint_path : path to the .pt file written during training.

    Returns
    -------
    GNNDetector ready for use with ``score_windows``.
    """
    import torch
    from soc_graph.model.encoder import EncoderConfig
    from soc_graph.model.decoder import DecoderConfig
    from soc_graph.model.gnn import GNNModelConfig, build_gnn_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    enc_cfg = EncoderConfig(**checkpoint["model_config"]["encoder"])
    dec_cfg = DecoderConfig(**checkpoint["model_config"]["decoder"])
    model_config = GNNModelConfig(encoder=enc_cfg, decoder=dec_cfg)

    model = build_gnn_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    learned_threshold = float(checkpoint.get("learned_threshold", 0.0))

    return GNNDetector(
        model=model,
        learned_threshold=learned_threshold,
        model_config=checkpoint["model_config"],
    )


def score_windows(
    detector: GNNDetector,
    artifacts: list[GraphTensorArtifact],
) -> list[dict[str, float]]:
    """
    Score a sequence of time-window artifacts with the trained GNN.

    Windows are processed in order so the GRU memory evolves chronologically
    (matching the training setup).

    Parameters
    ----------
    detector  : loaded GNNDetector from ``load_gnn_detector``.
    artifacts : list of GraphTensorArtifacts to score (typically the test set).

    Returns
    -------
    List of dicts, one per window: {edge_key: anomaly_score}.
    Empty windows produce an empty dict.
    """
    import torch
    import torch.nn.functional as F
    from soc_graph.data.pyg import artifact_to_pyg_data

    model = detector.model
    results: list[dict[str, float]] = []
    memory: dict = {}

    with torch.no_grad():
        for artifact in artifacts:
            if artifact.num_edges == 0:
                results.append({})
                continue

            data = artifact_to_pyg_data(artifact)
            z, pos_logits, memory = model(data, memory=memory)

            # BCE(logit, 1) = -log sigmoid(logit) = anomaly score per edge
            scores = F.binary_cross_entropy_with_logits(
                pos_logits,
                torch.ones_like(pos_logits),
                reduction="none",
            ).cpu().tolist()

            results.append(
                {key: score for key, score in zip(artifact.edge_keys, scores)}
            )

    return results


def score_artifact(
    detector: GNNDetector,
    artifact: GraphTensorArtifact,
    memory: dict | None = None,
) -> tuple[dict[str, float], dict]:
    """
    Score a single artifact, returning both the scores and the updated memory.

    Useful when integrating the GNN into a streaming / online setting where
    you manage the memory dict yourself.

    Returns
    -------
    (scores_dict, updated_memory)
    """
    import torch
    import torch.nn.functional as F
    from soc_graph.data.pyg import artifact_to_pyg_data

    if memory is None:
        memory = {}

    if artifact.num_edges == 0:
        return {}, memory

    model = detector.model
    data = artifact_to_pyg_data(artifact)

    with torch.no_grad():
        z, pos_logits, memory = model(data, memory=memory)

    scores = F.binary_cross_entropy_with_logits(
        pos_logits,
        torch.ones_like(pos_logits),
        reduction="none",
    ).cpu().tolist()

    scores_dict = {key: score for key, score in zip(artifact.edge_keys, scores)}
    return scores_dict, memory
