from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soc_graph.data.schemas import GraphTensorArtifact

from .gnn import GNNModelConfig, artifact_batch_to_pyg, build_gnn_model


@dataclass(frozen=True)
class GNNTrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    negative_ratio: float = 1.0
    threshold_k: float = 3.0
    checkpoint_path: str = "artifacts/models/gnn_detector.pt"


@dataclass(frozen=True)
class GNNTrainingSummary:
    epochs: int
    final_loss: float
    loss_history: list[float]
    checkpoint_path: str
    learned_threshold: float


def train_gnn_detector(
    artifacts: list[GraphTensorArtifact],
    model_config: GNNModelConfig | None = None,
    training_config: GNNTrainingConfig | None = None,
) -> GNNTrainingSummary:
    """
    Train the temporal GNN on benign provenance windows.

    Key design choices
    ------------------
    - Windows are fed in chronological order so the GRU memory is meaningful.
    - Negative edges use zero-filled edge attributes (not sliced real attributes).
    - Anomaly scores on the training set are collected to calibrate a threshold.
    - The full model state + threshold are saved to the checkpoint.
    """
    import torch
    import torch.nn.functional as F
    from statistics import mean, pstdev
    from torch_geometric.utils import negative_sampling

    if not artifacts:
        raise ValueError("artifacts must not be empty")

    model_config = model_config or GNNModelConfig()
    training_config = training_config or GNNTrainingConfig()

    model = build_gnn_model(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    edge_dim = model_config.encoder.edge_dim
    pyg_windows = artifact_batch_to_pyg(artifacts)

    final_loss = 0.0
    loss_history: list[float] = []

    for epoch in range(training_config.epochs):
        memory: dict = {}   # reset memory at the start of each epoch
        epoch_losses: list[float] = []

        for data in pyg_windows:
            optimizer.zero_grad()

            z, pos_logits, memory = model(data, memory=memory)

            num_neg = max(1, int(data.edge_index.size(1) * training_config.negative_ratio))
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.x.size(0),
                num_neg_samples=num_neg,
            )

            # Negative edges don't exist — give them zero edge features, not
            # borrowed real features.  Using real features would leak ground-truth
            # edge-type information into the negative samples and confuse training.
            neg_edge_attr = torch.zeros(
                neg_edge_index.size(1), edge_dim, dtype=torch.float32
            )

            neg_logits = model.decoder(z, neg_edge_index, neg_edge_attr)

            pos_labels = torch.ones(pos_logits.size(0), dtype=torch.float32)
            neg_labels = torch.zeros(neg_logits.size(0), dtype=torch.float32)

            loss = (
                F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
                + F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
            )
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
            epoch_losses.append(final_loss)

        # Detach memory at epoch boundary to avoid accumulating graph history
        memory = {k: v.detach() for k, v in memory.items()}
        if epoch_losses:
            loss_history.append(sum(epoch_losses) / len(epoch_losses))
        else:
            loss_history.append(final_loss)

    # ------------------------------------------------------------------
    # Threshold calibration: score every training window once more with
    # the trained model (in eval mode, no grad) and apply sigma method.
    # ------------------------------------------------------------------
    model.eval()
    all_scores: list[float] = []
    with torch.no_grad():
        memory = {}
        for data in pyg_windows:
            z, pos_logits, memory = model(data, memory=memory)
            # Anomaly score = -log sigmoid(logit) = -log P(edge exists)
            # = binary cross-entropy against label=1
            scores = F.binary_cross_entropy_with_logits(
                pos_logits,
                torch.ones_like(pos_logits),
                reduction="none",
            )
            all_scores.extend(scores.cpu().tolist())

    if len(all_scores) >= 2:
        mu = mean(all_scores)
        sigma = pstdev(all_scores)
        learned_threshold = mu + training_config.threshold_k * sigma
    else:
        learned_threshold = float(all_scores[0]) if all_scores else 0.0

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    checkpoint_path = Path(training_config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "encoder": model_config.encoder.__dict__,
                "decoder": model_config.decoder.__dict__,
            },
            "training_config": training_config.__dict__,
            "learned_threshold": learned_threshold,
            "loss_history": loss_history,
            "final_loss": final_loss,
        },
        checkpoint_path,
    )

    return GNNTrainingSummary(
        epochs=training_config.epochs,
        final_loss=final_loss,
        loss_history=loss_history,
        checkpoint_path=str(checkpoint_path),
        learned_threshold=learned_threshold,
    )
