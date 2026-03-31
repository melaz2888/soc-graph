from __future__ import annotations

from dataclasses import dataclass

from soc_graph.data.pyg import artifact_to_pyg_data
from soc_graph.data.schemas import GraphTensorArtifact

from .decoder import DecoderConfig, build_edge_decoder
from .encoder import EncoderConfig, build_gnn_encoder


@dataclass(frozen=True)
class GNNModelConfig:
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()


def build_gnn_model(config: GNNModelConfig):
    """
    Build the GraphEdgeAutoencoder (encoder + decoder).

    The model's forward() pass accepts an optional `memory` dict so the
    temporal GRU state can be threaded across consecutive windows.
    """
    import torch
    from torch import nn

    class GraphEdgeAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = build_gnn_encoder(config.encoder)
            self.decoder = build_edge_decoder(config.decoder)

        def forward(self, data, memory=None):
            """
            Parameters
            ----------
            data   : PyG Data object with .x, .edge_index, .edge_attr, .node_ids
            memory : {node_id: hidden_state} dict (updated in-place, returned)

            Returns
            -------
            z         : (N, hidden_dim) node embeddings for this window
            logits    : (E,) edge prediction logits (positive edges)
            memory    : updated memory dict
            """
            if memory is None:
                memory = {}

            node_ids = getattr(data, "node_ids", None)
            z, memory = self.encoder(
                data.x, data.edge_index, data.edge_attr,
                node_ids=node_ids, memory=memory,
            )
            logits = self.decoder(z, data.edge_index, data.edge_attr)
            return z, logits, memory

    return GraphEdgeAutoencoder()


def artifact_batch_to_pyg(artifacts: list[GraphTensorArtifact]):
    """Convert a list of GraphTensorArtifacts to PyG Data objects (no batching)."""
    return [artifact_to_pyg_data(a) for a in artifacts]
