from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderConfig:
    hidden_dim: int = 64
    edge_feature_dim: int = 8
    mlp_hidden_dim: int = 64


def build_edge_decoder(config: DecoderConfig):
    import torch
    from torch import nn

    class EdgePredictionDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_dim * 2 + config.edge_feature_dim, config.mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.mlp_hidden_dim, 1),
            )

        def forward(self, z, edge_index, edge_attr):
            src = z[edge_index[0]]
            dst = z[edge_index[1]]
            features = torch.cat([src, dst, edge_attr], dim=-1)
            return self.mlp(features).squeeze(-1)

    return EdgePredictionDecoder()

