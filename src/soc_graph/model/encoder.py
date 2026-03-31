from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderConfig:
    input_dim: int = 6
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    heads: int = 2
    edge_dim: int = 8


def build_gnn_encoder(config: EncoderConfig):
    """
    Build the temporal GAT encoder.

    Architecture per window
    -----------------------
    1. Linear projection: node_features (input_dim) → hidden_dim
    2. N × GATConv layers with edge features
    3. GRU temporal update: h_i^t = GRU(h_i^{t-1}, z_i^t)
       where h_i is a persistent hidden state vector per node, keyed by node_id.

    The GRU cell is called once per node that appears in the current window.
    Nodes not seen in a window keep their previous hidden state unchanged.

    Usage
    -----
    encoder = build_gnn_encoder(config)
    memory  = {}                        # node_id -> hidden state tensor
    for data in windows:
        z, memory = encoder(data.x, data.edge_index, data.edge_attr,
                            node_ids=data.node_ids, memory=memory)
    """
    import torch
    from torch import nn
    from torch_geometric.nn import GATConv

    class TemporalGraphEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)

            self.gat_layers = nn.ModuleList()
            for _ in range(config.num_layers):
                self.gat_layers.append(
                    GATConv(
                        in_channels=config.hidden_dim,
                        out_channels=config.hidden_dim // config.heads,
                        heads=config.heads,
                        dropout=config.dropout,
                        edge_dim=config.edge_dim,
                        concat=True,   # output dim = hidden_dim
                    )
                )

            # GRU: input = GAT output (hidden_dim), hidden = memory vector (hidden_dim)
            self.gru = nn.GRUCell(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
            )

            self.dropout = nn.Dropout(config.dropout)
            self.activation = nn.ReLU()

        def forward(
            self,
            x: "torch.Tensor",
            edge_index: "torch.Tensor",
            edge_attr: "torch.Tensor",
            node_ids: list[str] | None = None,
            memory: dict[str, "torch.Tensor"] | None = None,
        ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
            """
            Parameters
            ----------
            x           : (N, input_dim) node features for this window
            edge_index  : (2, E) edge connectivity
            edge_attr   : (E, edge_dim) edge features
            node_ids    : list of N string node IDs (same order as x rows)
            memory      : {node_id: hidden_state_tensor(hidden_dim)}
                          Pass an empty dict on the first window; the dict is
                          updated in-place and returned for the next window.

            Returns
            -------
            z      : (N, hidden_dim) updated node embeddings
            memory : updated memory dict (same object, mutated)
            """
            if memory is None:
                memory = {}

            # 1. Project raw features
            h = self.input_projection(x)

            # 2. Message-passing layers
            for layer in self.gat_layers:
                h = layer(h, edge_index, edge_attr=edge_attr)
                h = self.activation(h)
                h = self.dropout(h)

            # 3. GRU temporal update — one cell step per node
            if node_ids is not None:
                updated_rows: list[int] = []
                prev_hidden: list["torch.Tensor"] = []

                for i, nid in enumerate(node_ids):
                    if nid in memory:
                        updated_rows.append(i)
                        prev_hidden.append(memory[nid])

                if updated_rows:
                    idx = torch.tensor(updated_rows, dtype=torch.long, device=x.device)
                    h_prev = torch.stack(prev_hidden).to(x.device)
                    h_in   = h[idx]
                    h_new  = self.gru(h_in, h_prev)
                    # Scatter the GRU outputs back into h
                    h = h.clone()
                    h[idx] = h_new

                # Store all nodes' updated embeddings as the new memory
                for i, nid in enumerate(node_ids):
                    memory[nid] = h[i].detach()

            return h, memory

    return TemporalGraphEncoder()
