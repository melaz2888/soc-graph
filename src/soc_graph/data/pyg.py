from __future__ import annotations

from soc_graph.data.schemas import GraphTensorArtifact


def artifact_to_pyg_data(artifact: GraphTensorArtifact):
    import torch
    from torch_geometric.data import Data

    x = torch.tensor(artifact.node_features, dtype=torch.float32)
    edge_index = torch.tensor(artifact.edge_index, dtype=torch.long)
    edge_attr = torch.tensor(artifact.edge_features, dtype=torch.float32)
    edge_count = torch.tensor(artifact.edge_counts, dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_count=edge_count,
        edge_keys=artifact.edge_keys,
        node_ids=artifact.node_ids,
    )

