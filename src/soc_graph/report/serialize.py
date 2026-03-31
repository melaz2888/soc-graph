from __future__ import annotations

from soc_graph.data.schemas import GraphSnapshot
from soc_graph.detection.subgraph import CandidateAttackSubgraph, ReducedAttackSubgraph


def serialize_alert_subgraph(
    snapshot: GraphSnapshot,
    candidate: CandidateAttackSubgraph,
) -> dict[str, object]:
    """Serialize a CandidateAttackSubgraph (backward-compatible)."""
    nodes = [
        {"id": node.node_id, "type": node.node_type.value, "name": node.name}
        for node in sorted(snapshot.nodes.values(), key=lambda node: node.node_id)
    ]
    flagged_key_set = set(candidate.edge_keys)
    edges = [
        {
            "key": f"{edge.source_id}:{edge.edge_type.value}:{edge.target_id}",
            "src": edge.source_id,
            "dst": edge.target_id,
            "type": edge.edge_type.value,
            "count": edge.count,
        }
        for edge in snapshot.edges
        if f"{edge.source_id}:{edge.edge_type.value}:{edge.target_id}" in flagged_key_set
    ]
    return {
        "alert_id": candidate.alert_id,
        "window_start": snapshot.window_start.isoformat(),
        "window_end": snapshot.window_end.isoformat(),
        "flagged_edge_count": candidate.flagged_edge_count,
        "total_edge_count": candidate.total_edge_count,
        "nodes": nodes,
        "edges": edges,
    }


def serialize_reduced_subgraph(subgraph: ReducedAttackSubgraph) -> dict[str, object]:
    """
    Serialize a ReducedAttackSubgraph to a JSON-compatible dict.

    This is the preferred format for the API and LLM report — it includes
    only nodes/edges involved in anomalous activity, plus per-edge anomaly
    scores and the number of connected components.
    """
    nodes = [
        {"id": node.node_id, "type": node.node_type.value, "name": node.name}
        for node in sorted(subgraph.nodes.values(), key=lambda n: n.node_id)
    ]
    edges = []
    for edge in subgraph.edges:
        key = f"{edge.source_id}:{edge.edge_type.value}:{edge.target_id}"
        # Look up node types for MITRE mapping hints
        src_node = subgraph.nodes.get(edge.source_id)
        dst_node = subgraph.nodes.get(edge.target_id)
        edges.append({
            "key": key,
            "src": edge.source_id,
            "src_type": src_node.node_type.value if src_node else "",
            "dst": edge.target_id,
            "dst_type": dst_node.node_type.value if dst_node else "",
            "type": edge.edge_type.value,
            "count": edge.count,
            "anomaly_score": round(subgraph.anomaly_scores.get(key, 0.0), 4),
            "first_seen": edge.first_seen.isoformat(),
            "last_seen": edge.last_seen.isoformat(),
        })

    return {
        "alert_id": subgraph.alert_id,
        "flagged_edge_count": subgraph.flagged_edge_count,
        "total_edge_count": subgraph.total_edge_count,
        "component_count": subgraph.component_count,
        "nodes": nodes,
        "edges": edges,
    }
